import torch, os, torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stylegan import z_sample
from stylegan2_pytorch import model as stylegan_model
from nethook import InstrumentedModel
from dissect import collect_stack, stacked_map#GeneratorSegRunner,
from FF import Feedforward, MultiClassFF
from kmeans import kmean_viz

def regression(outdir, test_file, fname, model, raw_noise, given_seg=None, eva_im=None, eva_seg=None):
    pt = os.path.join(outdir, fname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(test_file)
        # os.makedirs('./results/img/'+args.model_n+'/train/')
    writer = SummaryWriter(outdir)

    w = z_sample(1, seed=3).unsqueeze(0)
    random_z = TensorDataset(w)
    test_img = torch.utils.data.DataLoader(random_z, batch_size=1)

    stylegan2_path = './stylegan2_pytorch/checkpoint/stylegan2-ffhq-config-f.pt'
    stylegan_state_dict = torch.load(stylegan2_path)
    models = stylegan_model.Generator(size=1024, style_dim=512, n_mlp=8, input_is_Wlatent=False)
    models.load_state_dict(stylegan_state_dict['g_ema'], strict=False)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()
    models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
                            'convs.5','convs.6','convs.7','convs.8','convs.9',
                            'convs.10','convs.11','convs.12','convs.13','convs.14',
                            'convs.15',])
    stylegan_stack = collect_stack(512, models, test_img)


    with torch.no_grad():
        noise_dataset = torch.utils.data.DataLoader(raw_noise,
                batch_size=1, num_workers=0, pin_memory=False)

        # Seg #
        seg_flat = np.reshape(given_seg, (512*512, -1)) #[512*512, 4]
        stack = collect_stack(512, model, noise_dataset)[0] #[total_c, h, w]

        num_chan = stack.shape[0]
        stack = stack.reshape(num_chan ,-1).T #[H*W, chan]

        seg_flat = torch.LongTensor(seg_flat)
        _,seg_flat = seg_flat.max(dim=1) #[512*512, 1]

        batch_size = 512
        trainDataset = TensorDataset(torch.FloatTensor(stack), torch.LongTensor(seg_flat))
        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=True)

    lr_rate = 0.001
    iterations = 100

    ## Model
    #reg_model = Feedforward(num_chan, 200).cuda()
    hidden_list = [2000]
    reg_model = MultiClassFF(num_chan, hidden_list, 4).cuda()

    ## Loss
    #criterion = FocalLoss().cuda()
    criterion = torch.nn.NLLLoss().cuda()

    optimizer = torch.optim.Adam(reg_model.parameters(), lr=lr_rate, weight_decay=0)
    decayRate = 0.96
    my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

    #for param in reg_model.parameters():
    #    print('Parameter shape = ', param.shape)
    #torch.autograd.set_detect_anomaly(True)

    for step in range(iterations):
        print('Epoch {} / {}'.format(step, iterations))
        total_loss = 0
        for (data, target) in trainLoader:
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()

            prediction = reg_model(data)
            # print(prediction.shape)
            # print(target.shape)
            # assert False

            loss = criterion(prediction, target)
            #loss = weighted_binary_cross_entropy(prediction, target, weights=None)

            total_loss+=loss
            loss.backward()
            optimizer.step()
        # print(prediction[0])

        print('Batch_Loss: ', total_loss.item()/batch_size)

        # Decay every 50 epoch
        if step%10 == 0 and step!=0 :
            my_lr_scheduler.step()
            for param_group in optimizer.param_groups:
                print('Learning rate = ', param_group['lr'])
            writer.add_scalar('training loss', total_loss.item()/batch_size, step)

            torch.save(reg_model.state_dict(), pt)

            combined = stacked_map(stylegan_stack, pt, hidden_list)
            k_im = kmean_viz(combined, 512)
            Image.fromarray((k_im).astype(np.uint8)).resize([1024,1024]).save(test_file+"im_{:03d}.png".format(step), optimize=True, quality=80)
    # with torch.no_grad():
    #     pt = os.path.join(outdir, fname)
    #     torch.save(reg_model.state_dict(), pt)

def weighted_binary_cross_entropy(output, target, weights=None):
    eps = 1e-10
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * (target * torch.log(output+eps)) + \
               weights[0] * ((1 - target) * torch.log(1 - output+eps))
    else:
        loss = target * torch.log(output+eps) + (1 - target) * torch.log(1 - output+eps)

    return -(torch.mean(loss))

class MaxLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.eye_line = torch.nn.Parameter(torch.Tensor([1.]))
        self.bg_line = torch.nn.Parameter(torch.Tensor([0.]))

    @staticmethod
    def distance(x,y):
        d = torch.norm(x-y, dim=1, keepdim=True)
        return d

    def forward(self, y_pred, target):
        margin = 0.2

        pos_loss = F.relu( self.distance(y_pred, self.eye_line) - self.distance(y_pred, self.bg_line)+margin )
        neg_loss = F.relu( self.distance(y_pred, self.bg_line) - self.distance(y_pred, self.eye_line)+margin )
        loss = torch.mm(target.t(), pos_loss) + torch.mm((1-target).t(), neg_loss)

        return loss/target.shape[0]

class ContrastiveLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, target):
        margin = 0.2
        pos_thr = 0.6
        neg_thr = 0.4

        pos_loss = F.relu(pos_thr - y_pred)
        neg_loss = F.relu(y_pred - neg_thr)
        loss = torch.mm(target.t(), pos_loss) + torch.mm((1-target).t(), neg_loss)

        return loss/target.shape[0]

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
