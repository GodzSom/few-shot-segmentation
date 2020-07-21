import torch, os, torchvision
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from stylegan import z_sample
from nethook import InstrumentedModel
from dissect import collect_stack, stacked_map_new#GeneratorSegRunner,
from CNN_models import dilated_CNN_101_up as CNN_net
# from unet_model import UNet as CNN_net
from kmeans import kmean_viz
from torchvision.utils import save_image

n_class = 6
img_size = 128

def cnn(outdir, test_file, fname, model, raw_noise, given_seg=None, eva_im=None, eva_seg=None):
    pt = os.path.join(outdir, fname)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
        os.makedirs(test_file)
        # os.makedirs('./results/img/'+args.model_n+'/train/')
    writer = SummaryWriter(outdir)


    with torch.no_grad():
        noise_dataset = torch.utils.data.DataLoader(raw_noise,
                batch_size=1, num_workers=0, pin_memory=False)

        # Seg #
        seg_flat = np.reshape(given_seg, (-1, img_size*img_size, n_class))#[batch, 512*512, 4]
        stack = collect_stack(img_size, model, noise_dataset)[0] #[batch, total_c, h, w]

        num_chan = stack.shape[0]
        stack = np.reshape(stack, (-1, 3008, img_size, img_size))

        seg_flat = torch.LongTensor(seg_flat)
        _,seg_flat = seg_flat.max(dim=2) #[batch, 512*512, 1]
        seg_flat = np.reshape(seg_flat, (-1, img_size*img_size)) #[batch, 512*512]

        batch_size = 1
        # print(stack.shape)
        # print(given_seg.shape)
        # assert False
        trainDataset = TensorDataset(torch.FloatTensor(stack), torch.LongTensor(seg_flat))
        trainLoader = torch.utils.data.DataLoader(dataset = trainDataset, batch_size=batch_size, shuffle=True, num_workers=10, pin_memory=False)

    lr_rate = 0.001
    iterations = 10000

    ## Model
    #reg_model = Feedforward(num_chan, 200).cuda()
    hidden_list = [2000]
    reg_model = CNN_net(n_class).cuda()

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
        total_nll = 0
        total_tv = 0
        for (data, target) in trainLoader:
            optimizer.zero_grad()
            data = data.cuda()
            target = target.cuda()
            print('------')
            print(data.shape)
            print(target.shape)

            prediction_ori = reg_model(data)
            prediction = torch.reshape(prediction_ori, (-1,n_class,img_size*img_size))

            nll = criterion(prediction, target)
            #loss = weighted_binary_cross_entropy(prediction, target, weights=None)
            tv = 1e-7 * (
                torch.sum(torch.abs(prediction_ori[:, :, :, :-1] - prediction_ori[:, :, :, 1:])) +
                torch.sum(torch.abs(prediction_ori[:, :, :-1, :] - prediction_ori[:, :, 1:, :])))

            loss = nll + tv

            total_loss += loss
            total_nll += nll
            total_tv += tv
            loss.backward()
            optimizer.step()
        # print(prediction[0])

        print('Batch_Loss: ', total_loss.item())

        # Decay every 50 epoch
        if step%50 == 0:# and step!=0 :
            my_lr_scheduler.step()
            for param_group in optimizer.param_groups:
                print('Learning rate = ', param_group['lr'])
            writer.add_scalar('training loss', total_loss.item()/batch_size, step)
            writer.add_scalar('nll', total_nll.item()/batch_size, step)
            writer.add_scalar('tv', total_tv.item()/batch_size, step)

            torch.save(reg_model.state_dict(), pt)
            # print('!!!'+str(torch.max(prediction_ori[0][0])))
            # print('!!!'+str(torch.max(prediction_ori)))
            # print('!!!'+str(torch.max(target)))
            # print('!!!'+str(torch.min(target)))
            # print(target)
            save_image(prediction_ori[0][0]/torch.max(prediction_ori[0][0]), './debug/img'+str(step)+'.png')

            # combined = stacked_map_new(stylegan_stack, reg_model)
            # k_im = kmean_viz(combined, 512)
            # Image.fromarray((k_im).astype(np.uint8)).resize([1024,1024]).save(test_file+"im_{:03d}.png".format(step), optimize=True, quality=80)
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
