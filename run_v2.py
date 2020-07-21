import torch, sys, os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import argparse
# from tqdm import tqdm
from stylegan2_pytorch_master2.bin.stylegan2_pytorch2 import train_from_folder

from nethook import InstrumentedModel
from dissect import dissect, collect_stack, VisualizeFeature#, GeneratorSegRunner
import stylegan
# from regression import regression
from cnn_v2 import cnn
# from stylegan2_pytorch import model

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_n', type=str, default='1', help='model_directory')
args = parser.parse_args()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
torch.cuda.empty_cache()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

stylegan2_path = './stylegan2_pytorch_master2/bin/models'
name = 'GANd'
mod_num = 61

# writer = SummaryWriter("./results/models/logs/"+args.model_di_n)

with torch.no_grad():
    '''
    Set input_is_Wlatent | True for W-latent , False for Z-latent
    '''
    # models = model.Generator(size=1024, style_dim=512, n_mlp=8, input_is_Wlatent=True).to(device)
    # models.load_state_dict(state_dict['g_ema'], strict=False)
    # models = InstrumentedModel(models)
    # models.eval()
    # models.cuda()

    models = train_from_folder(models_dir=stylegan2_path, name=name, get_model=True)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()

    # models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
    #                         'convs.5','convs.6','convs.7','convs.8','convs.9',
    #                         'convs.10','convs.11','convs.12','convs.13','convs.14',
    #                         'convs.15'])
    pf = 'blocks.'
    models.retain_layers([pf+'0.conv1', pf+'0.conv2', pf+'1.conv1', pf+'1.conv2',
                            pf+'2.conv1', pf+'2.conv2', pf+'3.conv1', pf+'3.conv2',
                            pf+'4.conv1', pf+'4.conv2', pf+'5.conv1', pf+'5.conv2'])

    # dict = models.state_dict()
    # for k,v in dict.items():
    #     print(k)
    # assert False


    '''
    Load Latent
    [1,1,512] for Z
    [1,18,512] for W+
    '''
    w_path = './noise_seed0.pt'
    w = torch.load(w_path)
    # w = torch.from_numpy(w).float().unsqueeze(0).cuda()

    ## For random Z-latent : Seed=None to random ##
    #w = stylegan.z_sample(1, seed=900).unsqueeze(0)

    given_w = TensorDataset(torch.cat((w[1][None], w[49][None], w[54][None]), 0).cuda())
    # print(w.shape)
    # assert False

    # Given Seg Mask

    mask_path = './masks/hands/'
    img_no = ['1', '49', '54']

    train_mask = np.zeros((3, 128, 128, 6))

    for i, im_n in enumerate(img_no):
        total_mask = np.zeros((128, 128, 6))

        mask1 = Image.open(mask_path + im_n + '_1.png')
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask2 = Image.open(mask_path + im_n + '_2.png')
        mask2.load()
        mask2 = np.asarray(mask2, dtype='int32')
        mask2 = (mask2/255).astype(np.float)

        mask3 = Image.open(mask_path + im_n + '_3.png')
        mask3.load()
        mask3 = np.asarray(mask3, dtype='int32')
        mask3 = (mask3/255).astype(np.float)

        mask4 = Image.open(mask_path + im_n + '_4.png')
        mask4.load()
        mask4 = np.asarray(mask4, dtype='int32')
        mask4 = (mask4/255).astype(np.float)

        mask5 = Image.open(mask_path + im_n + '_5.png')
        mask5.load()
        mask5 = np.asarray(mask5, dtype='int32')
        mask5 = (mask5/255).astype(np.float)

        total_mask[:,:,0] += mask1
        total_mask[:,:,1] += mask2
        total_mask[:,:,2] += mask3
        total_mask[:,:,3] += mask4
        total_mask[:,:,4] += mask5
        total_mask[:,:,5] += 1-(total_mask[:,:,0]+total_mask[:,:,1]+total_mask[:,:,2]+total_mask[:,:,3]+total_mask[:,:,4])
        total_mask[:,:,5][total_mask[:,:,5]>=1] = 1.
        total_mask[:,:,5][total_mask[:,:,5]<=0.999] = 0.
        total_mask[total_mask>=0.0001] = 1.
        train_mask[i] = total_mask

    # print(np.max(total_mask[:,:,1]))
    # im = Image.fromarray(total_mask[:,:,5]*255)
    # im = im.convert('RGB')
    # im.save("your_file.png")
    # assert False

    # total_mask[:,:,0] += hair_mask[:,:,0]
    # total_mask[:,:,1] = 1. - hair_mask[:,:,0]
    # train_mask = total_mask[:,:,0:2]

    # print(total_mask.shape)
    # print(train_mask.shape)
    # assert False

### run regression
# folder_path = './results/models/'+args.model_n#./results/logis/pretrained'
# output_path = './results/img/'+args.model_n+'/train/'
# regression(folder_path, output_path, 'weights.pt', models, given_w, given_seg=total_mask)

folder_path = './results/models/GANd/'+args.model_n#./results/logis/pretrained'
output_path = './results/img/GANd/'+args.model_n#+'/train/'
cnn(folder_path, output_path, 'weights.pt', models, given_w, given_seg=train_mask)
