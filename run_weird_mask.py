import torch, sys, os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import argparse
# from tqdm import tqdm

from nethook import InstrumentedModel
from dissect import dissect, collect_stack, VisualizeFeature#, GeneratorSegRunner
import stylegan
# from regression import regression
from cnn_2shot_new import cnn
from stylegan2_pytorch1 import model

load_stack = False

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_n', type=str, default='1', help='model_directory')
args = parser.parse_args()

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
torch.cuda.empty_cache()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

stylegan2_path = './stylegan2_pytorch1/checkpoint/stylegan2-ffhq-config-f.pt'
state_dict = torch.load(stylegan2_path)

# writer = SummaryWriter("./results/models/logs/"+args.model_di_n)
n_pic = 1
def load_mask(mask_path, img_list, glasses = False):
    n_class = 2
    train_mask = np.zeros((n_pic, 512, 512, n_class))

    mask1 = Image.open(mask_path + img_list[0] + '.png')
    mask1.load()
    mask1 = np.asarray(mask1, dtype='int32')
    mask1 = (mask1/255).astype(np.float)

    mask1[mask1>=0.0001] = 1.

    train_mask[0,:,:,0] += mask1
    train_mask[0,:,:,1] += 1-(mask1)
    train_mask[:,:,:,1][train_mask[:,:,:,1]>=1] = 1.
    train_mask[:,:,:,1][train_mask[:,:,:,1]<=0] = 0.
    return train_mask

def load_w(w_path, w_list):
    w = np.zeros((n_pic, 18, 512))
    for i, fn in enumerate(w_list):
        w[i] = np.load(w_path + fn)
    return torch.from_numpy(w).float().cuda()



with torch.no_grad():
    '''
    Set input_is_Wlatent | True for W-latent , False for Z-latent
    '''
    models = model.Generator(size=1024, style_dim=512, n_mlp=8, input_is_Wlatent=True).to(device)
    models.load_state_dict(state_dict['g_ema'], strict=False)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()

    models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
                            'convs.5','convs.6','convs.7','convs.8','convs.9',
                            'convs.10','convs.11','convs.12','convs.13','convs.14',
                            'convs.15'])

    '''
    Load Latent
    [1,1,512] for Z
    [1,18,512] for W+
    '''
    w_path = './dataset/WLatent200/'
    w_list = ['33.npy']#, '164.npy', '197.npy', '150.npy']
    # w_list = ['138.npy']#, '154.npy', '37.npy', '141.npy', '32.npy']

    all_w = load_w(w_path, w_list)

    given_w = TensorDataset(all_w)



    mask_path = './dataset/Mask33/weird_mask/'
    img_list = ['33_5']#, '00164_', '00197_', '00150_']
    # img_list = ['00138_']#, '00154_', '00037_', '00141_', '00032_']
    labels = load_mask(mask_path, img_list, False)

### run regression
# folder_path = './results/models/'+args.model_n#./results/logis/pretrained'
# output_path = './results/img/'+args.model_n+'/train/'
# regression(folder_path, output_path, 'weights.pt', models, given_w, given_seg=total_mask)

folder_path = './results/models/weird_mask/'+args.model_n#./results/logis/pretrained'
output_path = './results/img/weird_mask/'+args.model_n#+'/train/'
cnn(folder_path, output_path, 'weights.pt', models, given_w, given_seg=labels, load_pt=load_stack)
