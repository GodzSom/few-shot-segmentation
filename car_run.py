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

stylegan2_path = './stylegan2_pytorch1/checkpoint/stylegan2-car-config-f.pt'
state_dict = torch.load(stylegan2_path)

# writer = SummaryWriter("./results/models/logs/"+args.model_di_n)
n_pic = 3
def load_mask(mask_path, img_list):
    n_class = 4
    train_mask = np.zeros((n_pic, 512, 512, n_class))
    for i, prefix in enumerate(img_list):
        total_mask = np.zeros((512, 512, n_class))
        mask1 = Image.open(mask_path + prefix + '_wheels.png').resize((512, 512))
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask2 = Image.open(mask_path + prefix + '_glass.png').resize((512, 512))
        mask2.load()
        mask2 = np.asarray(mask2, dtype='int32')
        mask2 = (mask2/255).astype(np.float)

        mask3 = Image.open(mask_path + prefix + '_car.png').resize((512, 512))
        mask3.load()
        mask3 = np.asarray(mask3, dtype='int32')
        mask3 = (mask3/255).astype(np.float)

        mask1[mask1>=0.0001] = 1.
        mask2[mask2>=0.0001] = 1.
        mask3[mask3>=0.0001] = 1.

        mask3 = mask3 - mask1 - mask2
        mask3[mask3>=1] = 1.
        mask3[mask3<=0] = 0.

        total_mask[:,:,0] += mask1
        total_mask[:,:,1] += mask2
        total_mask[:,:,2] += mask3
        total_mask[:,:,3] += 1-(mask1+mask2+mask3)
        total_mask[:,:,3][total_mask[:,:,3]>=1] = 1.
        total_mask[:,:,3][total_mask[:,:,3]<=0] = 0.

        # print(np.max(total_mask[:,:,1]))
        # im = Image.fromarray(total_mask[:,:,3]*255)
        # im = im.convert('RGB')
        # im.save("your_file.png")
        # assert False


        train_mask[i] = total_mask
    return train_mask

def load_mask_wheel(mask_path, img_list):
    n_class = 2
    train_mask = np.zeros((n_pic, 512, 512, n_class))
    for i, prefix in enumerate(img_list):
        total_mask = np.zeros((512, 512, n_class))
        mask1 = Image.open(mask_path + prefix + '_wheels.png').resize((512, 512))
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask1[mask1>=0.0001] = 1.

        total_mask[:,:,0] += mask1
        total_mask[:,:,1] += 1-(mask1)
        total_mask[:,:,1][total_mask[:,:,1]>=1] = 1.
        total_mask[:,:,1][total_mask[:,:,1]<=0] = 0.

        # print(np.max(total_mask[:,:,1]))
        # im = Image.fromarray(total_mask[:,:,3]*255)
        # im = im.convert('RGB')
        # im.save("your_file.png")
        # assert False


        train_mask[i] = total_mask
    return train_mask

def load_w(w_list):
    w = np.zeros((n_pic, 18, 512))
    for i, fn in enumerate(w_list):
        w[i] = stylegan.z_sample(1, seed=w_list[i]).unsqueeze(0)
    return torch.from_numpy(w).float().cuda()



with torch.no_grad():
    '''
    Set input_is_Wlatent | True for W-latent , False for Z-latent
    '''
    models = model.Generator(size=512, style_dim=512, n_mlp=8, input_is_Wlatent=True).to(device)
    models.load_state_dict(state_dict['g_ema'], strict=False)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()

    models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
                            'convs.5','convs.6','convs.7','convs.8','convs.9',
                            'convs.10','convs.11','convs.12','convs.13'])

    '''
    Load Latent
    [1,1,512] for Z
    [1,18,512] for W+
    '''
    # w_path = './masks/WLatent200/'
    # w_list = ['33.npy', '180.npy', '164.npy', '197.npy', '150.npy']
    w_list = [0, 24, 40]

    all_w = load_w(w_list)

    given_w = TensorDataset(all_w)



    mask_path = './masks/cars/mask/'
    # img_list = ['00033_', '00180_', '00164_', '00197_', '00150_']
    img_list = ['0', '24', '40']
    labels = load_mask_wheel(mask_path, img_list)

### run regression
# folder_path = './results/models/'+args.model_n#./results/logis/pretrained'
# output_path = './results/img/'+args.model_n+'/train/'
# regression(folder_path, output_path, 'weights.pt', models, given_w, given_seg=total_mask)

folder_path = './results/models/cars/'+args.model_n#./results/logis/pretrained'
output_path = './results/img/cars/'+args.model_n#+'/train/'
cnn(folder_path, output_path, 'weights.pt', models, given_w, given_seg=labels, load_pt=load_stack)
