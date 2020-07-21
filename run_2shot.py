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

load_stack = True

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
    if glasses:
        n_class = 11
    else: n_class = 10
    train_mask = np.zeros((n_pic, 512, 512, n_class))
    for i, prefix in enumerate(img_list):
        total_mask = np.zeros((512, 512, n_class))
        mask1 = Image.open(mask_path + prefix + 'l_eye.png')
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask2 = Image.open(mask_path + prefix + 'r_eye.png')
        mask2.load()
        mask2 = np.asarray(mask2, dtype='int32')
        mask2 = (mask2/255).astype(np.float)

        mask3 = Image.open(mask_path + prefix + 'l_lip.png')
        mask3.load()
        mask3 = np.asarray(mask3, dtype='int32')
        mask3 = (mask3/255).astype(np.float)

        mask4 = Image.open(mask_path + prefix + 'u_lip.png')
        mask4.load()
        mask4 = np.asarray(mask4, dtype='int32')
        mask4 = (mask4/255).astype(np.float)

        if os.path.exists(mask_path + prefix + 'mouth.png'):
            mask5 = Image.open(mask_path + prefix + 'mouth.png')
            mask5.load()
            mask5 = np.asarray(mask5, dtype='int32')
            mask5 = (mask5/255).astype(np.float)
        else:
            mask5 = mask4*0

        mask6 = Image.open(mask_path + prefix + 'nose.png')
        mask6.load()
        mask6 = np.asarray(mask6, dtype='int32')
        mask6 = (mask6/255).astype(np.float)

        eye_mask = mask1 + mask2
        eye_mask[eye_mask>=1] = 1. # Prevent intersection of left right part

        lip_mask = mask3 + mask4 + mask5
        lip_mask[lip_mask>=1] = 1.

        nose_mask = mask6
        nose_mask[nose_mask>=1] = 1.

        if os.path.exists(mask_path + prefix + 'cloth.png'):
            cloth_mask = Image.open(mask_path + prefix + 'cloth.png')
            cloth_mask.load()
            cloth_mask = np.asarray(cloth_mask, dtype='int32')
            cloth_mask = (cloth_mask/255).astype(np.float)
        else:
            cloth_mask = mask4*0

        if os.path.exists(mask_path + prefix + 'hair.png'):
            hair_mask = Image.open(mask_path + prefix + 'hair.png')
            hair_mask.load()
            hair_mask = np.asarray(hair_mask, dtype='int32')
            hair_mask = (hair_mask/255).astype(np.float)
        else:
            hair_mask = mask4*0

        if os.path.exists(mask_path + prefix + 'l_brow.png'):
            mask7 = Image.open(mask_path + prefix + 'l_brow.png')
            mask7.load()
            mask7 = np.asarray(mask7, dtype='int32')
            mask7 = (mask7/255).astype(np.float)
        else:
            mask7 = mask4*0

        if os.path.exists(mask_path + prefix + 'r_brow.png'):
            mask8 = Image.open(mask_path + prefix + 'r_brow.png')
            mask8.load()
            mask8 = np.asarray(mask8, dtype='int32')
            mask8 = (mask8/255).astype(np.float)
        else:
            mask8 = mask4*0

        brow_mask = mask7 + mask8
        brow_mask[brow_mask>=1] = 1.

        if os.path.exists(mask_path + prefix + 'l_ear.png'):
            mask9 = Image.open(mask_path + prefix + 'l_ear.png')
            mask9.load()
            mask9 = np.asarray(mask9, dtype='int32')
            mask9 = (mask9/255).astype(np.float)
        else:
            mask9 = mask8*0

        if os.path.exists(mask_path + prefix + 'r_ear.png'):
            mask10 = Image.open(mask_path + prefix + 'r_ear.png')
            mask10.load()
            mask10 = np.asarray(mask10, dtype='int32')
            mask10 = (mask10/255).astype(np.float)
        else:
            mask10 = mask9*0

        ear_mask = mask9 + mask10
        ear_mask[ear_mask>=1] = 1.

        neck_mask = Image.open(mask_path + prefix + 'neck.png')
        neck_mask.load()
        neck_mask = np.asarray(neck_mask, dtype='int32')
        neck_mask = (neck_mask/255).astype(np.float)

        skin_mask = Image.open(mask_path + prefix + 'skin.png')
        skin_mask.load()
        skin_mask = np.asarray(skin_mask, dtype='int32')
        skin_mask = (skin_mask/255).astype(np.float)

        if glasses:
            if os.path.exists(mask_path + prefix + 'eye_g.png'):
                glasses_mask = Image.open(mask_path + prefix + 'eye_g.png')
                glasses_mask.load()
                glasses_mask = np.asarray(glasses_mask, dtype='int32')
                glasses_mask = (glasses_mask/255).astype(np.float)
                total_mask[:,:,9] += glasses_mask[:,:,0]

            else: glasses_mask = eye_mask[:,:,0]*0

        total_mask[:,:,0] += eye_mask[:,:,0]
        total_mask[:,:,1] += lip_mask[:,:,0]
        total_mask[:,:,2] += nose_mask[:,:,0]
        total_mask[:,:,4] += cloth_mask[:,:,0]
        total_mask[:,:,5] += hair_mask[:,:,0]
        total_mask[:,:,6] += brow_mask[:,:,0]
        total_mask[:,:,7] += ear_mask[:,:,0]
        total_mask[:,:,8] += neck_mask[:,:,0]
        # total_mask[:,:,9] += skin_mask[:,:,0]
        total_mask[:,:,3] += (skin_mask[:,:,0]-eye_mask[:,:,0]-lip_mask[:,:,0]-nose_mask[:,:,0]-brow_mask[:,:,0])
        total_mask[:,:,n_class-1] = 1-(total_mask[:,:,0]+total_mask[:,:,1]+total_mask[:,:,2]+total_mask[:,:,4]+total_mask[:,:,5]+total_mask[:,:,6]+total_mask[:,:,7]+total_mask[:,:,8]+total_mask[:,:,3])
        if glasses:
            total_mask[:,:,n_class-1] = 1-(total_mask[:,:,0]+total_mask[:,:,1]+total_mask[:,:,2]+total_mask[:,:,4]+total_mask[:,:,5]+total_mask[:,:,6]+total_mask[:,:,7]+total_mask[:,:,8]+total_mask[:,:,3]+total_mask[:,:,9])

        total_mask[:,:,n_class-1][total_mask[:,:,n_class-1]>=1] = 1.
        total_mask[:,:,n_class-1][total_mask[:,:,n_class-1]<=0] = 0.


        train_mask[i] = total_mask
    return train_mask

def load_mask_S(mask_path, img_list):
    n_class = 4
    train_mask = np.zeros((n_pic, 512, 512, n_class))
    for i, prefix in enumerate(img_list):
        total_mask = np.zeros((512, 512, n_class))
        mask1 = Image.open(mask_path + prefix + 'l_eye.png')
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask2 = Image.open(mask_path + prefix + 'r_eye.png')
        mask2.load()
        mask2 = np.asarray(mask2, dtype='int32')
        mask2 = (mask2/255).astype(np.float)

        mask3 = Image.open(mask_path + prefix + 'l_lip.png')
        mask3.load()
        mask3 = np.asarray(mask3, dtype='int32')
        mask3 = (mask3/255).astype(np.float)

        mask4 = Image.open(mask_path + prefix + 'u_lip.png')
        mask4.load()
        mask4 = np.asarray(mask4, dtype='int32')
        mask4 = (mask4/255).astype(np.float)

        if os.path.exists(mask_path + prefix + 'mouth.png'):
            mask5 = Image.open(mask_path + prefix + 'mouth.png')
            mask5.load()
            mask5 = np.asarray(mask5, dtype='int32')
            mask5 = (mask5/255).astype(np.float)
        else:
            mask5 = mask4*0

        mask6 = Image.open(mask_path + prefix + 'nose.png')
        mask6.load()
        mask6 = np.asarray(mask6, dtype='int32')
        mask6 = (mask6/255).astype(np.float)

        eye_mask = mask1 + mask2
        eye_mask[eye_mask>=1] = 1. # Prevent intersection of left right part

        lip_mask = mask3 + mask4 + mask5
        lip_mask[lip_mask>=1] = 1.

        nose_mask = mask6
        nose_mask[nose_mask>=1] = 1.



        total_mask[:,:,0] += eye_mask[:,:,0]
        total_mask[:,:,1] += lip_mask[:,:,0]
        total_mask[:,:,2] += nose_mask[:,:,0]
        total_mask[:,:,3] += (1-eye_mask[:,:,0]-lip_mask[:,:,0]-nose_mask[:,:,0])
        total_mask[:,:,3][total_mask[:,:,3]>=1] = 1.
        total_mask[:,:,3][total_mask[:,:,3]<=0] = 0.


        train_mask[i] = total_mask
    return train_mask

def load_mask_eye(mask_path, img_list):
    n_class = 2
    train_mask = np.zeros((n_pic, 512, 512, n_class))
    for i, prefix in enumerate(img_list):
        total_mask = np.zeros((512, 512, n_class))
        mask1 = Image.open(mask_path + prefix + 'l_eye.png')
        mask1.load()
        mask1 = np.asarray(mask1, dtype='int32')
        mask1 = (mask1/255).astype(np.float)

        mask2 = Image.open(mask_path + prefix + 'r_eye.png')
        mask2.load()
        mask2 = np.asarray(mask2, dtype='int32')
        mask2 = (mask2/255).astype(np.float)

        eye_mask = mask1 + mask2
        eye_mask[eye_mask>=1] = 1. # Prevent intersection of left right part

        total_mask[:,:,0] += eye_mask[:,:,0]
        total_mask[:,:,1] += (1-eye_mask[:,:,0])
        total_mask[:,:,1][total_mask[:,:,1]>=1] = 1.
        total_mask[:,:,1][total_mask[:,:,1]<=0] = 0.


        train_mask[i] = total_mask
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
    w_list = ['33.npy']#, '180.npy']#, '164.npy', '197.npy', '150.npy']
    # w_list = ['138.npy']#, '154.npy', '37.npy', '141.npy', '32.npy']

    all_w = load_w(w_path, w_list)

    given_w = TensorDataset(all_w)



    mask_path = './dataset/Mask200/'
    img_list = ['00033_']#, '00180_']#, '00164_', '00197_', '00150_']
    # img_list = ['00138_']#, '00154_', '00037_', '00141_', '00032_']
    labels = load_mask(mask_path, img_list)

### run regression
# folder_path = './results/models/'+args.model_n#./results/logis/pretrained'
# output_path = './results/img/'+args.model_n+'/train/'
# regression(folder_path, output_path, 'weights.pt', models, given_w, given_seg=total_mask)

folder_path = './results/models/2-shot/'+args.model_n#./results/logis/pretrained'
output_path = './results/img/2-shot/'+args.model_n#+'/train/'
cnn(folder_path, output_path, 'weights.pt', models, given_w, given_seg=labels, load_pt=load_stack)
