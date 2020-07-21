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
from cnn import cnn
from stylegan2_pytorch1 import model

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

    # from torchsummary import summary
    # summary(models, (18,512))
    # print(models)
    # dict = models.state_dict()
    # for k,v in dict.items():
    #     print(k)

    # print(models.retain_layers(['convs.0']))
    # assert False
    # assert False


    '''
    Load Latent
    [1,1,512] for Z
    [1,18,512] for W+
    '''
    w_path = './33.npy'
    w = np.load(w_path)
    w = torch.from_numpy(w).float().unsqueeze(0).cuda()

    ## For random Z-latent : Seed=None to random ##
    #w = stylegan.z_sample(1, seed=900).unsqueeze(0)

    given_w = TensorDataset(w)

    # Given Seg Mask
    total_mask = np.zeros((512, 512, 10)) # 4 = eye, nose, mouth, bg

    mask_path = './dataset/Mask33/'

    mask1 = Image.open(mask_path + '00033_l_eye.png')
    mask1.load()
    mask1 = np.asarray(mask1, dtype='int32')
    mask1 = (mask1/255).astype(np.float)

    mask2 = Image.open(mask_path + '00033_r_eye.png')
    mask2.load()
    mask2 = np.asarray(mask2, dtype='int32')
    mask2 = (mask2/255).astype(np.float)

    mask3 = Image.open(mask_path + '00033_l_lip.png')
    mask3.load()
    mask3 = np.asarray(mask3, dtype='int32')
    mask3 = (mask3/255).astype(np.float)

    mask4 = Image.open(mask_path + '00033_u_lip.png')
    mask4.load()
    mask4 = np.asarray(mask4, dtype='int32')
    mask4 = (mask4/255).astype(np.float)

    mask5 = Image.open(mask_path + '00033_mouth.png')
    mask5.load()
    mask5 = np.asarray(mask5, dtype='int32')
    mask5 = (mask5/255).astype(np.float)

    mask6 = Image.open(mask_path + '00033_nose.png')
    mask6.load()
    mask6 = np.asarray(mask6, dtype='int32')
    mask6 = (mask6/255).astype(np.float)

    eye_mask = mask1 + mask2
    eye_mask[eye_mask>=1] = 1. # Prevent intersection of left right part

    lip_mask = mask3 + mask4 + mask5
    lip_mask[lip_mask>=1] = 1.

    nose_mask = mask6
    nose_mask[nose_mask>=1] = 1.

    cloth_mask = Image.open(mask_path + '00033_cloth.png')
    cloth_mask.load()
    cloth_mask = np.asarray(cloth_mask, dtype='int32')
    cloth_mask = (cloth_mask/255).astype(np.float)

    hair_mask = Image.open(mask_path + '00033_hair.png')
    hair_mask.load()
    hair_mask = np.asarray(hair_mask, dtype='int32')
    hair_mask = (hair_mask/255).astype(np.float)

    mask7 = Image.open(mask_path + '00033_l_brow.png')
    mask7.load()
    mask7 = np.asarray(mask7, dtype='int32')
    mask7 = (mask7/255).astype(np.float)

    mask8 = Image.open(mask_path + '00033_r_brow.png')
    mask8.load()
    mask8 = np.asarray(mask8, dtype='int32')
    mask8 = (mask8/255).astype(np.float)

    brow_mask = mask7 + mask8
    brow_mask[brow_mask>=1] = 1.

    mask9 = Image.open(mask_path + '00033_l_ear.png')
    mask9.load()
    mask9 = np.asarray(mask9, dtype='int32')
    mask9 = (mask9/255).astype(np.float)

    mask10 = Image.open(mask_path + '00033_r_ear.png')
    mask10.load()
    mask10 = np.asarray(mask10, dtype='int32')
    mask10 = (mask10/255).astype(np.float)

    ear_mask = mask9 + mask10
    ear_mask[ear_mask>=1] = 1.

    neck_mask = Image.open(mask_path + '00033_neck.png')
    neck_mask.load()
    neck_mask = np.asarray(neck_mask, dtype='int32')
    neck_mask = (neck_mask/255).astype(np.float)

    skin_mask = Image.open(mask_path + '00033_skin.png')
    skin_mask.load()
    skin_mask = np.asarray(skin_mask, dtype='int32')
    skin_mask = (skin_mask/255).astype(np.float)


    total_mask[:,:,0] += eye_mask[:,:,0]
    total_mask[:,:,1] += lip_mask[:,:,0]
    total_mask[:,:,2] += nose_mask[:,:,0]
    total_mask[:,:,4] += cloth_mask[:,:,0]
    total_mask[:,:,5] += hair_mask[:,:,0]
    total_mask[:,:,6] += brow_mask[:,:,0]
    total_mask[:,:,7] += ear_mask[:,:,0]
    total_mask[:,:,8] += neck_mask[:,:,0]
    # total_mask[:,:,9] += skin_mask[:,:,0]
    total_mask[:,:,9] += (skin_mask[:,:,0]-eye_mask[:,:,0]-lip_mask[:,:,0]-nose_mask[:,:,0]-brow_mask[:,:,0])
    total_mask[:,:,3] = 1-(total_mask[:,:,0]+total_mask[:,:,1]+total_mask[:,:,2]+total_mask[:,:,4]+total_mask[:,:,5]+total_mask[:,:,6]+total_mask[:,:,7]+total_mask[:,:,8]+total_mask[:,:,9])
    total_mask[:,:,3][total_mask[:,:,3]>=1] = 1.
    total_mask[:,:,3][total_mask[:,:,3]<=0] = 0.
    train_mask = total_mask

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

folder_path = './results/models/CNN/'+args.model_n#./results/logis/pretrained'
output_path = './results/img/CNN/'+args.model_n#+'/train/'
cnn(folder_path, output_path, 'weights.pt', models, given_w, given_seg=train_mask)
