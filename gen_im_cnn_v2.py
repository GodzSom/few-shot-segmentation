import torch, json, sys, os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from tqdm import tqdm

from nethook import InstrumentedModel
from dissect import dissect, collect_stack, stacked_map_new, safe_dir_name
from dissect import VisualizeFeature
import stylegan
# from evaluate import get_score
from kmeans import kmean_viz
import argparse
from stylegan2_pytorch_master2.bin.stylegan2_pytorch2 import train_from_folder


parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_n', type=str, default='1', help='model_directory')
args = parser.parse_args()

np_load_old = np.load
image_size = 128

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#os.environ['CUDA_VISIBLE_DEVICES']="0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

print("Loading network")
stylegan2_path = './stylegan2_pytorch_master2/bin/models'
name = 'GANd'
mod_num = 61
outdir = './results/img/GANd/'+args.model_n
# state_dict = torch.load(stylegan2_path)

regression_path = './results/models/'+name+'/'+args.model_n+'/weights.pt'
num_pic = 20
im_list = []

w_path = './noise_seed0.pt'
w = torch.load(w_path)
print(w[2][None].shape)

for i in range(num_pic):
    im_list.append(w[i+1][None].unsqueeze(0))
    # W latent
    #w_path = '/home/ntt/Celeb/Encode/npy/' + str(i+33) + '.npy'
    #w = np.load(w_path)
    #w = torch.from_numpy(w).float().unsqueeze(0)
    '''
    # Z_latent
    '''
    # w = stylegan.z_sample(1, seed=i).unsqueeze(0)
    # print(w.shape)
    # print(w.shape)
    # assert False

    # im_list.append(w.unsqueeze(0))

im_latent = torch.cat([i for i in im_list], dim=0) #20 1 1 512

with torch.no_grad():
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


    all_iou = []
    image_detail, all_image = [], []
    record = dict(image=image_detail, all_image=all_image)

    prefix = ''
    os.makedirs(os.path.join(outdir, safe_dir_name('cluster')), exist_ok=True)
    #########################################################################3
    # w_path = './33.npy'
    # w = np.load(w_path)
    # w = torch.from_numpy(w).float().unsqueeze(0)#.cuda()
    # torch.cuda.empty_cache()
    #
    # given_w = TensorDataset(w)
    # noise_dataset33 = torch.utils.data.DataLoader(given_w, batch_size=1)
    #
    # stack33 = collect_stack(512, models, noise_dataset33)
    # combined33 = stacked_map_new(stack33, regression_path).view(-1)
    # k_im33 = kmean_viz(combined33, 512)
    #
    #
    # for batch in noise_dataset33:
    #     rgb_im33 = models(batch[0].to(device))
    # images33 = ((rgb_im33 + 1) / 2 * 255)
    # rgb_im33 = images33.permute(0, 2, 3, 1).clamp(0, 255).byte()
    # rgb_im33 = rgb_im33.cpu().numpy()
    # for suffix, im in [ ('clus.png', k_im33), ('ori.png', rgb_im33[0])]:
    #     filename = os.path.join(outdir, safe_dir_name('cluster'), prefix + 'image', '%d-%s' %(33,suffix))
    #     Image.fromarray((im).astype(np.uint8)).resize([1024,1024]).save(filename, optimize=True, quality=80)
    # assert False
    ############################################################################33

    for idx in tqdm(range(num_pic)):
        torch.cuda.empty_cache()
        # print(given_w.shape)
        # print(random_z.shape)

        random_z = TensorDataset(im_latent[idx])
        noise_dataset = torch.utils.data.DataLoader(random_z, batch_size=1)
        stack = collect_stack(image_size, models, noise_dataset)
        combined = stacked_map_new(stack, regression_path).view(-1)
        k_im = kmean_viz(combined, 128)
        for batch in noise_dataset:
            rgb_im = models(batch[0].to(device))
        images = ((rgb_im + 1) / 2 * 255)
        rgb_im = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
        rgb_im = rgb_im.cpu().numpy()
        # print(rgb_im.shape)
        for suffix, im in [ ('clus.png', k_im), ('ori.png', rgb_im[0])]:
            filename = os.path.join(outdir, safe_dir_name('cluster'), '%d-%s' %(idx,suffix))
            Image.fromarray((im).astype(np.uint8)).resize([128,128]).save(filename, optimize=True, quality=80)
