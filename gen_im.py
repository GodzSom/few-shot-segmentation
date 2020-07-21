import torch, json, sys, os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from tqdm import tqdm

from nethook import InstrumentedModel
from dissect import dissect, collect_stack, stacked_map, safe_dir_name
from dissect import VisualizeFeature
import stylegan
from stylegan2_pytorch import model
# from evaluate import get_score
from kmeans import kmean_viz
import argparse

parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_n', type=str, default='1', help='model_directory')
args = parser.parse_args()

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

#os.environ['CUDA_VISIBLE_DEVICES']="0"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
torch.cuda.empty_cache()

print("Loading network")
stylegan2_path = './stylegan2_pytorch/checkpoint/stylegan2-ffhq-config-f.pt'
regression_path = './results/models/MLP/'+args.model_n+'/weights.pt'
outdir = './results/img/MLP/'+args.model_n
state_dict = torch.load(stylegan2_path)

num_pic = 20
im_list = []

for i in range(num_pic):
    # W latent
    #w_path = '/home/ntt/Celeb/Encode/npy/' + str(i+33) + '.npy'
    #w = np.load(w_path)
    #w = torch.from_numpy(w).float().unsqueeze(0)
    '''
    # Z_latent
    '''
    w = stylegan.z_sample(1, seed=i).unsqueeze(0)

    im_list.append(w.unsqueeze(0))

im_latent = torch.cat([i for i in im_list], dim=0)


with torch.no_grad():
    models = model.Generator(size=1024, style_dim=512, n_mlp=8, input_is_Wlatent=False).to(device)
    models.load_state_dict(state_dict['g_ema'], strict=False)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()

    models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
                            'convs.5','convs.6','convs.7','convs.8','convs.9',
                            'convs.10','convs.11','convs.12','convs.13','convs.14',
                            'convs.15',])


    all_iou = []
    image_detail, all_image = [], []
    record = dict(image=image_detail, all_image=all_image)

    for idx in tqdm(range(num_pic)):
        torch.cuda.empty_cache()
        random_z = TensorDataset(im_latent[idx])
        noise_dataset = torch.utils.data.DataLoader(random_z, batch_size=1)

        stack = collect_stack(512, models, noise_dataset)
        combined = stacked_map(stack, regression_path, [2000])
        # print(combined.shape)
        # assert False

        k_im = kmean_viz(combined, 512)
        prefix = ''
        os.makedirs(os.path.join(outdir, safe_dir_name('cluster'), prefix + 'image'), exist_ok=True)
        for batch in noise_dataset:
            rgb_im = models(batch[0].to(device))
        images = ((rgb_im + 1) / 2 * 255)
        rgb_im = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
        rgb_im = rgb_im.cpu().numpy()
        print(rgb_im.shape)
        for suffix, im in [ ('clus.png', k_im), ('ori.png', rgb_im[0])]:
            filename = os.path.join(outdir, '%d-%s' %(idx,suffix))
            Image.fromarray((im).astype(np.uint8)).resize([1024,1024]).save(filename, optimize=True, quality=80)
