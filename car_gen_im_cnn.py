import torch, json, sys, os, time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from tqdm import tqdm

from nethook import InstrumentedModel
from dissect import dissect, collect_stack, stacked_map_new, safe_dir_name, get_seg
from dissect import VisualizeFeature
import stylegan
from stylegan2_pytorch1 import model
# from evaluate import get_score
from kmeans import kmean_viz
import argparse
from CNN_models import iou_pytorch
num_pic = 20
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
stylegan2_path = './stylegan2_pytorch1/checkpoint/stylegan2-car-config-f.pt'
# regression_path = './results/models/CNN/'+args.model_n+'/weights.pt'
regression_path = './results/models/cars/'+args.model_n+'/weights.pt'
outdir = './results/img/cars/'+args.model_n
state_dict = torch.load(stylegan2_path)


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
    # print(w.shape)
    # assert False

    im_list.append(w.unsqueeze(0))

# w_path = './dataset/WLatent200/'
#
# all_w = load_w(w_path, num_pic)
#
# ffhq = TensorDataset(all_w)

im_latent = torch.cat([i for i in im_list], dim=0)


with torch.no_grad():
    models = model.Generator(size=512, style_dim=512, n_mlp=8, input_is_Wlatent=False).to(device)
    models.load_state_dict(state_dict['g_ema'], strict=False)
    models = InstrumentedModel(models)
    models.eval()
    models.cuda()

    models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
                            'convs.5','convs.6','convs.7','convs.8','convs.9',
                            'convs.10','convs.11','convs.12','convs.13'])
    # ffhq_noise_dataset = torch.utils.data.DataLoader(ffhq,
    #         batch_size=1, num_workers=0, pin_memory=False)


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
        stack = collect_stack(512, models, noise_dataset)
        combined = stacked_map_new(stack, regression_path).view(-1)
        k_im = kmean_viz(combined, 512)
        for batch in noise_dataset:
            rgb_im = models(batch[0].to(device))
        images = ((rgb_im + 1) / 2 * 255)
        rgb_im = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
        rgb_im = rgb_im.cpu().numpy()
        # print(rgb_im.shape)
        for suffix, im in [ ('clus.png', k_im), ('ori.png', rgb_im[0])]:
            filename = os.path.join(outdir, safe_dir_name('cluster'), '%d-%s' %(idx,suffix))
            Image.fromarray((im).astype(np.uint8)).resize([1024,1024]).save(filename, optimize=True, quality=80)




####################################################################
    # mask_path = './dataset/Mask200/'
    # img_list = ['00000_', '00001_', '00002_', '00003_', '00004_', '00005_',
    #             '00006_', '00007_', '00008_', '00009_', '00010_', '00011_',
    #             '00012_', '00013_', '00014_', '00015_', '00017_', '00018_',
    #             '00019_', '00020_']
    # labels = load_mask(mask_path, img_list).transpose(0,3,1,2)
    # labels = np.reshape(labels, (-1, 512, 512))
    #
    # # torch.cuda.empty_cache()
    # stack = collect_stack(512, models, ffhq_noise_dataset)[0]
    # stack = np.reshape(stack, (-1, 5056, 512, 512))
    #
    #
    # one_hot = get_seg(stack, regression_path)#b*10, 1, h, w
    #
    # iou = iou_pytorch(one_hot, labels)
