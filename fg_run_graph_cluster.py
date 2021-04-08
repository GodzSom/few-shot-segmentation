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
from regression import regression
from fg_cnn_v2_pre import cnn
from stylegan2_pytorch1 import model
from GANDataLoader import GANDataLoader
from torchvision.utils import save_image
from kmeans import kmean_viz


parser = argparse.ArgumentParser(description='Add Some Values')
parser.add_argument('-model_n', type=str, default='1', help='model_directory')
parser.add_argument('-n_pic', type=int, default=1, help='number of training images')
parser.add_argument('-im', type=str, default='33', help='number of training images')
parser.add_argument('-num_ep', type=int, default=10000, help='number of training images')
args = parser.parse_args()

img_list = ['33', '180', '164', '197', '150', '138', '154', '37', '141', '32']
img_list = ['33']#, args.im]
fn = 'graph_cluster/'

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
torch.cuda.empty_cache()

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

stylegan2_path = './stylegan2_pytorch1/checkpoint/stylegan2-ffhq-config-f.pt'
state_dict = torch.load(stylegan2_path)

# writer = SummaryWriter("./results/models/logs/"+args.model_di_n)
n_pic = 2


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
    models#.cuda()

    # models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
    #                         'convs.5','convs.6','convs.7','convs.8','convs.9',
    #                         'convs.10','convs.11','convs.12','convs.13','convs.14',
    #                         'convs.15'])

    #B
    # models.retain_layers(['convs.4', 'convs.5','convs.6','convs.7'])
    models.retain_layers(['convs.5','convs.7'])

    # models.retain_layers(['convs.0','convs.1','convs.2','convs.3','convs.4',
    #                         'convs.5'])

    # models.retain_layers(['convs.10','convs.11','convs.12','convs.13','convs.14',
    #                     'convs.15'])

    # dict = models.state_dict()
    # for k,v in dict.items():
    #     print(k)
    # # print(models.retain_layers([pf+'1.conv1']))
    # assert False

    '''
    Load Latent
    [1,1,512] for Z
    [1,18,512] for W+
    '''
    w_path = './dataset/WLatent200/'
    # w_list =  [s+'.npy' for s in img_list[:n_pic]]
    #
    # all_w = load_w(w_path, w_list) #n 18 512
    # # print(all_w.shape)
    # # assert False
    #
    # given_w = TensorDataset(all_w)



    mask_path = './dataset/Mask200/'
    # img_list = [s.zfill(5)+'_' for s in img_list[:n_pic]]
    #
    # if args.segment_mode == '4c':
        # labels = load_mask_S(mask_path, img_list)
    n_class = 10
    # elif args.segment_mode == '10c':
    #     n_class = 10
        # labels = load_mask(mask_path, img_list)
    # else:
    #     assert False

    train_data = GANDataLoader(w_path, mask_path, img_list, '', n_pic, models)
    train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=1, shuffle=False)
high_contrast = [[0, 0, 0],
    [255, 255, 0], [28, 230, 255], [255, 52, 255], [122, 73, 0],
    [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89],
    [255, 219, 229], [0, 0, 166], [99, 255, 172],
    [183, 151, 98], [0, 77, 67], [143, 176, 255], [153, 125, 135],
    [90, 0, 7], [128, 150, 147], [254, 255, 230], [27, 68, 0],
    [79, 198, 1], [59, 93, 255], [74, 59, 83], [255, 47, 128],
    [97, 97, 90], [186, 9, 0], [107, 121, 0], [0, 194, 160],
    [255, 170, 146], [255, 144, 201], [185, 3, 170], [209, 97, 0],
    [221, 239, 255], [0, 0, 53], [123, 79, 75], [161, 194, 153],
    [48, 0, 24], [10, 166, 216], [1, 51, 73], [0, 132, 111],
    [55, 33, 1], [255, 181, 0], [194, 255, 237], [160, 121, 191],
    [204, 7, 68], [192, 185, 178], [194, 255, 153], [0, 30, 9],
    [0, 72, 156], [111, 0, 98], [12, 189, 102], [238, 195, 255]]
# [255, 255, 0], [28, 230, 255], [255, 52, 255], [0, 0, 0], [0, 137, 65], [0, 111, 166], [163, 0, 89]]

high_contrast_arr = np.array(high_contrast, dtype=np.uint8)


cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
num_chan = 1024
size = 128

# data = torch.zeros((1,num_chan, size, size))#.cuda()
# target = torch.zeros((size, size))
i = 0
for (data, target) in train_loader:
    print('-------')
    # assert False
    data = data
    # print(data[i])
    target = target#.cuda()
    i = i+1
    # print(data.shape)


# edges = torch.zeros((size, size, 4))
#
# for i in range(size):
#     for j in range(size):
#         if i+1 < size:
#             edges[i,j,0] = cossim(data[0,:,i,j], data[0,:,i+1,j])
#         # else: edges[i,j,0] = 0.88
#         if j+1 < size:
#             edges[i,j,1] = cossim(data[0,:,i,j], data[0,:,i,j+1])
#         # else: edges[i,j,1] = 0.88
#         if i > 0:
#             edges[i,j,2] = cossim(data[0,:,i,j], data[0,:,i-1,j])
#         # else: edges[i,j,2] = 0.88
#         if j > 0:
#             edges[i,j,3] = cossim(data[0,:,i,j], data[0,:,i,j-1])
#         # else: edges[i,j,3] = 0.88
#
# torch.save(edges, fn + 'edges_33')


edges = torch.load(fn + 'edges_33').numpy()
print(np.max(edges), np.min(edges))
# print(torch.min(edges[:-1,:,0]))#0.9429
# print(torch.min(edges[:,:-1,1]))#0.8877
# print(torch.min(edges[1:,:,2]))
# print(torch.min(edges[:,1:,3]))
#
# edges = edges - 0.88
# edges = edges/torch.max(edges)
# save_image(edges[:,:,0], fn+'A.png')
# save_image(edges[:,:,1], fn+'B.png')
# save_image(edges[:,:,2], fn+'C.png')
# save_image(edges[:,:,3], fn+'D.png')
# print(torch.min(edges[:-1,:,0]))

sorted_index = edges.argsort(axis=None, kind='mergesort')
sorted_index = np.vstack(np.unravel_index(sorted_index, edges.shape)).T

print(sorted_index[-1])# s s d
print(sorted_index.shape)

print(edges[sorted_index[-1,0],sorted_index[-1,1],sorted_index[-1,2]])
# assert False

n = -1
clus_n = 1
del_clus = 0
cluster = torch.zeros((size,size)).type(torch.IntTensor)

cluster_im = torch.zeros((3,size,size))
i=0
# for i in range(200):
while(torch.min(cluster)==0):
    print('--------------------------------------------')
    idex1 = sorted_index[n]
    # print(idex1)
    if idex1[2] == 0:
        idex2 = idex1.copy()
        idex2[0] = idex2[0] + 1
    elif idex1[2] == 1:
        idex2 = idex1.copy()
        idex2[1] = idex2[1] + 1
    elif idex1[2] == 2:
        idex2 = idex1.copy()
        idex2[0] = idex2[0] - 1
    elif idex1[2] == 3:
        idex2 = idex1.copy()
        idex2[1] = idex2[1] - 1

    if cluster[idex1[0], idex1[1]] == 0 and cluster[idex2[0], idex2[1]] == 0:
        cluster[idex1[0], idex1[1]] = clus_n
        cluster[idex2[0], idex2[1]] = clus_n
        clus_n = clus_n + 1
        print('new cluster ', idex1, idex2)

    elif cluster[idex1[0], idex1[1]] == 0:
        cluster[idex1[0], idex1[1]] = cluster[idex2[0], idex2[1]]
        print('add to cluster 1 ', idex1, idex2)

    elif cluster[idex2[0], idex2[1]] == 0:
        cluster[idex2[0], idex2[1]] = cluster[idex1[0], idex1[1]]
        print('add to cluster 2 ', idex1, idex2)

    elif cluster[idex1[0], idex1[1]] != cluster[idex2[0], idex2[1]]:
        cluster[cluster==cluster[idex2[0], idex2[1]]] = cluster[idex1[0], idex1[1]].clone()
        del_clus = del_clus + 1
        print('merge cluster ', idex1, idex2)
    else:
        print('none ', idex1, idex2)

    n = n - 1
    i = i + 1

    if i%1000 == 0:
        print(i, clus_n, del_clus, idex1, edges[idex1[0],idex1[1],idex1[2]])


        for p in range(size):
            for q in range(size):
                # print(cluster_im[:,p,q])
                # print(high_contrast_arr[cluster[p,q]])
                cluster_im[0,p,q] = high_contrast_arr[(cluster[p,q])%52][0]/255.0
                cluster_im[1,p,q] = high_contrast_arr[(cluster[p,q])%52][1]/255.0
                cluster_im[2,p,q] = high_contrast_arr[(cluster[p,q])%52][2]/255.0
        save_image(cluster_im, fn + 'cluster/'+str(i)+'_'+str(clus_n-del_clus)+'.png')
print(i, clus_n, del_clus, idex1, edges[idex1[0],idex1[1],idex1[2]])


for p in range(size):
    for q in range(size):
        # print(cluster_im[:,p,q])
        # print(high_contrast_arr[cluster[p,q]])
        cluster_im[0,p,q] = high_contrast_arr[(cluster[p,q])%52][0]/255.0
        cluster_im[1,p,q] = high_contrast_arr[(cluster[p,q])%52][1]/255.0
        cluster_im[2,p,q] = high_contrast_arr[(cluster[p,q])%52][2]/255.0
save_image(cluster_im, fn + 'cluster/'+str(i)+'_'+str(clus_n-del_clus)+'.png')
