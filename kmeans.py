import torch, os, torchvision, json, tempfile
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict, defaultdict
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
#from spherecluster import SphericalKMeans

from dissect import collect_stack, safe_dir_name
from runningstats import RunningQuantile
from actviz import activation_visualization


def kmean(dataset, k=4, Normal=True):
    print('Normalizing')
    if Normal:
        dataset = normalize(dataset)
    max_iteration = 300
    print('Create cluster')
    kmean = KMeans(n_clusters=k, n_init=5, max_iter=max_iteration,
                    verbose=True, n_jobs=3)
    print('Fitting')
    kmean.fit_predict(dataset) #[N_samples, feature]

    return kmean

def preprocess(segmenter, seg_size, model, noise_dataset):
    stack = collect_stack(segmenter, seg_size, model, noise_dataset)
    # [n, sum_chan, h, w]
    N = stack.shape[0]
    chan = stack.shape[1]

    feature = np.reshape(stack, (N, chan, -1)).squeeze(0)
    #[chan, H*W]
    return feature.T

# Within Sum Square (ELBOW)
def calculate_WSS(model, raw_noise, kmax):
    noise_dataset = torch.utils.data.DataLoader(raw_noise,
                batch_size=1, num_workers=0,
                pin_memory=False)
    sse = []
    points = preprocess(model, noise_dataset)
    for k in tqdm(range(1, kmax+1)):
        kmeans = KMeans(n_clusters = k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse

def visualize_cluster(outdir, model, raw_noise, k=4, one_hot=True, prefix=''):
    print('Visualizing Cluster...')
    device = next(model.parameters()).device
    segmenter = GeneratorSegRunner()
    noise_dataset = torch.utils.data.DataLoader(raw_noise,
                batch_size=1, num_workers=0,
                pin_memory=False)

    for i, batch_data in enumerate(noise_dataset):
        _, _, rgb_im, _ = segmenter.run_and_segment_batch(batch_data, model, want_rgb=True)
    rgb_im = rgb_im.cpu().numpy() #[n h w c]
    seg_size = 512

    all_features = preprocess(segmenter, seg_size, model, noise_dataset)
    print('Clustering...')
    cluster = kmean(all_features, k=k, Normal=False)
    print(cluster.labels_.shape)
    label = np.array(cluster.labels_)
    labels = np.expand_dims(cluster.labels_, axis=1)
    #centers = cluster.cluster_centers_

    k_im = kmean_viz(labels, seg_size)
    os.makedirs(os.path.join(outdir, safe_dir_name('cluster'), prefix + 'image'), exist_ok=True)

    for suffix, im in [('ori.jpg', rgb_im[0]), ('clus.png', k_im)]:
        filename = os.path.join(outdir, safe_dir_name('cluster'), prefix + 'image', '%s' %(suffix))
        Image.fromarray((im).astype(np.uint8)).resize(rgb_im.shape[1:3]).save(filename, optimize=True, quality=80)

    if one_hot == True:
        one_hot_emb = get_one_hot(label, k, seg_size)
        filename = os.path.join(outdir, safe_dir_name('cluster'), prefix + 'one_hot.npy')
        np.save(filename, one_hot_emb)

def kmean_viz(label, seg_size):
    # [H*W, 1]
    result = np.zeros((seg_size * seg_size, 3), dtype=np.uint8)
    for pixel in range(len(label)):
        result[pixel] = high_contrast_arr[label[pixel]]
    result = result.reshape((seg_size, seg_size, 3))
    return result

def get_one_hot(label, num_k, seg_size):
    one_hot = np.zeros((seg_size*seg_size, num_k))
    one_hot[np.arange(seg_size*seg_size), label] = 1
    return one_hot

high_contrast = [
    [255, 255, 0], [28, 230, 255], [255, 52, 255], [0, 0, 0],
    [255, 74, 70], [0, 137, 65], [0, 111, 166], [163, 0, 89],
    [255, 219, 229], [122, 73, 0], [0, 0, 166], [99, 255, 172],
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

high_contrast_arr = np.array(high_contrast, dtype=np.uint8)
