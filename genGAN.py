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
from stylegan2_pytorch_master2.bin.stylegan2_pytorch2 import train_from_folder
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
stylegan2_path = './stylegan2_pytorch_master2/bin/models'
name = 'GANd'
mod_num = 61
outdir = './stylegan2_pytorch_master2/bin/results'
# state_dict = torch.load(stylegan2_path)

num_pic = 50
im_list = []

train_from_folder(results_dir=outdir, models_dir=stylegan2_path, name=name, generate=True)
