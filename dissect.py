import torch, os, torchvision, json, tempfile
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict, defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# from segmenter import CustomSegmenter
from runningstats import RunningQuantile
from actviz import activation_visualization
from FF import Feedforward, MultiClassFF

from CNN_models import dilated_CNN_61 as CNN_net
# from unet_model import UNet as CNN_net

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

n_class = 2#10
image_size = 512#512

def dissect(outdir, model, raw_noise, given_seg=None, quantile_threshold=0.005, batch_size = 1, num_workers = 0):
    segmenter = GeneratorSegRunner()
    with torch.no_grad():
        device = next(model.parameters()).device
        noise_dataset = torch.utils.data.DataLoader(raw_noise, batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        print('Feeding data and collecting quantiles...')
        quantiles = collect_quantiles(outdir, model, noise_dataset, segmenter)

        ## threshold = quantile 0.995 of max 1
        levels = {k: qc.quantiles([1.0 - quantile_threshold])[:,0] for k, qc in quantiles.items()}
        for batch in noise_dataset:
            seg, seg_count,_,imshape = segmenter.run_and_segment_batch(batch, model, want_bincount=True, want_rgb=False)
            seg = seg.cpu()
            #segment_counts = seg_count[1].cpu()

        total_counts = 0
        features = model.retained_features() # {layer : (batch, c, h, w)}
        iou_layer = []
        '''
        for key in features.keys():
            value = features[key].cpu() #[batch ,c, h, w]
            upsample_grids = upsample_grid(value.shape[2:],
                    seg.shape[2:], imshape, scale_offset=None,
                    dtype=value.dtype, device=value.device)
            upsampled = F.grid_sample(value, upsample_grids,
                    padding_mode='border').cpu().numpy()
            #[batch ,c, h, w]

            for unit in range(value.shape[0]):
                feature_map = torch.from_numpy(upsampled[0,unit])
                threshold = levels[key][unit]
                iou_score = get_iou(seg, segment_counts, total_counts, feature_map, threshold)
                iou_layer.append(iou_score)
        '''
        dissect_image(outdir, features, model, levels, noise_dataset, segmenter)
        #generate_report(outdir, features, score_data, levels)

def collect_stack(seg_size, model, noise_dataset):
    with torch.no_grad():
        next(model.parameters()).device
        features = model.retained_features() #Orderdict of interested layer ('layer' : None)
        # print(features)
        # assert False
        stacks = []
        all_layers = list(features.keys()) # ['layer1','layer2']
        for i, batch_data in enumerate(noise_dataset):
            # print(i)
            model(batch_data[0].to(device))
            features = model.retained_features() #['layer1', 'layer2']
            for key in all_layers:
                # print(key)
                # print(features[key].shape)
                value = features[key].cpu()
                # assert False
                ''' Create grid '''
                upsample_grids = upsample_grid(value.shape[2:],
                    [seg_size,seg_size], [1024,1024], scale_offset=None,
                    dtype=value.dtype, device=value.device)
                upsampled = F.grid_sample(value, upsample_grids,
                                padding_mode='border').cpu().numpy()
                stacks.append(upsampled)

        stack = np.concatenate([i for i in stacks], axis=1)
        #print('Combined size = ', stack.shape)
        # [n, sum_chan, h, w]
    return stack

def generate_report(outdir, features, iou_score, levels):
    #Layerwise record
    for layer in features.keys():
        units, iou_rankings, index_rankings, combine_iou = [], [], [], []
        record = dict(layer=layer, units=units, index_rankings=index_rankings,
                    iou_rankings=iou_rankings, combine_iou=combine_iou)
        num_unit = features[layer].shape[1]
        lev = levels[layer]
        ious = iou_score[layer]
        rank, idx = torch.sort(ious, dim=1, descending=True)

        for unit in range(num_unit):
            units.append(dict(
                unit=unit,
                level=lev[unit].tolist(),
                iou=ious[unit].tolist()
            ))
        iou_rankings.append(rank.tolist())
        index_rankings.append(idx.tolist())

        os.makedirs(os.path.join(outdir, safe_dir_name(layer)), exist_ok=True)
        with open(os.path.join(outdir, safe_dir_name(layer), 'dissect.json'),
                'w') as jsonfile:
            json.dump(record, jsonfile, indent=1)

def dissect_image(outdir, torch_features, model, levels, noise_dataset, segmenter, prefix=''):

    levels = {k: v.cpu().numpy() for k, v in levels.items()}
    for i, batch_data in enumerate(noise_dataset):
        seg, _, rgb_im, _ = segmenter.run_and_segment_batch(batch_data, model, want_rgb=True)
        rgb_im = rgb_im.cpu().numpy() #[n h w c]
        seg_im = seg.cpu().numpy() #[n, 3, h, w]
        seg_im = np.transpose(seg_im, [0,2,3,1]).squeeze(0) #[h,w,3]
        seg_im = np.array(Image.fromarray((seg_im*255).astype(np.uint8)).resize(rgb_im.shape[1:3])) #[h,w,c]

        for layer in torch_features.keys():
            print('Generating images for ', layer)
            os.makedirs(os.path.join(outdir, safe_dir_name(layer), prefix + 'image'), exist_ok=True)
            ori_img = rgb_im[0]
            seg_mask = seg_im
            numpy_features = torch_features[layer].cpu().numpy() # : [batch(1), chan(unit), h, w]
            for suffix, im in [('ori.jpg', ori_img), ('seg.jpg', seg_mask)]:
                filename = os.path.join(outdir, safe_dir_name(layer), prefix + 'image', '%s' %(suffix))
                Image.fromarray(im).save(filename, optimize=True, quality=80)
            for unit in tqdm(range(torch_features[layer].shape[1])):
                activated_map = numpy_features[0, unit] # 1 feature mapping : (h, w)
                unit_levels = levels[layer][unit] # threshold of each unit : scalar
                thrs, mask = activation_visualization(
                        rgb_im,
                        activated_map,
                        unit_levels,
                        scale_offset=None,
                        return_mask=True)
                for suffix, im in [('alpha.jpg', thrs), ('mask.png', mask)]:
                    filename = os.path.join(outdir, safe_dir_name(layer), prefix + 'image', '%d-%s' %(unit, suffix))
                    Image.fromarray((im).astype(np.uint8)).save(filename, optimize=True, quality=80)

def collect_quantiles(outdir, model, noise_dataset, segmenter, resolution=256):
    '''
    Collects (estimated) quantile informationfor every channel in the
    retained layers of the model.
    Returns a map of quantiles (one RunningQuantile for each layer).
    '''
    device = next(model.parameters()).device
    features = model.retained_features() #Orderdict of interested layer ('layer' : None)

    layer_batch_size = 8
    all_layers = list(features.keys()) # ['layer1','layer2']

    layer_batches = [all_layers[i:i+layer_batch_size]
            for i in range(0, len(all_layers), layer_batch_size)] # start,stop,step :i=0 only
    quantiles = {}
    for layer_batch in layer_batches:
        for i, batch in enumerate(noise_dataset):
            # We don't care about the model output.
            model(batch[0].to(device))
            features = model.retained_features() #['layer1', 'layer2']

            for key in layer_batch:
                value = features[key]
                if quantiles.get(key, None) is None:
                    quantiles[key] = RunningQuantile(resolution=resolution)
                if len(value.shape) > 2:
                    # Put the channel index last.
                    value = value.permute(
                            (0,) + tuple(range(2, len(value.shape))) + (1,)
                            ).contiguous().view(-1, value.shape[1])
                quantiles[key].add(value)
        # Save GPU memory
        for key in layer_batch:
            quantiles[key].to_(torch.device('cpu'))
    for layer in quantiles:
        save_state_dict(quantiles[layer],
                os.path.join(outdir, safe_dir_name(layer), 'quantiles.npz'))
    return quantiles

def get_iou(seg, segment_counts, total_counts, feature_map, threshold):
    '''
    seg : [N, 3, H, W] default = 512
    seg_counts = int
    total_counts = int
    feature_map = [H, W] default = 512
    threshold = float
    '''

    upsampled = feature_map.cpu()
    threshold = np.array(threshold)
    #if len(threshold.shape) < 2:
    #    threshold = np.expand_dims(np.expand_dims(threshold, axis=0), axis=0)
    #print('Threshold = ', threshold)
    threshold = torch.from_numpy(threshold)
    # accepted mask(max_label>threshold) and pixel count
    #print(upsampled.shape)
    #print(threshold[None,: ,None,None].shape)
    amask = (upsampled > threshold).int()
    ac = amask.sum()
    #print(ac)
    # intersect mask (seg && amask) and pixel count
    imask = amask * seg[0,0]
    ic = imask.sum()
    iou_scores = score_tally_stats( total_counts, segment_counts, ac, ic)
    return iou_scores
'''
def dissect_iou(model, noise_dataset, levels, segmenter):
    # Require updated model

    (iou_scores,
        iqr_scores,
        total_counts,
        segment_counts,
        accepted_counts,
        intersection_counts) = {}, {}, None, None, {}, {}

    device = next(model.parameters()).device
    print('Calculating IOU...')
    assert noise_dataset.batch_size == 1
    segment_counts = 0
    total_counts = 0

    scale_offset_map = getattr(model, 'scale_offset', None)

    for i, batch in enumerate(noise_dataset):
        seg, seg_count,_, imshape = segmenter.run_and_segment_batch(
                            batch, model, want_bincount=True, want_rgb=False)
        segment_counts += seg_count[1].cpu().numpy()
        total_counts += (seg.shape[0] * seg.shape[2] * seg.shape[3])
        seg = seg.cpu()
        features = model.retained_features() #OrderDict[feature : value]

        for key, value in features.items():
            #upsampling grid
            upsample_grids = upsample_grid(value.shape[2:],
                    seg.shape[2:], imshape, scale_offset=None,
                    dtype=value.dtype, device=value.device)
            # upsampling value in key
            upsampled = torch.nn.functional.grid_sample(value,
                    upsample_grids, padding_mode='border').cpu()


            # accepted mask (max_label>threshold) and pixel count
            amask = (upsampled > levels[key][None,: ,None,None].cpu())
            ac = amask.int().view(amask.shape[1], -1).sum(1)
            print(amask.shape)
            # *****(batch, #labels(1), h, w)

            # intersect mask (seg && amask) and pixel count
            imask = amask * (seg.max(dim=1, keepdim=True)[0].cpu())
            ic = imask.int().view(imask.shape[1], -1).sum(1)
            print(ic)
            if key not in intersection_counts:
                intersection_counts[key] = torch.zeros(1, amask.shape[1],
                                        dtype=torch.long)
                accepted_counts[key] = torch.zeros(1, amask.shape[1],
                                        dtype=torch.long)
            accepted_counts[key] += ac
            intersection_counts[key] += ic

    iou_scores = {}
    for k in intersection_counts:
        iou_scores[k] = score_tally_stats(
            total_counts, segment_counts, accepted_counts[k], intersection_counts[k])
    return iou_scores
'''
def score_tally_stats(total, seg, ac, ic):

    epsilon = 1e-20 # avoid division-by-zero
    union = seg + ac - ic
    #print('intersection : ', ic.item())
    #print('union :', union.item())
    iou = ic.double() / (union.double() + epsilon)

    #arr = torch.empty(size=(2, 2) + ic.shape, dtype=ic.dtype, device=ic.device)
    '''
    sum arr[:,:] = 1
    arr[0,] = ac
    arr[1,] = seg
    '''
    '''
    arr[0, 0] = ic
    arr[0, 1] = ac - ic

    arr[1, 0] = seg - ic
    arr[1, 1] = total - union
    arr = arr / total
    arr = arr.double()
    ii = mutual_information(arr) # mutual_information
    hh = joint_entropy(arr) # joint_entropy
    iqr = ii / hh #I/H
    iqr[torch.isnan(iqr)] = 0 # 0/0 = 0
    '''
    return iou

def mutual_information(arr):
    '''
    mi(threshold_upsample, seg_mask) =
        sum sum p_joint(tu, sm) * log_2( p_joint(tu, sm)/ p(tu)p(sm) )
    '''
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p_joint = arr[i,j]
            p_margn = arr[i,:].sum(dim=0) * arr[:,j].sum(dim=0)
            term = p_joint * ((p_joint / p_margn).log())
            term[torch.isnan(term)] = 0
            total += term
    return total.clamp_(0)

def joint_entropy(arr):
    '''
    je(threshold_upsample, seg_mask) =
        - sum sum p_joint(tu, sm) * log_2( p_joint(tu, sm) )
    '''
    total = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p_joint = arr[i,j]
            term = p_joint * (p_joint.log())
            term[torch.isnan(term)] = 0
            total += term
    return (-total).clamp_(0)

def information_quality_ratio(arr):
    iqr = mutual_information(arr) / joint_entropy(arr)
    iqr[torch.isnan(iqr)] = 0
    return iqr

def upsample_grid(data_shape, target_shape, input_shape=None,
        scale_offset=None, dtype=torch.float, device=None):
    '''Prepares a grid to use with grid_sample to upsample a batch of
    features in data_shape to the target_shape. Can use scale_offset
    and input_shape to center the grid in a nondefault way: scale_offset
    maps feature pixels to input_shape pixels, and it is assumed that
    the target_shape is a uniform downsampling of input_shape.'''
    # Default is that nothing is resized.
    if target_shape is None:
        target_shape = data_shape
    # Make a default scale_offset to fill the image if there isn't one
    if scale_offset is None:
        scale = tuple(float(ts) / ds
                for ts, ds in zip(target_shape, data_shape))
        offset = tuple(0.5 * s - 0.5 for s in scale)
    else:
        scale, offset = (v for v in zip(*scale_offset))
        # Handle downsampling for different input vs target shape.
        if input_shape is not None:
            scale = tuple(s * (ts - 1) / (ns - 1)
                    for s, ns, ts in zip(scale, input_shape, target_shape))
            offset = tuple(o * (ts - 1) / (ns - 1)
                    for o, ns, ts in zip(offset, input_shape, target_shape))
    # Pytorch needs target coordinates in terms of source coordinates [-1..1]
    ty, tx = (((torch.arange(ts, dtype=dtype, device=device) - o)
                  * (2 / (s * (ss - 1))) - 1)
        for ts, ss, s, o, in zip(target_shape, data_shape, scale, offset))
    # Whoa, note that grid_sample reverses the order y, x -> x, y.
    grid = torch.stack(
        (tx[None,:].expand(target_shape), ty[:,None].expand(target_shape)),
        dim=2)[None,:,:,:].expand((1, target_shape[0], target_shape[1], 2))
    return grid

def save_state_dict(obj, filepath):
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname, exist_ok=True)
    dic = obj.state_dict()
    np.savez(filepath, **dic)

def safe_dir_name(filename):
    keepcharacters = (' ','.','_','-')
    return ''.join(c
            for c in filename if c.isalnum() or c in keepcharacters).rstrip()

# class GeneratorSegRunner:
#     def __init__(self):
#
#         self.segrunner = CustomSegmenter(segsizes=512)
#         self.num_classes = len(self.segrunner.get_label_and_category_names())
#
#     def get_label_and_category_names(self):
#         return self.segrunner.get_label_and_category_names()
#     def run_and_segment_batch(self, batch, model, want_bincount=False, want_rgb=False):
#         '''
#         Runs the dissected model on one batch of the dataset, and
#         returns a segmentation for the data.
#
#         Given a batch of size (n, c, y, x) the segmentation should
#         be a (long integer) tensor of size (n, d, y//r, x//r) where
#         d is the maximum number of simultaneous labels given to a pixel,
#         and where r is some (optional) resolution reduction factor.
#         In the segmentation returned, the label `0` is reserved for
#         the background "no-label".
#
#         In addition to the segmentation, rgb, and shape are returned
#         rgb is a viewable (n, y, x, rgb) byte image tensor for the data
#         for visualizations (reversing normalizations, for example), and
#         shape is the (y, x) size of the data.
#         '''
#         device = next(model.parameters()).device
#         z_inp = batch[0] #noise in
#         tensor_images = model(z_inp.to(device))
#         seg = self.segrunner.segment_batch(tensor_images, downsample=1) #Mask
#         if want_bincount:
#             index = torch.arange(z_inp.shape[0], dtype=torch.long, device=device)
#             bc = (seg + index[:, None, None, None] * self.num_classes).view(-1)
#             bc = bc.bincount(minlength=z_inp.shape[0] * self.num_classes)
#             #bc is [num of 0 , num of 1]
#             #bc = bc.view(z_inp.shape[0], self.num_classes)
#         else:
#             bc = None
#
#         if want_rgb:
#             images = ((tensor_images + 1) / 2 * 255)
#             rgb = images.permute(0, 2, 3, 1).clamp(0, 255).byte()
#             # rgb = (N, h, w, 3) #1024 1024
#         else:
#             rgb = None
#         # seg = (N, 3, h, w) h w = segsizes
#         return seg, bc, rgb, tensor_images.shape[2:]

def VisualizeFeature(outdir, combined_map, rgb_im, threshold=None, indice='', prefix=''):
    #print('Generating image...')
    '''
        feature_map : torch [h, w]
    '''
    rgb_im = rgb_im.cpu().numpy() #[n h w c]
    os.makedirs(os.path.join(outdir, safe_dir_name('Features'), prefix + 'image'), exist_ok=True)
    # Features
    combined_map = combined_map.numpy()
    # Thresholds
    if threshold is None:
        q_thr = RunningQuantile(resolution=combined_map.shape[0])
        value = torch.from_numpy(combined_map).view(-1).unsqueeze(1)
        q_thr.add(value)
        threshold = q_thr.quantiles([0.995]).cpu().numpy()
        #print('Threshold = ',threshold)
    else:
        threshold = np.array(threshold)
        if len(threshold.shape)<2:
            threshold = np.expand_dims(np.expand_dims(threshold, axis=0), axis=0)
    #print('Threshold = ', threshold)
    _, mask = activation_visualization(
            rgb_im,
            combined_map,
            threshold,
            return_mask=True)

    for suffix, im in [('ori.jpg', rgb_im[0]), ('mask.png', mask)]:
        filename = os.path.join(outdir, safe_dir_name('Features'), prefix + 'image', '%s-%s' %(indice,suffix))
        Image.fromarray((im).astype(np.uint8)).save(filename, optimize=True, quality=80)

def stacked_map(stack, regression_path, hidden_size = [2000]):
    '''
    Inference LOgistic regression on CPU
    '''
    with torch.no_grad():
        stack = torch.from_numpy(stack)
        #(n, c, h, w) 1 2564 512 512

        num_chan = stack.shape[1]
        size = stack.shape[2]

        #reg_model = Feedforward(num_chan, 200)
        reg_model = MultiClassFF(num_chan, hidden_size, 4)

        mod_weight = remove_layer_weight(regression_path, None)

        reg_model.load_state_dict(mod_weight)
        reg_model.eval()

        #for param in reg_model.state_dict():
        #    print(reg_model.state_dict()[param])

        flat_feature = stack[0].view(num_chan ,-1).T #[h*w, chan]
        combined = reg_model(flat_feature)

        logit = torch.exp(combined) ##Cluster MLP
        _,logit = logit.max(dim=1) ##Cluster MLP

    #combined = combined.view(size,size).cpu() ##Close for Cluster MLP
    #(512 512) Torch
    #return combined #return logit for Cluster
    return logit

def stacked_map_new(stack, regression_path):
    '''
    Inference LOgistic regression on CPU
    '''
    with torch.no_grad():
        stack = torch.from_numpy(stack)
        #(n, c, h, w) 1 2564 512 512

        num_chan = stack.shape[1]
        size = stack.shape[2]

        reg_model = CNN_net(n_class)

        mod_weight = remove_layer_weight(regression_path, None)

        reg_model.load_state_dict(mod_weight)
        reg_model.eval()

        #for param in reg_model.state_dict():
        #    print(reg_model.state_dict()[param])

        flat_feature = stack#.view(num_chan ,-1).T #[h*w, chan]
        # print(flat_feature.shape)
        # print(flat_feature)
        combined = reg_model(flat_feature)
        # print('!!!!')
        # print(torch.max(combined))

        logit = torch.exp(combined) ##Cluster MLP
        _,logit = logit.max(dim=1) ##Cluster MLP
        # print(torch.max(logit))
        # print(logit)

    #combined = combined.view(size,size).cpu() ##Close for Cluster MLP
    #(512 512) Torch
    #return combined #return logit for Cluster
    return logit

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def get_seg(stack, regression_path):
    with torch.no_grad():
        stack = torch.from_numpy(stack)
        #(n, c, h, w) 1 2564 512 512

        num_chan = stack.shape[1]
        size = stack.shape[2]

        reg_model = CNN_net(n_class)

        mod_weight = remove_layer_weight(regression_path, None)

        reg_model.load_state_dict(mod_weight)
        reg_model.eval()

        flat_feature = stack

        combined = reg_model(flat_feature)

        logit = torch.exp(combined) ##Cluster MLP
        _,logit = logit.max(dim=1) ## b, h, w, 1

        out = logit.view(-1)
        out_one_hot = one_hot_embedding(out, n_class)

        out_one_hot = torch.reshape(out_one_hot, (-1, image_size, image_size, n_class)).permute(0,3,1,2)
    return torch.reshape(out_one_hot, (-1, 1, image_size, image_size))


def remove_layer_weight(checkpoint, position):
    size_list = [512, 512, 512, 512, 512, 512, 512, 512, 256, 256, 128, 128, 64, 64, 32, 32]
    # 512 1024 1536 2048 2560 3072 3584 4096 4352 4608 4736 4864 4928 4992 5024 5056
    idx = 0
    pt = torch.load(checkpoint, map_location=lambda storage,loc: storage)
    #pt = torch.load(checkpoint)
    if position is None:
        return pt
    else:
        for i in range(position):
            idx+= size_list[i] #exclude ending
        pt['linear.weight'][:, idx:] = 0
        return pt
