from glob import glob
import scipy.io as sio
import torch
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
# import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.io import imread, imsave

from api import PRN

#from utils.cv_plot import plot_kpt
prn = PRN(is_dlib = True) #face detector

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype = np.int32) - 1
def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0])[36:48]:
        st = kpt[i, :2]
        image = cv2.circle(image,(st[0], st[1]), 1, (0,0,255), 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image

def pr_segment(tensor_images):
    [n, c, h, w] = tensor_images.shape
    tensor_images = tensor_images.cpu().detach().numpy().astype(np.uint8)
    seg=[]
    for i in range(n):
        image = tensor_images[i]
        # PR requires (h, w, c)
        image = np.swapaxes(image, 0, 1)
        image = np.swapaxes(image, 1, 2)

        pos = prn.process(image)

        if c>3:
            image = image[:,:,:3]
        max_size = max(h, w)
        if max_size>1000:
            image = rescale(image, max_size//2)
            image = (image*255).astype(np.uint8)

        if pos is None:
            print("No Pose Detected")
            masked = np.zeros([h,w,1])
            seg.append(masked)
        else:
            kpt = prn.get_landmarks(pos)
            left_eye = kpt[36:42, :2]
            right_eye = kpt[42:48, :2]
            left_eye = np.array(left_eye, dtype='int32')
            right_eye = np.array(right_eye, dtype='int32')

            left_eye = np.expand_dims(left_eye, axis=0)
            right_eye = np.expand_dims(right_eye, axis =0)
            mask = np.zeros(image.shape).astype(np.uint8)
            masked = cv2.fillPoly(mask, left_eye, 255)
            masked = cv2.fillPoly(masked, right_eye, 255)
            masked[masked>0] = 1
            masked = masked[:,:,0]
            masked = masked.reshape([masked.shape[0], masked.shape[1], 1])

            seg.append(masked)
    seg = np.array(seg)
    #swap to [n, c, h, w]
    seg = np.swapaxes(seg, 3, 2)
    seg = np.swapaxes(seg, 2, 1)
    segm = torch.from_numpy(seg)
    return segm

#image_folder ="test_seg/"
#save_folder ="test_seg/result"

#if not os.path.exists(save_folder):
#  os.mkdir(save_folder)

#types = ('*.jpg', '*.png')
#image_path_list= []

#for files in types:
#    image_path_list.extend(glob(os.path.join(image_folder, files)))
#total_num = len(image_path_list)

#main
'''
for i, image_path in enumerate(image_path_list):
    name = image_path.strip().split('/')[-1][:-4]
    image = imread(image_path)
    [h, w, c] = image.shape
    if c>3:
      image = image[:,:,:3]
    max_size = max(image.shape[0], image.shape[1])
    if max_size>1000:
      image = rescale(image, 1000./max_size)
      image = (image*255).astype(np.uint8)
    pos = prn.process(image)
    image = image/255
    if pos is None:
      continue
    kpt = prn.get_landmarks(pos)
    left_eye = kpt[36:42, :2]
    right_eye = kpt[42:48, :2]
    left_eye = np.array(left_eye, dtype='int32')
    right_eye = np.array(right_eye, dtype='int32')
    left_eye = np.expand_dims(left_eye, axis=0)
    right_eye = np.expand_dims(right_eye, axis =0)
    mask = np.zeros_like(image)
    masked = cv2.fillPoly(mask, left_eye, 255)
    masked = cv2.fillPoly(masked, right_eye, 255)
    #np.savetxt(os.path.join(save_folder, name + '_kpt.txt'), kpt)
    #cv2.imshow(mask)
    #cv2.imshow('sparse alignment', plot_kpt(image, kpt))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    plt.imshow(masked)
    #plt.imshow(image)
    plt.show()
'''
