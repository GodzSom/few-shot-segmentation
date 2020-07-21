import numpy as np
from skimage.transform import resize
import os, torch, json, glob
import skimage.morphology
from PRsegment import pr_segment
from collections import OrderedDict

class BaseSegmenter:
    def get_label_and_category_names(self):
        '''
        Returns two lists: first, a list of tuples [(label, category), ...]
        where the label and category are human-readable strings indicating
        the meaning of a segmentation class.  The 0th segmentation class
        should be reserved for a label ('-') that means "no prediction."
        The second list should just be a list of [category,...] listing
        all categories in a canonical order.
        '''
        raise NotImplemented()

    def segment_batch(self, tensor_images, downsample=1):
        '''
        Returns a multilabel segmentation for the given batch of (RGB [-1...1])
        images.  Each pixel of the result is a torch.long indicating a
        predicted class number.  Multiple classes can be predicted for
        the same pixel: output shape is (n, multipred, y, x), where
        multipred is 3, 5, or 6, for how many different predicted labels can
        be given for each pixel (depending on whether subdivision is being
        used).  If downsample is specified, then the output y and x dimensions
        are downsampled from the original image.
        '''
        raise NotImplemented()

    def predict_single_class(self, tensor_images, classnum, downsample=1):
        '''
        Given a batch of images (RGB, normalized to [-1...1]) and
        a specific segmentation class number, returns a tuple with
           (1) a differentiable ([0..1]) prediction score for the class
               at every pixel of the input image.
           (2) a binary mask showing where in the input image the
               specified class is the best-predicted label for the pixel.
        Does not work on subdivided labels.
        '''
        raise NotImplemented()

class CustomSegmenter(BaseSegmenter):
    def __init__(self, segsizes=None):
        if segsizes is None:
            segsizes = 512
        self.segsizes = segsizes
        self.labelcats = ['eye']

    def get_label_and_category_names(self):
        return self.labelcats

    def segment_batch(self, tensor_images, downsample=1):
        # input (batch, chan, h, w)
        pred = self.raw_seg_prediction(tensor_images)
        y,x = tensor_images.shape[2:]
        seg_shape = [self.segsizes, self.segsizes]
        segmask = torch.zeros(len(tensor_images), 1, seg_shape[0], seg_shape[1],
                        dtype=torch.long, device = tensor_images.device)
        segmask = pred['eye']
        segmask = torch.cat((segmask, segmask, segmask), dim=1)

        maskout = (segmask).int()
        return maskout

    def raw_seg_prediction(self, tensor_images):
        # input (batch, chan, h, w)
        y,x = tensor_images.shape[2:]
        b = len(tensor_images)
        tensor_images = (tensor_images + 1) / 2 * 255 #normalize

        size = (self.segsizes, self.segsizes)
        pred = { label : torch.zeros(len(tensor_images), 1,
                    self.segsizes, self.segsizes).cuda()
                    for label in self.labelcats }
        if size == tensor_images.shape[2:]:
            resized = tensor_images
        else:
            resized = torch.nn.AdaptiveAvgPool2d(size)(tensor_images)
        resized_pred = pr_segment(resized).cuda()
        for k in pred:
            pred[k] += resized_pred
        return pred
