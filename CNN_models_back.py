import torch
import torch.nn.functional as F
import torch.nn as nn
in_dim = 5056# 3008 #5056 #4992

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0, std=1e-2)

SMOOTH = 1e-6

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def iou_pytorch(outputs, labels, n_class):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).type(torch.IntTensor)  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.type(torch.IntTensor)

    weights = torch.sum(labels, dim=[1,2])

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    # thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    iou = (iou*100)

    # print(iou)
    rs_iou = torch.reshape(iou, (-1, n_class))
    rs_weights = torch.reshape(weights, (-1, n_class))
    print(rs_iou.type(torch.IntTensor))

    # print("!!!")
    # print(rs_iou*rs_weights)
    wm_iou = torch.sum(rs_iou*rs_weights, dim=1)/torch.sum(rs_weights, dim=1)

    return rs_iou.type(torch.IntTensor), torch.mean(wm_iou).type(torch.IntTensor)


class CNN(torch.nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0)
        self.conv2 = nn.Conv2d(128, 128, 3, stride = 1, padding = 0)
        self.conv3 = nn.Conv2d(128, 64, 3, stride = 1, padding = 0)
        self.conv4 = nn.Conv2d(64, num_class, 3, stride = 1, padding = 0)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = self.conv4(x)

        x = self.softmax(x)



        x = F.pad(x, (3,3,3,3))

        return x


# class CNN(torch.nn.Module): #a2
#     def __init__(self, num_class):
#         super(CNN, self).__init__()
#
#         self.conv1 = nn.Conv2d(5056, 128, 1, stride = 1, padding = 0)
#         self.conv2 = nn.Conv2d(128, 128, 7, stride = 1, padding = 0)
#         self.conv3 = nn.Conv2d(128, 64, 5, stride = 1, padding = 0)
#         self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0)
#         self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0)
#         self.conv6 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0)
#         self.conv7 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0)
#
#         self.softmax = torch.nn.LogSoftmax(dim=1)
#
#         self.apply(weights_init)
#
#     def forward(self, x):
#         # print(self.linears)
#         x = F.leaky_relu(self.conv1(x))
#         x = F.leaky_relu(self.conv2(x))
#         x = F.leaky_relu(self.conv3(x))
#         x = F.leaky_relu(self.conv4(x))
#         x = F.leaky_relu(self.conv5(x))
#         x = F.leaky_relu(self.conv6(x))
#         x = self.conv7(x)
#
#         x = self.softmax(x)
#
#
#
#         x = F.pad(x, (9,9,9,9))
#
#         return x

class dilated_CNN(torch.nn.Module):
    def __init__(self, num_class):
        super(dilated_CNN, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0, dilation=2)
        self.conv2 = nn.Conv2d(128, 128, 7, stride = 1, padding = 0, dilation=2)
        self.conv3 = nn.Conv2d(128, 64, 5, stride = 1, padding = 0, dilation=2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv6 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0, dilation=2)
        self.conv7 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0, dilation=2)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        # self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = self.conv7(x)

        x = self.softmax(x)
        # print(x.shape)

        x = F.pad(x, (18,18,18,18))

        return x

class dilated_CNN_101(torch.nn.Module):
    def __init__(self, num_class):
        super(dilated_CNN_101, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = nn.Conv2d(128, 64, 7, stride = 1, padding = 0, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 7, stride = 1, padding = 0, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 5, stride = 1, padding = 0, dilation=4)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=8)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv7 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv8 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0, dilation=4)
        self.conv9 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0, dilation=4)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = self.conv9(x)

        x = self.softmax(x)
        # print(x.shape)

        x = F.pad(x, (50,50,50,50))

        return x

class dilated_CNN_101_up_2(torch.nn.Module):
    def __init__(self, num_class):
        super(dilated_CNN_101_up, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = nn.Conv2d(128, 64, 7, stride = 1, padding = 0, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 7, stride = 1, padding = 0, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 5, stride = 1, padding = 0, dilation=4)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=8)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv7 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv8 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0, dilation=4)
        self.conv9 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0, dilation=4)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = nn.Conv2d(num_class, num_class, 1, stride = 1, padding = 0, dilation=1)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))

        x_up = self.up(x)
        x_up = F.leaky_relu(self.conv_up(x_up))
        x_up = self.up(x_up)
        out = self.conv_up(x_up)

        out = self.softmax(out)
        # print(x.shape)

        out = F.pad(out, (8,8,8,8))

        return out


class dilated_CNN_101_up(torch.nn.Module):
    def __init__(self, num_class):
        super(dilated_CNN_101_up, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = nn.Conv2d(128, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=8)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv7 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv8 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0, dilation=8)
        self.conv9 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0, dilation=0)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up = nn.Conv2d(num_class, num_class, 1, stride = 1, padding = 0, dilation=1)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        # self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = F.leaky_relu(self.conv9(x))

        x_up = self.up(x)
        x_up = F.leaky_relu(self.conv_up(x_up))
        x_up = self.up(x_up)
        out = self.conv_up(x_up)

        out = self.softmax(out)
        # print(x.shape)

        out = F.pad(out, (8,8,8,8))

        return out


class dilated_CNN_101_2(torch.nn.Module):
    def __init__(self, num_class):
        super(dilated_CNN_101, self).__init__()

        self.conv1 = nn.Conv2d(in_dim, 128, 1, stride = 1, padding = 0, dilation=1)
        self.conv2 = nn.Conv2d(128, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv4 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=8)
        self.conv5 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=1)
        self.conv6 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=2)
        self.conv7 = nn.Conv2d(64, 64, 3, stride = 1, padding = 0, dilation=4)
        self.conv8 = nn.Conv2d(64, 32, 3, stride = 1, padding = 0, dilation=8)
        self.conv9 = nn.Conv2d(32, num_class, 3, stride = 1, padding = 0, dilation=1)

        self.softmax = torch.nn.LogSoftmax(dim=1)

        # self.apply(weights_init)

    def forward(self, x):
        # print(self.linears)
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))
        x = F.leaky_relu(self.conv8(x))
        x = self.conv9(x)

        x = self.softmax(x)
        print(x.shape)

        x = F.pad(x, (50,50,50,50))

        return x
