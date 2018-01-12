import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec

from dham import Dham

def show_attention(images, params, cmap="PuBu_r"):    
    assert len(images) == 2, "Parameter <images> must be a tuple of two torch tensors"
    org_imgs, att_imgs = [torch_img.data.cpu().numpy()[:, 0] for torch_img in images]
    params = [torch_param.data.cpu().numpy()[:, 0] for torch_param in params]

    rows = org_imgs.shape[0]//2
    plt.figure(figsize=(rows*2, 8))
    G = gridspec.GridSpec(rows, 4)
    # G.update(wspace=0.05, hspace=0.005)

    for row in range(rows):
        for i in range(2):
            axes = plt.subplot(G[row, i*2])
            plt.xticks(())
            plt.yticks(())
            axes.imshow(org_imgs[2*row+i], cmap=cmap)
            Y, X = org_imgs[2*row+i].shape
            x, y, scale = [param[2*row+i] for param in params]
            A, B, C, D = map(int, [(x+1)*X/2, (y+1)*Y/2, X*scale, Y*scale])
            rect = patches.Rectangle((-C//2 + A, -D//2 + B), C, D, linewidth=2, edgecolor='r', facecolor='none')
            axes.add_patch(rect)

            axes = plt.subplot(G[row, i*2+1])
            axes.imshow(att_imgs[2*row+i], cmap=cmap)
            plt.xticks(())
            plt.yticks(())

    # plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def train(epoch, optimizer, train_loader, model, apply_func):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = apply_func(data).cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output, image_new, att_params = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
    show_attention((data[:8], image_new[:8]), att_params[:8])
            


def resize(images, out_size, scale_min=0.24, scale_max=0.4):
    """
    Scales the images randomly with a constant which is defined between [0.3, 0.8].
    Scaled image is randomly postioned in empty output image.
    Parameters
        - images: Batch of images in a 4D torch tensor.
        - out_size: Resize dimensions of the output tensor.
         Tuple like (y, x). 
    """
    background = images[0, 0, 0, 0]
    B, C, Y, X = images.size()
    y, x = out_size

    # out_tensor = torch.zeros(B, C, y, x)+background
    out_tensor = torch.normal(torch.zeros(B, C, y, x), torch.ones(B, C, y, x)*0.5) + background

    for b in range(B):
        scale = np.random.uniform(scale_min, scale_max)
        y_, x_ = int(y*scale), int(x*scale)
        y_offset = int(np.random.uniform(0, 1.0)*(y-y_))
        x_offset = int(np.random.uniform(0, 1.0)*(x-x_))       
        indy = np.around(np.linspace(0, 1.0, y_).reshape(1, -1, 1)*(Y-1))
        indx = np.around(np.linspace(0, 1.0, x_).reshape(1, 1, -1)*(X-1))
        indy = np.tile(indy.astype(np.int32), (C, 1, x_))
        indx = np.tile(indx.astype(np.int32), (C, y_, 1))
        indc = np.tile(np.arange(C).astype(np.int32).reshape(-1, 1, 1), (1, y_, x_))

        mask = images[b][indc, indy, indx].gt(background).float()
        out_tensor[b, :,  y_offset:y_+y_offset, x_offset:x_+x_offset].mul_(1-mask)
        out_tensor[b, :,  y_offset:y_+y_offset, x_offset:x_+x_offset].add_(images[b][indc, indy, indx].mul_(mask))

    return out_tensor

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_att_1 = nn.Conv2d(1, 48, kernel_size=5)
        self.conv_att_2 = nn.Conv2d(48, 1, kernel_size=5)
        self.batchnorm2d_att = nn.BatchNorm2d(1)

        self.attention = Dham((28, 28))

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, image):
        feature_map = F.relu(self.conv_att_1(image))
        feature_map = self.conv_att_2(feature_map)
        feature_map = self.batchnorm2d_att(feature_map)

        x, attention_params = self.attention(image, feature_map)
        transformed_image = torch.squeeze(x, 2)

        x = F.relu(F.max_pool2d(self.conv1(transformed_image), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1), transformed_image, attention_params

if __name__ == "__main__":

    train_data = train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=128, shuffle=True)
    model = Net()
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(1, 1 + 1):
        train(epoch, optimizer, train_data, model, lambda image: resize(image, (120, 120)))



    

    