import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from dham import Dham

def draw_rect(images_tuple, x, y, scale, set_timer=True):    

    num_img = len(images_tuple)
    fig, axs = plt.subplots(1, num_img)

    def close_event():
        plt.close() #timer calls this function after 3 seconds and closes the window 

    timer = fig.canvas.new_timer(interval = 5000) #creating a timer object and setting an interval of 3000 milliseconds
    timer.add_callback(close_event)

    axs[0].imshow(images_tuple[0], cmap="PuBu_r")
    Y, X = images_tuple[0].shape
    A, B, C, D = map(int, [(x+1)*X/2, (y+1)*Y/2, X*scale, Y*scale])

    rect = patches.Rectangle((-C//2 + A, -D//2 + B), C, D, linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(rect)

    for ax, img in zip(axs[1:], images_tuple[1:]):
        ax.imshow(img, cmap="PuBu_r")
    
    if set_timer:
        timer.start()
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
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

            draw_rect(list(map(lambda x: x.data.cpu().numpy()[0, 0], (data, image_new))), *map(lambda a: a[0, 0], att_params), set_timer=True)

def resize(images, out_size, scale_min=0.3, scale_max=0.8):
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

    out_tensor = torch.zeros(B, C, y, x)+background

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

        out_tensor[b, :,  y_offset:y_+y_offset, x_offset:x_+x_offset] = images[b][indc, indy, indx]

    return out_tensor

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_att_1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv_att_2 = nn.Conv2d(16, 1, kernel_size=3)
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
    for epoch in range(1, 3+ 1):
        train(epoch, optimizer, train_data, model, lambda image: resize(image, (90, 90)))



    

    