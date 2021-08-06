#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms

from timm.models import resnet50 as Network

# from torchvision.models import resnet50 as Network


# In[3]:


plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100


# In[6]:


device = torch.device('cuda:0')
model = Network(pretrained=False).to(device)
# model = torch.nn.DataParallel(model.to(device))

# path = '../logs/train_20210709120956/checkpoint.pth.tar'
path = '/root/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth'
# x = torch.load(path)['model_state_dict']
x = torch.load(path)
model.load_state_dict(x)


# In[ ]:

from collections import OrderedDict

new_state_dict = OrderedDict()
for key in x:
    new_key = 'backbone.bottom_up.model.' + key  # [7:]
    new_state_dict[new_key] = x[key]
torch.save(new_state_dict, '../logs/imagenet_init_resnet50_fpn.pth.tar')
print(new_state_dict.keys())

# In[7]:


transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


# In[8]:


from torchvision.datasets import ImageNet

# from efficient_net_v2.train import get_transforms

dataset = ImageNet(
    root='/workspace/Downloads/datasets/imagenet/',
    split='val',
    transform=transform,
)
print(len(dataset))


# In[ ]:


model.eval()
correct = 0

with tqdm.tqdm(total=len(dataset)) as pbar:
    for i, (im, target) in enumerate(dataset):

        im, target = dataset.__getitem__(100)
        # plt.imshow(im.cpu().numpy().transpose(1, 2, 0))
        # plt.show()

        im = im.unsqueeze(0).to(device)
        with torch.no_grad():
            r = model(im)
        r = torch.nn.Softmax(dim=1)(
            r,
        )
        r = torch.argmax(r).cpu().numpy()

        # print([r, target])

        if r == target:
            correct += 1

        pbar.set_description(
            desc='Acc %3.5f |  Cum Acc %3.3f'
            % ((correct / len(dataset)), (correct / (i + 1)))
        )
        pbar.update(1)

        # if i == 10:
        #     break
