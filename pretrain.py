'''
pytorch implementation Adrien Bitton
link: https://github.com/adrienchaton/PerceptualAudio_pytorch
paper codes Pranay Manocha
link: https://github.com/pranaymanocha/PerceptualAudio
'''


import torch
import numpy as np
import argparse
import os
import timeit

from models import JNDnet


parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id',type=int,default=0)
parser.add_argument('--mname',type=str,default='dataset_combined_linear')
args = parser.parse_args()

GPU_id = args.GPU_id
mname = args.mname
device = torch.device("cuda:{}".format(GPU_id) if torch.cuda.is_available() else "cpu")
print(device)

pretrained_path = './pretrained/'


if mname=='dataset_combined_linear':
    print('loading model for dataset_combined + dataset_linear ; trained with random time shift')
    ## average training loss =  0.41721172072989415 / average training accuracy =  0.8919461403362048 / average distance for train groudtruth 0,1 =  0.667132 2.796383
    ## average test loss =  0.5696099024886886 / average test accuracy =  0.736429038587312 / average distance for test groudtruth 0,1 =  0.9438109 2.3809838
    nconv = 14
    nchan = 16
    dist_dp = 0.
    dist_act = 'no'
    ndim0 = 8
    ndim1 = 4
    classif_dp = 0.2
    classif_BN = 2
    classif_act = 'sig'

if mname=='dataset_combined_linear_tshrink':
    print('loading model for dataset_combined + dataset_linear ; trained with random time shift and tanhshrink distance activation')
    ## average training loss =  0.3939998057374986 / average training accuracy =  0.917956009032366 / average distance for train groudtruth 0,1 =  0.07538187 0.81570065
    ## average test loss =  0.564762340591097 / average test accuracy =  0.7393721386527142 / average distance for test groudtruth 0,1 =  0.1715624 0.6238277
    nconv = 14
    nchan = 16
    dist_dp = 0.
    dist_act = 'tshrink'
    ndim0 = 8
    ndim1 = 4
    classif_dp = 0.2
    classif_BN = 2
    classif_act = 'sig'


state = torch.load(pretrained_path+mname+'.pth',map_location="cpu")['state']
model = JNDnet(nconv=nconv,nchan=nchan,dist_dp=dist_dp,dist_act=dist_act,ndim=[ndim0,ndim1],classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,dev=device)
model.load_state_dict(state)
model.to(device)
model.eval()


print('dummy forward JND net')
sr = 22050
dummy_ref = torch.zeros(1,sr).uniform_(-1,1).to(device)
dummy_per = torch.zeros(1,sr).uniform_(-1,1).to(device)
dummy_label = torch.zeros(1,1).long().to(device)
loss,dist,class_pred,class_prob = model.forward(dummy_ref,dummy_per,dummy_label)

print('dummy forward distance net')
dist = model.model_dist.forward(dummy_ref,dummy_per)






