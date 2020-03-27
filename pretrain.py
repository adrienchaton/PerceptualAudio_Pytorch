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
parser.add_argument('--mname',type=str,default='dataset_all')
args = parser.parse_args()

GPU_id = args.GPU_id
mname = args.mname
device = torch.device("cuda:{}".format(GPU_id) if torch.cuda.is_available() else "cpu")
print(device)

pretrained_path = './pretrained/'


if mname=='dataset_all':
    print('loading model for dataset_all ; trained with random time shift')
    ## average training loss =  0.5330756419173184 / average training accuracy =  0.7538885486834048
    ## average test loss =  0.5746318787898658 / average test accuracy =  0.710408307764928
    nconv = 14
    nchan = 32
    dist_dp = 0.
    dist_act = 'tshrink'
    ndim0 = 16
    ndim1 = 6
    classif_dp = 0.
    classif_BN = 2
    classif_act = 'no'

if mname=='dataset_linear':
    print('loading model for dataset_linear ; trained with random time shift and random gain')
    ## average training loss =  0.4631005721286643 / average training accuracy =  0.8449570815450643
    ## average test loss =  0.5571945954945462 / average test accuracy =  0.7481865284974093
    nconv = 14
    nchan = 8
    dist_dp = 0.
    dist_act = 'no'
    ndim0 = 4
    ndim1 = 3
    classif_dp = 0.2
    classif_BN = 2
    classif_act = 'sig'

if mname=='dataset_combined_linear':
    print('loading model for dataset_combined_linear ; trained with random time shift and random gain')
    ## average training loss =  0.4586653571157532 / average training accuracy =  0.8441916868779794
    ## average test loss =  0.5662889395219585 / average test accuracy =  0.737083060824068
    nconv = 18
    nchan = 8
    dist_dp = 0.
    dist_act = 'no'
    ndim0 = 4
    ndim1 = 3
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






