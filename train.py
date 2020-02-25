'''
pytorch implementation Adrien Bitton
link:
paper codes Pranay Manocha
link: https://github.com/pranaymanocha/PerceptualAudio
'''


import torch
import numpy as np
import argparse
import os
import timeit

from models import JNDnet
from utils import print_time,import_data,loss_plot,acc_plot,eval_scores


np.random.seed(666)
try:
    torch.backends.cudnn.benchmark = True
except:
    print('cudnn.benchmark not available')


###############################################################################
### PARSE SETTINGS ; sr = 22050Hz is fixed and data is preprocessed accordingly

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id',type=int,default=0)
parser.add_argument('--mname',type=str,default='scratchJNDdefault')
parser.add_argument('--epochs',type=int,default=1000)
parser.add_argument('--bs',type=int,default=64)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--nconv',type=int,default=14)
parser.add_argument('--nchan',type=int,default=32)
parser.add_argument('--dist_dp',type=float,default=0.1)
parser.add_argument('--nhiddens',type=int,default=3)
parser.add_argument('--ndim',type=int,default=64)
parser.add_argument('--classif_dp',type=float,default=0.1)
parser.add_argument('--Lsize',type=int,default=22050)
parser.add_argument('--shift',type=int,default=1)
parser.add_argument('--sub',type=int,default=-1)
args = parser.parse_args()

# default usage is
# python train.py --GPU_id 0 --mname 'name' --epochs 1000 --bs 64 --lr 0.0001 --nconv 14 --nchan 32 --dist_dp 0.1 --nhiddens 3 --ndim 64 --classif_dp 0.1

GPU_id = args.GPU_id
mname = args.mname
device = torch.device("cuda:{}".format(GPU_id) if torch.cuda.is_available() else "cpu")
print(device)

epochs = args.epochs
batch_size = args.bs
lr = args.lr

lr_step = 25
lr_decay = 0.97

print('\nTRAINING '+mname+' for epochs,batch_size,lr')
print(epochs,batch_size,lr)


###############################################################################
### DATA SETTINGS AND IMPORT

data_path = '/data/unagi0/adrien/audio_perception/'
output_path = '/data/unagi0/adrien/audio_perception/outputs/'
mpath = output_path+mname+'/'
if args.sub==-1:
    subsets = ['dataset_combined','dataset_eq','dataset_linear','dataset_reverb']
else: # select single subset 0,1,2,3
    subsets = [['dataset_combined','dataset_eq','dataset_linear','dataset_reverb'][args.sub]]

Lsize = args.Lsize
print('audio input size at training == ',Lsize)
# shorter segments are discarded ; longer segments are chunked in multiples of Lsize
shift = args.shift
n_shift = 1000
if shift==1:
    print('at training, xref or xper can be randomly shifted by '+str(n_shift)+' samples ~ ',n_shift/22050)

train_loader,test_loader,train_refloader,test_refloader = import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8)


###############################################################################
### BUILD MODEL

nconv = args.nconv
nchan = args.nchan
dist_dp = args.dist_dp
nhiddens = args.nhiddens
ndim = args.ndim
classif_dp = args.classif_dp
print('\nBUILDING with settings nconv,nchan,dist_dp,nhiddens,ndim,classif_dp')
print(nconv,nchan,dist_dp,nhiddens,ndim,classif_dp)

model = JNDnet(nconv=nconv,nchan=nchan,dist_dp=dist_dp,nhiddens=nhiddens,ndim=ndim,classif_dp=classif_dp,dev=device)
model.to(device)
model.train()

optimizer = torch.optim.Adam(model.parameters(),lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,lr_step,gamma=lr_decay)


###############################################################################
### PRE-CHECKS

for _, minibatch in enumerate(train_loader):
    break
model.grad_check(minibatch,optimizer)

model.eval()
train_acc,test_acc,train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device)
epoch_log = [0]
train_acc_log = [train_acc]
test_acc_log = [test_acc]


###############################################################################
### TRAINING

model.train()

os.makedirs(mpath)

loss_log = np.zeros((epochs,2)) # train/test losses
itr = 0

start_time = timeit.default_timer()


for epoch in range(epochs):
    
    #### training step
    model.train()
    n_mb = 0
    ep_loss = torch.tensor([0.]).to(device,non_blocking=True)
    
    for _, minibatch in enumerate(train_loader):
        
        xref = minibatch[0].to(device,non_blocking=True)
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xref = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xref),dim=1)[:,:-n_shift]
            else:
                xref = torch.cat((xref,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        xper = minibatch[1].to(device,non_blocking=True)
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xper = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xper),dim=1)[:,:-n_shift]
            else:
                xper = torch.cat((xper,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        labels  = minibatch[2].to(device,non_blocking=True)
        loss,dist,class_pred,class_prob = model.forward(xref,xper,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ep_loss+=loss
        n_mb+=1
        itr+=1
    
    loss_log[epoch,0] = ep_loss.item()/n_mb
    
    #### testing step
    model.eval()
    n_mb = 0
    ep_loss = torch.tensor([0.]).to(device,non_blocking=True)
    
    with torch.no_grad():
        for _,minibatch in enumerate(test_loader):
            xref = minibatch[0].to(device,non_blocking=True)
            xper = minibatch[1].to(device,non_blocking=True)
            labels  = minibatch[2].to(device,non_blocking=True)
            loss,dist,class_pred,class_prob = model.forward(xref,xper,labels)
            ep_loss+=loss
            n_mb+=1
    
    loss_log[epoch,1] = ep_loss.item()/n_mb
    
    if (epoch+1)%7==0:
        print('\n***  '+mname+' -  EPOCH #'+str(epoch+1)+' out of '+str(epochs)+' current itr=',itr)
        print('averaged training loss',loss_log[epoch,0])
        print('averaged test loss',loss_log[epoch,1])
        train_acc,test_acc,train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device,report=False)
        train_acc_log.append(train_acc)
        test_acc_log.append(test_acc)
        epoch_log.append(epoch+1)
        
        plot_name = mpath+'loss_plot'
        loss_plot(plot_name,loss_log)
        plot_name = mpath+'acc_plot'
        acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log)
        
        print_time(timeit.default_timer()-start_time)
    
    scheduler.step()


###############################################################################
#### POST-TRAINING save and export

print('\nTRAINING FINISHED for model '+mname+'\n')

for g in optimizer.param_groups:
    lr_end = g['lr']
print('\nlr_end == ',lr_end)
print_time(timeit.default_timer()-start_time)
print('#iter = ',itr)

plot_name = mpath+'loss_plot'
loss_plot(plot_name,loss_log)

model.eval()
print('\n\nREPORT for model '+mname)
train_acc,test_acc,train_loss,test_loss = eval_scores(model,train_refloader,test_refloader,device)
train_acc_log.append(train_acc)
test_acc_log.append(test_acc)
epoch_log.append(epochs)
plot_name = mpath+'acc_plot'
acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log)

states = {'epochs':epochs,'state':model.state_dict(),'optim':optimizer.state_dict(),'itr':itr,\
          'train_acc':train_acc,'test_acc':test_acc,'train_loss':train_loss,'test_loss':test_loss}
torch.save(states,mpath+mname+'.pth')
np.save(mpath+'losses.npy',loss_log)




