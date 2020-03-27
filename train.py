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
from utils import print_time,import_data,loss_plot,acc_plot,eval_scores


np.random.seed(12345)
try:
    torch.backends.cudnn.benchmark = True
except:
    print('cudnn.benchmark not available')


###############################################################################
### PARSE SETTINGS ; sr = 22050Hz is fixed and data is preprocessed accordingly

parser = argparse.ArgumentParser()
parser.add_argument('--GPU_id',type=int,default=0)
parser.add_argument('--mname',type=str,default='scratchJNDdefault')
parser.add_argument('--epochs',type=int,default=2000)
parser.add_argument('--bs',type=int,default=16)
parser.add_argument('--lr',type=float,default=0.0001)
parser.add_argument('--wdec',type=float,default=0.) # weight decay for optimizer
parser.add_argument('--nconv',type=int,default=14) # lossnet convolution depth
parser.add_argument('--nchan',type=int,default=32) # first channel dimension, to be doubled every 5 layers
parser.add_argument('--dist_dp',type=float,default=0.05) # droupout ratio in lossnet
parser.add_argument('--dist_act',type=str,default='no') # 'no' or 'sig' or 'tanh' or 'tshrink' or 'exp'
parser.add_argument('--ndim0',type=int,default=16) # first hidden size of the classifier
parser.add_argument('--ndim1',type=int,default=6) # second hidden size of the classifier
parser.add_argument('--classif_dp',type=float,default=0.05) # droupout ratio in classifnet
parser.add_argument('--classif_BN',type=int,default=2) # 1 if classifnet with batch-norm on 1st layer / 2 if on both hidden layers
parser.add_argument('--classif_act',type=str,default='no') # 'no' or 'sig' or 'tanh'
parser.add_argument('--Lsize',type=int,default=40000) # input signal size of lossnet
parser.add_argument('--shift',type=int,default=1) # 1 if randomly shifting signals to encourage shift invariance
parser.add_argument('--randgain',type=int,default=0) # 1 if randomly applying gain on training data to encourage amplitude invariance
parser.add_argument('--sub',type=int,default=-1) # -1 or an index to select a single perturbation subset to train on
parser.add_argument('--sub2',type=int,default=-1) # -1 or an additional index to select a single perturbation subset to train on
parser.add_argument('--minit',type=int,default=0) # 0 is using default pytorch setting, 1 is using random normal init
parser.add_argument('--opt',type=str,default='adam') # 'adam' or 'rmsp'
args = parser.parse_args()

GPU_id = args.GPU_id
mname = args.mname
device = torch.device("cuda:{}".format(GPU_id) if torch.cuda.is_available() else "cpu")
print(device)

epochs = args.epochs
batch_size = args.bs
lr = args.lr
wdec = args.wdec

lr_step = 50
lr_decay = 0.98

print('optimizer with batch_size,lr,wdec,lr_step,lr_decay = ',batch_size,lr,wdec,lr_step,lr_decay)

print('\nTRAINING '+mname+' for epochs,batch_size,lr')
print(epochs,batch_size,lr)


###############################################################################
### DATA SETTINGS AND IMPORT

data_path = './data/'
# it should contain numpy dictionaries named as 'subset'_data.npy ; e.g. dataset_combined_data.npy
# each dictionary entry is [first signal, second signal, human label]
output_path = './outputs/'
mpath = output_path+mname+'/'
if args.sub==-1:
    subsets = ['dataset_combined','dataset_eq','dataset_linear','dataset_reverb']
else: # select single subset 0,1,2,3
    subsets = [['dataset_combined','dataset_eq','dataset_linear','dataset_reverb'][args.sub]]
    if args.sub2!=-1:
        subsets.append(['dataset_combined','dataset_eq','dataset_linear','dataset_reverb'][args.sub2])

Lsize = args.Lsize
print('audio input size at training == ',Lsize)
# shorter segments are discarded ; longer segments are chunked in multiples of Lsize
shift = args.shift
n_shift = 4000
if shift==1:
    print('at training, xref or xper can be randomly shifted by '+str(n_shift)+' samples ~ ',n_shift/22050)

randgain = args.randgain
if randgain==1:
    gainmin = 0.1
    gainmax = 0.8 # scaling the input range ~ [-1.25,1.25] in [-1,1]
    print('at training, for every forward, apply random gain to [xref,xper] between ',gainmin,gainmax)
    print('test data is loaded with random gains, kept fixed throughout the training')
    rgains = [gainmin,gainmax]
else:
    rgains = False

train_loader,test_loader,train_refloader,test_refloader = import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8,rgains=rgains)


###############################################################################
### BUILD MODEL

nconv = args.nconv
nchan = args.nchan
dist_dp = args.dist_dp
dist_act = args.dist_act
ndim = [args.ndim0,args.ndim1]
classif_dp = args.classif_dp
classif_BN = args.classif_BN
classif_act = args.classif_act
minit = args.minit
print('\nBUILDING with settings nconv,nchan,dist_dp,dist_act,ndim,classif_dp,classif_BN,classif_act,minit')
print(nconv,nchan,dist_dp,dist_act,ndim,classif_dp,classif_BN,classif_act,minit)

model = JNDnet(nconv=nconv,nchan=nchan,dist_dp=dist_dp,dist_act=dist_act,ndim=ndim,classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,dev=device,minit=minit)
model.to(device)
model.train()

if args.opt=='rmsp':
    print('optimizer == RMSprop')
    optimizer = torch.optim.RMSprop(model.parameters(),lr=lr,weight_decay=wdec)
if args.opt=='adam':
    print('optimizer == ADAM')
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wdec)

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
        xper = minibatch[1].to(device,non_blocking=True)
        
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xref = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xref),dim=1)[:,:-n_shift]
            else:
                xref = torch.cat((xref,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        if shift==1 and np.random.rand()>0.75:
            if np.random.rand()>0.5:
                xper = torch.cat((torch.zeros(batch_size,n_shift).to(device,non_blocking=True),xper),dim=1)[:,:-n_shift]
            else:
                xper = torch.cat((xper,torch.zeros(batch_size,n_shift).to(device,non_blocking=True)),dim=1)[:,n_shift:]
        
        if randgain==1:
            g = torch.zeros(batch_size).to(device,non_blocking=True)
            torch.nn.init.uniform_(g,gainmin,gainmax)
            xref = xref*g.unsqueeze(1)
            xper = xper*g.unsqueeze(1)
            
        
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
    
    if (epoch+1)%3==0:
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




