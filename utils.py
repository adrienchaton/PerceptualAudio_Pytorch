'''
pytorch implementation Adrien Bitton
link: https://github.com/adrienchaton/PerceptualAudio_pytorch
paper codes Pranay Manocha
link: https://github.com/pranaymanocha/PerceptualAudio
'''


import torch
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display
import glob
import matplotlib
matplotlib.rcParams['agg.path.chunksize'] = 10000
matplotlib.use('Agg') # for the server
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report,accuracy_score


###############################################################################
### misc.

def print_time(s_duration):
    m,s = divmod(s_duration,60)
    h, m = divmod(m, 60)
    print('elapsed time = '+"%d:%02d:%02d" % (h, m, s))


###############################################################################
### data import

def import_data(data_path,subsets,Lsize,batch_size,train_ratio=0.8):
    train_y0 = []
    train_y1 = []
    train_labels = []
    test_y0 = []
    test_y1 = []
    test_labels = []
    
    for subset in subsets:
        print('loading '+subset)
        data_dic = np.load(data_path+subset+'_data.npy',allow_pickle=True).item()
        # one numpy dic per pre-processed subset of audio distortion
        # each dic entry is [first signal, second signal, human label]
        fcount = 0
        for fid in data_dic:
            y0 = data_dic[fid][0] # first signal
            y1 = data_dic[fid][1] # second signal
            lab = data_dic[fid][2] # human label
            min_len = np.min([y0.shape[0],y1.shape[0]])
            N = min_len//Lsize
            if N>0:
                if np.random.rand()>train_ratio:
                    test_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                    test_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                    test_labels.append(np.zeros((N,1),dtype='int')+lab)
                else:
                    train_y0.append(y0[:N*Lsize].reshape(N,Lsize))
                    train_y1.append(y1[:N*Lsize].reshape(N,Lsize))
                    train_labels.append(np.zeros((N,1),dtype='int')+lab)
                fcount+=1
        print('paired files amount to ',fcount)
    
    train_y0 = torch.from_numpy(np.concatenate(train_y0)).float()
    train_y1 = torch.from_numpy(np.concatenate(train_y1)).float()
    train_labels = torch.from_numpy(np.concatenate(train_labels)).long()
    train_ones = float(torch.sum(train_labels).item())
    
    test_y0 = torch.from_numpy(np.concatenate(test_y0)).float()
    test_y1 = torch.from_numpy(np.concatenate(test_y1)).float()
    test_labels = torch.from_numpy(np.concatenate(test_labels)).long()
    test_ones = float(torch.sum(test_labels).item())
    
    print('train/test Lsize pairs amount to ',train_y0.shape[0],test_y0.shape[0])
    print('train/test labels == one ("different") are ',int(train_ones),int(test_ones))
    print('train/test ratio of labels == one ("different") are ',train_ones/train_y0.shape[0],test_ones/test_y0.shape[0])
    
    train_dataset = torch.utils.data.TensorDataset(train_y0,train_y1,train_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    train_refloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    test_dataset = torch.utils.data.TensorDataset(test_y0,test_y1,test_labels)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,drop_last=True)
    test_refloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False,drop_last=False)
    
    return train_loader,test_loader,train_refloader,test_refloader


###############################################################################
### evaluation functions

def loss_plot(plot_name,loss_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('loss log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(loss_log[:,0])
    plt.subplot(2,1,2)
    plt.plot(loss_log[:,1])
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def acc_plot(plot_name,epoch_log,train_acc_log,test_acc_log):
    plt.figure(figsize=(12,8))
    plt.suptitle('accuracy log, rows=train/test')
    plt.subplot(2,1,1)
    plt.plot(epoch_log,train_acc_log)
    plt.subplot(2,1,2)
    plt.plot(epoch_log,test_acc_log)
    plt.savefig(plot_name+'.png',format='png')
    plt.close()

def eval_scores(model,train_refloader,test_refloader,device,report=True):
    train_pred = []
    train_labels = []
    train_dist = []
    test_pred = []
    test_labels = []
    test_dist = []
    
    with torch.no_grad():
        
        train_loss = 0
        for _,minibatch in enumerate(train_refloader):
            xref = minibatch[0].to(device)
            xper = minibatch[1].to(device)
            labels  = minibatch[2].to(device)
            loss,dist,class_pred,class_prob = model.forward(xref,xper,labels)
            labels = labels.squeeze()
            train_pred.append(class_pred.cpu().numpy())
            train_labels.append(labels.cpu().numpy())
            train_loss += loss.item()
            train_dist.append(dist.squeeze().cpu().numpy())
        train_loss /= len(train_pred)
        # loss is averaged in the minibatch "(reduction='mean')", then divided by the number of minibatches
        
        test_loss = 0
        for _,minibatch in enumerate(test_refloader):
            xref = minibatch[0].to(device)
            xper = minibatch[1].to(device)
            labels  = minibatch[2].to(device)
            loss,dist,class_pred,class_prob = model.forward(xref,xper,labels)
            labels = labels.squeeze()
            test_pred.append(class_pred.cpu().numpy())
            test_labels.append(labels.cpu().numpy())
            test_loss += loss.item()
            test_dist.append(dist.squeeze().cpu().numpy())
        test_loss /= len(test_pred)
    
    train_pred = np.concatenate(train_pred)
    train_labels = np.concatenate(train_labels)
    test_pred = np.concatenate(test_pred)
    test_labels = np.concatenate(test_labels)
    train_dist = np.concatenate(train_dist)
    test_dist = np.concatenate(test_dist)
    
    train_dist_0 = np.mean(train_dist[np.where(train_labels==0)])
    train_dist_1 = np.mean(train_dist[np.where(train_labels==1)])
    test_dist_0 = np.mean(test_dist[np.where(test_labels==0)])
    test_dist_1 = np.mean(test_dist[np.where(test_labels==1)])
    
    if report is True:
        print('TRAINING SET')
        print('average training loss = ',train_loss)
        print(classification_report(train_labels, train_pred, labels=[0,1], target_names=['same','different']))
    train_acc = accuracy_score(train_labels, train_pred)
    print('average training accuracy = ',train_acc)
    print('average distance for train groudtruth 0,1 = ',train_dist_0,train_dist_1)
    
    if report is True:
        print('TEST SET')
        print('average test loss = ',test_loss)
        print(classification_report(test_labels, test_pred, labels=[0,1], target_names=['same','different']))
    test_acc = accuracy_score(test_labels, test_pred)
    print('average test accuracy = ',test_acc)
    print('average distance for test groudtruth 0,1 = ',test_dist_0,test_dist_1)
    
    return train_acc,test_acc,train_loss,test_loss

