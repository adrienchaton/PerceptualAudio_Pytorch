'''
pytorch implementation Adrien Bitton
link: https://github.com/adrienchaton/PerceptualAudio_pytorch
paper codes Pranay Manocha
link: https://github.com/pranaymanocha/PerceptualAudio
'''


import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
### sub networks

class lossnet(nn.Module):
    def __init__(self,nconv=14,nchan=32,dp=0.1,dist_sig=1):
        
        # base settings for 16kHz, applied to 22kHz
        # in the case of dropout, at training, forward over two same tensors does not give dist=0
        # the droupout is randomized differently for each pass on xref/xper
        
        super(lossnet, self).__init__()
        self.nconv = nconv
        self.dist_sig = dist_sig
        self.convs = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        for iconv in range(nconv):
            if iconv==0:
                chin = 1
            else:
                chin = nchan
            if (iconv+1)%5==0:
                nchan = nchan*2
            if iconv<nconv-1:
                conv = [nn.Conv1d(chin,nchan,3,stride=2,padding=1),nn.BatchNorm1d(nchan),nn.LeakyReLU()]
                if dp!=0:
                    conv.append(nn.Dropout(p=dp))
            else:
                # last conv has no stride and no dropout
                conv = [nn.Conv1d(chin,nchan,3,stride=1,padding=1),nn.BatchNorm1d(nchan),nn.LeakyReLU()]
            self.convs.append(nn.Sequential(*conv))
            self.chan_w.append(nn.Parameter(torch.randn(nchan),requires_grad=True))
        
        if dist_sig==1:
            self.act = nn.Sigmoid()
        elif dist_sig==2:
            self.act = nn.Tanh()
        else:
            self.act = nn.Identity()
    
    def forward(self,xref,xper):
        # xref and xper are [batch,L]
        xref = xref.unsqueeze(1)
        xper = xper.unsqueeze(1)
        dist = 0
        for iconv in range(self.nconv):
            xref = self.convs[iconv](xref)
            xper = self.convs[iconv](xper)
            diff = (xref-xper).permute(0,2,1) # channel last
            wdiff = diff*self.chan_w[iconv]
            wdiff = torch.sum(torch.abs(wdiff),dim=(1,2))/diff.shape[0]/diff.shape[1] # average by batch and time dimensions
            dist = dist+wdiff
        return self.act(dist)

class classifnet(nn.Module):
    def __init__(self,ndim=[16,6],dp=0.1,BN=1):
        
        # lossnet is pair of [batch,L] -> dist [batch]
        # classifnet goes dist [batch] -> pred [batch,2] == evaluate BCE with low-capacity
        
        super(classifnet, self).__init__()
        n_layers = 2
        MLP = []
        for ilayer in range(n_layers):
            if ilayer==0:
                fin = 1
            else:
                fin = ndim[ilayer-1]
            MLP.append(nn.Linear(fin,ndim[ilayer]))
            if BN==1:
                MLP.append(nn.BatchNorm1d(ndim[ilayer]))
            MLP.append(nn.LeakyReLU())
            if dp!=0:
                MLP.append(nn.Dropout(p=dp))
        # last linear maps to binary class probabilities ; loss includes LogSoftmax
        MLP.append(nn.Linear(ndim[ilayer],2))
        self.MLP = nn.Sequential(*MLP)
    
    def forward(self,dist):
        return self.MLP(dist.unsqueeze(1))


###############################################################################
### full model

class JNDnet(nn.Module):
    def __init__(self,nconv=14,nchan=32,dist_dp=0.1,dist_sig=1,ndim=[16,6],classif_dp=0.1,classif_BN=0,dev=torch.device('cpu')):
        super(JNDnet, self).__init__()
        self.model_dist = lossnet(nconv=nconv,nchan=nchan,dp=dist_dp,dist_sig=dist_sig)
        self.model_classif = classifnet(ndim=ndim,dp=classif_dp,BN=classif_BN)
        self.CE = nn.CrossEntropyLoss(reduction='mean')
        self.dev = dev
#        self.apply(self.winit)
    
    def forward(self,xref,xper,labels):
        dist = self.model_dist.forward(xref,xper)
        pred = self.model_classif.forward(dist)
        loss = self.CE(pred,labels.squeeze()) # pred is [batch,2] and labels [batch] long and binary
        class_prob = F.softmax(pred,dim=-1)
        class_pred = torch.argmax(class_prob,dim=-1)
        return loss,dist,class_pred,class_prob
    
#    def winit(self, m):
#        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
#            nn.init.xavier_normal_(m.weight)
#            if m.bias is not None:
#                m.bias.data.fill_(0.01)
#        else:
#            print(m)
    
    def grad_check(self,minibatch,optimizer):
        xref = minibatch[0].to(self.dev)
        xper = minibatch[1].to(self.dev)
        labels  = minibatch[2].to(self.dev)
        
        loss,dist,class_pred,class_prob = self.forward(xref,xper,labels)
        print('\nbackward on classification loss')
        optimizer.zero_grad()
        loss.backward()
        tot_grad = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_abs_paramgrad = torch.sum(torch.abs(param.grad)).item()
                if sum_abs_paramgrad==0:
                    print(name,'sum_abs_paramgrad==0')
                else:
                    tot_grad += sum_abs_paramgrad
            else:
                print(name,'param.grad is None')
        print('tot_grad = ',tot_grad)
        
        norm_type = 2
        loss,dist,class_pred,class_prob = self.forward(xref,xper,labels)
        optimizer.zero_grad()
        loss.backward()
        total_norm = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
            else:
                print(name,'param.grad is None')
        total_norm = total_norm ** (1. / norm_type)
        print('total_norm over all layers ==',total_norm)


