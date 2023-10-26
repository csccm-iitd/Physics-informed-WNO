"""
This code defines grident functions
"""

import torch
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

#from scipy.stats import qmc
import numpy as np

#Set default dtype to float32
torch.set_default_dtype(torch.float)
torch.manual_seed(1234)
np.random.seed(1234)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')
# device= torch.device('cpu')

class gradientfree(nn.Module):
    def __init__(self):
        super(gradientfree,self).__init__() #call __init__ from parent class 
        self.loss_function = nn.MSELoss(reduction ='sum')
        self.n_neighbour = 9
        self.radius= 0.024
        
    def neighbour_index(self,X_f_train):
        nr = self.n_neighbour
        r = self.radius
        zn = torch.zeros([X_f_train.shape[0],nr],dtype=torch.int32).to(device)
        for j in range(X_f_train.shape[0]): 
            ngr1 = (torch.ones([1,nr],dtype=torch.int32)*j)[0].tolist()
            x = X_f_train[j,:][None,:]
            n = x.shape[1]
            ngr1_d= torch.where(torch.linalg.norm(X_f_train-x,axis=1)< r)[0].tolist()
            ngr1[0:len(ngr1_d)] = ngr1_d
            zn[j,:] = torch.tensor(ngr1)[0:nr]
        return zn 
    
    def inverse_index(self,X_f_train):
        nr = self.n_neighbour
        r = self.radius
        zn = torch.zeros([X_f_train.shape[0],nr],dtype=torch.int32).to(device)
        zmd = torch.zeros([X_f_train.shape[0],X_f_train.shape[1],X_f_train.shape[1]]).to(device)
        for j in range(X_f_train.shape[0]): 
                ngr1 = (torch.ones([1,nr],dtype=torch.int32)*j)[0].tolist()
                x = X_f_train[j,:][None,:]
                n = x.shape[1]
                md = torch.zeros((n,n)).to(device);
                ngr1_d= torch.where(torch.linalg.norm(X_f_train-x,axis=1)< r)[0].tolist()
                ngr1[0:len(ngr1_d)] = ngr1_d
                zn[j,:] = torch.tensor(ngr1)[0:nr]
                xd = X_f_train[ngr1_d[0:nr],:][None,:]-x;
                md = torch.einsum('abi,abj->aij',xd,xd)
                md_inv = torch.inverse(md)
                zmd[j,:,:] = md_inv
        return zmd
    
    def loss_BC(self,up,usol):
        loss_u = self.loss_function(up,usol)
        return loss_u
    
    def grad1(self,xx,u_0,n_index,inv_mat):
        zd = torch.zeros(xx.shape, dtype=torch.float)
        m = n_index.shape[1]
        x_ngr = xx[n_index.tolist(),:]
        u_ngr = u_0[n_index.tolist(),:]
        x_d  = x_ngr-xx.unsqueeze(1).repeat(1,m,1)
        u_d = u_ngr-u_0.unsqueeze(1).repeat(1,m,1)
        u_ds = torch.sum(torch.einsum('ijp,ikp->ikp',(u_d.permute(0,2,1)),(x_d.permute(0,2,1))),dim=2)
        zd = torch.einsum('kj,kji->ki',u_ds,inv_mat)
        return zd
    
    def grad2(self,xx,u_x_t,n_index,inv_mat):
        zdd = torch.zeros(u_x_t.shape[0], 2*u_x_t.shape[1],dtype=torch.float)
        n = u_x_t.shape[0]
        m = n_index.shape[1]
        x_ngr = xx[n_index.tolist(),:]
        u_xngr = u_x_t[n_index.tolist(),:]
        x_d  = x_ngr-xx.unsqueeze(1).repeat(1,m,1)
        u_xd = u_xngr-u_x_t.unsqueeze(1).repeat(1,m,1)
        u_xds = torch.sum(torch.einsum('ijr,ipr->ijpr',(u_xd.permute(0,2,1)),(x_d.permute(0,2,1))),dim=3)
        zdd = torch.einsum('kij,kjl->kil',u_xds,inv_mat).reshape([n,4])
        return zdd
    
    def loss_PDE(self,ut,x_to_train_f,ut1,n_index,inv_mat):
        
        nu = 0.01/np.pi
                                
        u_x_y = self.grad1(x_to_train_f,ut1,n_index,inv_mat)                                        
        u_xx_yy = self.grad2(x_to_train_f,u_x_y,n_index,inv_mat)                                                                                                              
        # u_x = u_x_y[:,[0]]
        # u_y = u_x_y[:,[1]]
      
        u_xx = u_xx_yy[:,[0]]
        u_yy = u_xx_yy[:,[3]]
        
        f = ut1-ut - 0.02*(0.01*(u_xx+u_yy) + ut-ut**3)
        
        f_hat =  torch.zeros(x_to_train_f.shape[0],1).to(device)
        loss_f = self.loss_function(f,f_hat)
        return  loss_f
    
    
      
    def loss_PDE1(self,ut,x_to_train_f,ut1,n_index,inv_mat):
        
        
        nu = 0.01/np.pi
                                
        u_x_y = self.grad1(x_to_train_f,ut,n_index,inv_mat)                                        
        u_xx_yy = self.grad2(x_to_train_f,u_x_y,n_index,inv_mat)                                                                                                              
        # u_x = u_x_y[:,[0]]
        # u_y = u_x_y[:,[1]]
      
        u_xx = u_xx_yy[:,[0]]
        u_yy = u_xx_yy[:,[3]]
        
                                  
        u_x_y1 = self.grad1(x_to_train_f,ut1,n_index,inv_mat)                                        
        u_xx_yy1 = self.grad2(x_to_train_f,u_x_y1,n_index,inv_mat)                                                                                                              
        # u_x = u_x_y[:,[0]]
        # u_y = u_x_y[:,[1]]
      
        u_xx1 = u_xx_yy1[:,[0]]
        u_yy1 = u_xx_yy1[:,[3]]
        
        
        f = ut1-ut - 0.5*0.02*((0.01*(u_xx+u_yy) + ut-ut**3)+(0.01*(u_xx1+u_yy1) + ut1-ut1**3))
        
        f_hat =  torch.zeros(x_to_train_f.shape[0],1).to(device)
        loss_f = self.loss_function(f,f_hat)
        
        return  loss_f
    
    
    
    
    def loss(self,up,usol,ut,x_to_train_f,ut1,n_index,inv_mat):

        loss_u = self.loss_BC(up,usol)
        loss_f = self.loss_PDE1(ut,x_to_train_f,ut1,n_index,inv_mat)
        
        loss_val = loss_u + 4*loss_f
        return loss_val
     