from IPython import get_ipython
get_ipython().magic('reset -sf')


# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from wno_2d_time_cwt_AC import *
from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
import gradfree_fun 
torch.manual_seed(0)
np.random.seed(0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')
# device= torch.device('cpu')

# %%
""" Model configurations """

TRAIN_PATH = 'Allen_Cahn_pde_65_65_1000.mat'
# TEST_PATH = '/DATA/Phy_wno Allencan/Allen_Cahn_pde_129_129_1000.mat'

ntrain = 600
ntest = 20

batch_size = 10
learning_rate = 0.0001

epochs = 400
step_size = 10
gamma = 0.5

level = 4 # The automation of the mode size will made and shared upon acceptance and final submission
width = 32

r = 1
h = int(((65 - 1)/r) + 1)
s = h
T_in = 10
T = 10
step = 1
S = 65

# %%
""" Read data """
reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('sol').permute(3,0,1,2)
# y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

train_a = x_train[:ntrain,:,:,:T_in]
train_u = x_train[:ntrain,:,:,T_in:T+T_in]

test_a = x_train[-ntest:,:,:,:T_in]
test_u = x_train[-ntest:,:,:,T_in:T+T_in]


train_a = train_a.reshape(ntrain,s,s,T_in)
test_a = test_a.reshape(ntest,s,s,T_in)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=batch_size, shuffle=False)

# %%
lb = np.array([0, 0])
ub = np.array([1, 1])
N_f = S
xt1=lb[0] + (ub[0]-lb[0])*np.linspace(0,1,N_f)
yt1=lb[1] + (ub[1]-lb[1])*np.linspace(0,1,N_f)
Xt1, Yt1 = np.meshgrid(xt1,yt1)
X_f_train = np.hstack([Xt1.reshape(N_f*N_f,1),Yt1.reshape(N_f*N_f,1)])
x_f_train = torch.tensor(X_f_train,dtype=torch.float).to(device)

gf = gradfree_fun.gradientfree()
p_index = gf.neighbour_index(x_f_train)
invp_index = gf.inverse_index(x_f_train)

# %%
""" The model definition """
model = WNO2d(width, level, train_a[0:1,:,:,:].permute(0,3,1,2)).to(device)
print(count_params(model))

""" Training and testing """
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

train_loss = torch.zeros(epochs)
test_loss = torch.zeros(epochs)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_dl = 0
    train_pl = 0
    for xx, yy in train_loader:
        data_loss = 0
        xx = xx.to(device)
        yy = yy.to(device)     
        loss_physics = 0  
        for t in range(0, T, step):
            y = yy[..., t:t + step]  # t:t+step, retains the third dimension, butt only t don't,
            im = model(xx)  
            # data_loss += F.mse_loss(im, y, reduction = 'sum') 
            
            if t == 0: 
                pred = im
            else:
                pred = torch.cat((pred, im), -1)
    
            xx = torch.cat((xx[..., step:], im), dim=-1)
            
            for kk in range(im.shape[0]):
                y_pred = im[kk,:,:,:].squeeze(-1)
                y_pf = y_pred.reshape(N_f*N_f,1)
                x_pf = xx[kk,:,:,-2].reshape(N_f*N_f,1)
                y_dash = y[kk,:,:,:].squeeze(-1)
                #bound condition 1
                tp_u = y_pred[0,:][:,None]
                tp_usol = y_dash[0,:][:,None] 
                bt_u = y_pred[-1,:][:,None]
                bt_usol = y_dash[-1,:][:,None] 
                
                lt_u =  y_pred[:,0][:,None] #L2
                lt_usol = y_dash[:,0][:,None] #L2
                rt_u = y_pred[:,-1][:,None]
                rt_usol = y_dash[:,-1][:,None]
                
                all_u_train = torch.vstack([tp_u,bt_u,lt_u,rt_u]) # X_u_train [200,2] (800 = 200(L1)+200(L2)+200(L3)+200(L4))
                all_u_sol = torch.vstack([tp_usol,bt_usol,lt_usol,rt_usol])   #corresponding u [800x1]
                
                loss_physics += gf.loss(all_u_train,all_u_sol,x_pf,x_f_train,y_pf,p_index,invp_index)
        
        loss =  loss_physics
        loss.backward()
        optimizer.step()
        
        train_l2 += loss.item()
        train_dl += 0
        train_pl += loss_physics.item()
        
    scheduler.step()
    model.eval()
    test_l2_step = 0
    test_l2_full = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
        
            for t in range(0, T, step):
                y = yy[..., t:t + step]
                im = model(xx)
                loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
        
                if t == 0:
                    pred = im
                else:
                    pred = torch.cat((pred, im), -1)
        
                xx = torch.cat((xx[..., step:], im), dim=-1)
                
            test_l2_step += loss.item()
            test_l2_full += (torch.norm(pred-yy, p=2)/torch.norm(yy, p=2)).item()
            
            
    train_l2 /= (ntrain*T)
    train_dl /= (ntrain*T)
    train_pl /= (ntrain*T)
    
    test_l2_step /= (ntest*T)
    train_loss[ep] = train_l2
    test_loss[ep] = test_l2_step
    
    t2 = default_timer()
    print('Epoch %d - Time %0.4f - Train %0.4f - PDE %0.4f - data %0.4f - Test %0.4f' 
          % (ep, t2-t1, train_l2, train_pl, train_dl, test_l2_step))    
    
    
# %%
""" Prediction """
model = torch.load('model/ns_wno_allencan_p_3.5mse1')
pred0 = torch.zeros(test_u.shape)
index = 0
test_e = torch.zeros(test_u.shape)        
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)

with torch.no_grad():
    for xx, yy in test_loader:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)

        for t in range(0, T, step):
            y = yy[..., t:t + step]
            im = model(xx)
            # loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)
            loss += (torch.mean((im-y)**2)/torch.mean(y**2))
            # loss += (torch.linalg.norm(im-y,dim=(1,2,3))/torch.linalg.norm(y,dim=(1,2,3)))

            if t == 0:
                pred = im
            else:
                pred = torch.cat((pred, im), -1)

            xx = torch.cat((xx[..., step:], im), dim=-1)
            
        pred0[index] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.mean((pred-yy)**2)/torch.mean(yy**2)).item()
        # test_l2_full += (torch.linalg.norm(pred-yy,dim=(1,2,3))/torch.linalg.norm(yy,dim=(1,2,3))).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
        index = index + 1
print('Mean Testing Error:', 100*torch.mean(test_e).numpy() / (T/step), '%')
print('STD Testing Error:', 100*torch.std(test_e).numpy() / (T/step), '%')
# %%
""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
figure7 = plt.figure(figsize = (10, 5))
plt.subplots_adjust(hspace=0.04)

batch_no = 10
index = 0
for tvalue in range(test_u.shape[3]):
    if tvalue % 4 == 1:
        plt.subplot(3,5, index+1)
        plt.imshow(test_u.numpy()[batch_no,:,:,tvalue], label='True', cmap='seismic',vmin=-0.3,vmax=0.5)
        test_u.numpy()[batch_no,:,:,tvalue]
        plt.colorbar()
        plt.title('Actual')
        plt.subplot(3,5, index+1+5)
        plt.imshow(pred0.cpu().detach().numpy()[batch_no,:,:,tvalue], cmap='seismic',vmin=-0.3,vmax=0.5)
        plt.colorbar()
        plt.title('Identified')
        plt.subplot(3,5, index+1+10)
        plt.imshow(np.abs(pred0.cpu().detach().numpy()[batch_no,:,:,tvalue]-test_u.numpy()[batch_no,:,:,tvalue]), cmap='jet',vmin=0.0,vmax=0.2)
        plt.colorbar()
        plt.title('Error')
        plt.margins(0)
        index = index + 1


# # %%
# """
# For saving the trained model and prediction data
# """
#%%
torch.save(model, 'model/ns_wno_allencan_p_3.5mse1')
scipy.io.savemat('pred/ns_wno_allencan_p_3.5mse1.mat', mdict={'pred': pred0.cpu().numpy()})
scipy.io.savemat('loss_epoch/train_loss_ns_wno_allencan.mat', mdict={'train_loss': train_loss.cpu().numpy()})
scipy.io.savemat('loss_epoch/pred_loss_ns_wno_allencan.mat', mdict= {'test_loss': test_loss.cpu().numpy()})
#%%
pred_t = pred0
test_u_t = test_u 
test_e_t = torch.zeros([10,1])
for t in range(0,T,step):
    test_e_t[t]= (torch.norm(pred[:,:,:,t]-yy[:,:,:,t], p=2)/torch.norm(yy[:,:,:,t], p=2)).item()
# torch.cuda.empty_cache()
# %%
# """ Prediction """
# predt = torch.zeros(test_u.shape)
# data_grid = torch.zeros(test_u.shape)
# index = 0
# test_e = torch.zeros(test_u.shape)        
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
# k_gridp1 = torch.zeros(100,1)
# k_gridt1 = torch.zeros(100,1)
# k_gridp2 = torch.zeros(100,1)
# k_gridt2 = torch.zeros(100,1)
# k_gridp3 = torch.zeros(100,1)
# k_gridt3 = torch.zeros(100,1)

# mean1_image_predicted = torch.zeros(100,65,65)
# mean1_image_actual = torch.zeros(100,65,65)

# mean2_image_predicted = torch.zeros(100,65,65)
# mean2_image_actual = torch.zeros(100,65,65)

# mean3_image_predicted = torch.zeros(100,65,65)
# mean3_image_actual = torch.zeros(100,65,65)

# with torch.no_grad():
#     for xx, yy in test_loader:
#         test_l2_step = 0
#         test_l2_full = 0
#         loss = 0
#         xx = xx.to(device)
#         yy = yy.to(device)

#         for t in range(0, T, step):
#             y = yy[..., t:t + step]
#             im = model(xx)
#             loss += torch.norm(im-y, p=2)/torch.norm(y, p=2)

#             if t == 0:
#                 pred = im
#             else:
#                 pred = torch.cat((pred, im), -1)

#             xx = torch.cat((xx[..., step:], im), dim=-1)
           
#         predt[index] = pred
#         test_l2_step += loss.item()
#         test_l2_full += (torch.norm(pred-yy, p=2)/torch.norm(yy, p=2)).item()
        
#         k_gridp1[index,:] = pred[:,20,20,9]
#         k_gridt1[index,:] = yy[:,20,20,9]
#         k_gridp2[index,:] = pred[:,30,30,9]
#         k_gridt2[index,:] = yy[:,30,30,9]
#         k_gridp3[index,:] = pred[:,40,40,9]
#         k_gridt3[index,:] = yy[:,40,40,9]
        
#         mean1_image_predicted[index,:,:] = pred[0,:,:,9]
#         mean1_image_actual[index,:,:] = yy[0,:,:,9]
#         mean2_image_predicted[index,:,:] = pred[0,:,:,6]
#         mean2_image_actual[index,:,:] = yy[0,:,:,6]
#         mean3_image_predicted[index,:,:] = pred[0,:,:,3]
#         mean3_image_actual[index,:,:] = yy[0,:,:,3]
        
#         test_e[index] = test_l2_step
        
#         print(index, test_l2_step/ ntest/ (T/step), test_l2_full/ ntest)
#         index = index + 1
        
# print('Mean Testing Error:', 100*torch.mean(test_e).numpy() /(T/step), '%')

# plot_mean1p = torch.mean(mean1_image_predicted,axis=0)
# plot_mean1t = torch.mean(mean1_image_actual,axis=0)
# plot_std1p = torch.std(mean1_image_predicted,axis=0)
# plot_std1t = torch.std(mean1_image_actual,axis=0)

# plot_mean2p = torch.mean(mean2_image_predicted,axis=0)
# plot_mean2t = torch.mean(mean2_image_actual,axis=0)
# plot_std2p = torch.std(mean2_image_predicted,axis=0)
# plot_std2t = torch.std(mean2_image_actual,axis=0)

# plot_mean3p = torch.mean(mean3_image_predicted,axis=0)
# plot_mean3t = torch.mean(mean3_image_actual,axis=0)
# plot_std3p = torch.std(mean3_image_predicted,axis=0)
# plot_std3t = torch.std(mean3_image_actual,axis=0)


# plt.subplot(1,4,1)
# plt.imshow(plot_mean1p, label='True', cmap='jet',vmin=-0.015,vmax=0.02)
# plt.title('Mean_pred')
# plt.colorbar()
# plt.subplot(1,4,2)
# plt.imshow(plot_mean1t, label='True', cmap='jet',vmin=-0.015,vmax=0.02)
# plt.title('Mean_Actual')
# plt.colorbar()
# plt.subplot(1,4,3)
# plt.imshow(plot_std1p, label='True', cmap='jet',vmin=-0.0,vmax=0.14)
# plt.title('STD_pred')
# plt.colorbar()
# plt.subplot(1,4,4)
# plt.imshow(plot_std1t, label='True', cmap='jet',vmin=-0.0,vmax=0.14)
# plt.title('STD_Actual')
# plt.colorbar()


# plt.subplot(1,4,5)
# plt.imshow(plot_mean2p, label='True', cmap='jet')
# plt.title('Mean_pred')
# plt.subplot(1,4,6)
# plt.imshow(plot_mean2t, label='True', cmap='jet')
# plt.title('Mean_Actual')
# plt.subplot(1,4,7)
# plt.imshow(plot_std2p, label='True', cmap='jet')
# plt.title('STD_pred')
# plt.subplot(1,4,)
# plt.imshow(plot_std2t, label='True', cmap='jet')
# plt.title('STD_Actual')


# plt.subplot(1,4,1)
# plt.imshow(plot_mean3p, label='True', cmap='jet')
# plt.title('Mean_pred')
# plt.subplot(1,4,2)
# plt.imshow(plot_mean3t, label='True', cmap='jet')
# plt.title('Mean_Actual')
# plt.subplot(1,4,3)
# plt.imshow(plot_std3p, label='True', cmap='jet')
# plt.title('STD_pred')
# plt.subplot(1,4,4)
# plt.imshow(plot_std3t, label='True', cmap='jet')
# plt.title('STD_Actual')


# plt.subplot(1,4,1)
# plt.imshow(plot_mean2p, label='True', cmap='jet',vmin=-0.06,vmax=0.08)
# plt.colorbar()
# plt.title('Mean_pred')
# plt.subplot(1,4,2)
# plt.imshow(plot_mean2t, label='True', cmap='jet',vmin=-0.06,vmax=0.08)
# plt.colorbar()
# plt.title('Mean_Actual')
# plt.subplot(1,4,3)
# plt.imshow(plot_std2p, label='True', cmap='jet',vmin=0.14,vmax=0.28)
# plt.colorbar()
# plt.title('STD_pred')
# plt.subplot(1,4,4)
# plt.imshow(plot_std2t, label='True', cmap='jet',vmin=0.14,vmax=0.28)
# plt.colorbar()
# plt.title('STD_Actual')

# np.savetxt('tf1p.txt',k_gridp1)
# np.savetxt('tf1t.txt',k_gridt1)
# np.savetxt('tf2p.txt',k_gridp2)
# np.savetxt('tf2t.txt',k_gridt2)
# np.savetxt('tf3p.txt',k_gridp3)
# np.savetxt('tf3t.txt',k_gridt3)

# import seaborn as sb
# fig = plt.figure(figsize=(10,6))
# # sb.distplot(torch.cat([k_gridp,k_gridt],-1).cpu().detach().numpy(),kind="kde", bw_adjust=2,rug=True, label ='test_label1')
# sb.distplot(torch.cat([k_gridp,k_gridt],-1).cpu().detach().numpy(),hist =False,bins=50, label ='test_label1')
# # sb.displot(k_gridt.cpu().detach().numpy(),kind="kde", bw_adjust=2,rug=True)
# # fig.legend(labels=['test_label1','test_label2'])


# # %%

# sb.displot(torch.cat([k_gridp,k_gridt],-1).cpu().detach().numpy(),kind="kde", bw_adjust=2,rug=True, label ='Predicted',legend= 'Predicted')
# plt.legend()

# from scipy.stats.kde import gaussian_kde
# from numpy import linspace
# import matplotlib.pyplot as plt

# kde = gaussian_kde(k_gridt.cpu().detach().numpy())
# dist_space = linspace( 0, 4, 100 )
# plt.plot( dist_space, kde(dist_space))
# kde = gaussian_kde(k_gridt.cpu().detach().numpy())
# dist_space = linspace(0, 4, 100 )
# plt.plot( dist_space, kde(dist_space))
