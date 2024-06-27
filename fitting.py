import numpy as np
import torch
import matplotlib.pyplot as plt


class fitting:
    def __init__(self, observation_target_name, isochrone_path, gridding_path, save_fitting_path):
        '''
        INPUT:
            observation_target_name: str, the name to save data, canbe 'None'
            isochrone_path: str, the isochrone path
            gridding_path: str, the gridding path
            save_fitting_path: str, the path to save fitting result
        '''
        self.save_fitting_path = save_fitting_path
        self.isochrone_path = isochrone_path
        self.observation_target_name = observation_target_name
        self.gridding_path = gridding_path

    
    def multi_fit(self, fitting_bin_type,  metal_input = None, logage_input = None):
        '''
        INPUT:
            fitting_bin_type: a list. including 'uniform', 'quadtree', 'voronoi'
            metal_input: 1d array, isochrone metal list 
            logage_input: 1d array, isochrone logage list 
        '''
        isochrone_path_restore = self.isochrone_path+'/newstore/'
        if metal_input is None:
            self.metal = np.load(isochrone_path_restore+'/metal_list.npy')
        else:
            self.metal = metal_input
        if logage_input is None:
            self.logage = np.load(isochrone_path_restore+'/logage_list.npy')
        else:
            self.logage = logage_input
        for i in range(len(fitting_bin_type)):
            targ = np.load(self.gridding_path+'/'+self.observation_target_name+'_obs_'+str(fitting_bin_type[i])+'_bins.npy')
            temp = np.load(self.gridding_path+'/'+self.observation_target_name+'_temp_'+str(fitting_bin_type[i])+'_bins.npy')
            res = self.__data_fit_torch(targ, temp)
            np.save(self.save_fitting_path+'/'+self.observation_target_name+'_fitres_'+str(fitting_bin_type[i])+'.npy', res)
            self.__plot_res(np.hstack((res)), targ, temp, str(fitting_bin_type[i]))

        return None
    
    def fit(self, fitting_bin_type,  metal_input = None, logage_input = None):
        '''
        INPUT:
            fitting_bin_type: a str. including 'uniform', 'quadtree', 'voronoi'
            metal_input: 1d array, isochrone metal list 
            logage_input: 1d array, isochrone logage list 
        '''
        isochrone_path_restore = self.isochrone_path+'/newstore/'
        if metal_input is None:
            self.metal = np.load(isochrone_path_restore+'/metal_list.npy')
        else:
            self.metal = metal_input
        if logage_input is None:
            self.logage = np.load(isochrone_path_restore+'/logage_list.npy')
        else:
            self.logage = logage_input
        
        targ = np.load(self.gridding_path+'/'+self.observation_target_name+'_obs_'+str(fitting_bin_type)+'_bins.npy')
        temp = np.load(self.gridding_path+'/'+self.observation_target_name+'_temp_'+str(fitting_bin_type)+'_bins.npy')
        res = self.__data_fit_torch(targ, temp)
           
        return res

    def __data_fit_torch(self, targ, temp):
        Y = torch.tensor(targ.flatten(), dtype=torch.float64)
        X = torch.tensor(temp.reshape(temp.shape[0], -1), dtype=torch.float64)
        C = torch.rand(temp.shape[0], requires_grad=True, dtype=torch.float64)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam([C], lr=0.001)
        epoches = 50000
        min_loss = float('inf')
        patience = 100
        patience_counter = 0

        for epoch in range(epoches):
            C.data = torch.clamp(C.data, min=0)
            pred = torch.matmul(C, X)
            pred = pred / torch.sum(pred) * torch.sum(Y)
            loss = loss_func(Y, pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < min_loss:
                min_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch + 1}, Loss: {min_loss:.4f}')
                break

            if (epoch + 1) % 10000 == 0:
                print(f'Epoch [{epoch + 1}/{epoches}], Loss: {loss.item():.4f}')

        with torch.no_grad():
            res = C.numpy()
            res[res < 0] = 0

        res_new = (res*np.nansum(temp,axis=1)).reshape(len(self.metal), len(self.logage))
        res_new = res_new/np.nansum(res_new)*np.nansum(targ)

        return res_new
    
    ### 拟合结果残差图
    def __plot_res(self, res, data_bin, temp_bin, bintype):

        plt.figure(figsize=(7, 6))
        plt.subplot(221)
        y_pred = np.dot(res, temp_bin.reshape(temp_bin.shape[0], -1))
        y_pred = y_pred / np.sum(y_pred) * 100
        y_true = data_bin.flatten() / np.sum(data_bin) * 100
        plt.hist(y_true - y_pred)
        plt.xlabel('True% - Pred%')
        plt.ylabel('Count')

        plt.subplot(222)
        plt.hist(res, bins=np.logspace(-6, 4, 20))
        plt.xscale('log')
        plt.xlabel('C_factor')
        plt.ylabel('Count')
        
        plt.subplot(212)
        plt.imshow(res.reshape(len(self.metal), len(self.logage)), cmap='Spectral', aspect='auto')
        plt.xlabel('Age')
        plt.ylabel('Metal')
        plt.savefig('{}/{}_fitres_{}.png'.format(self.save_fitting_path, self.observation_target_name, bintype), dpi=300, bbox_inches='tight')
        plt.close()

        return None

            
    