import numpy as np
from qthist2d import qthist, qtcount
from collections import deque
import matplotlib.pyplot as plt

class gridding:
    def __init__(self, observation_data_path, observation_target_name, template_path, save_gridding_path):
        '''
        INPUT:
            observation_data_path: str, the path to save data
            observation_target_name: str, the name to save data, canbe 'None'
            template_path: str, the template path
            save_gridding_path: str, the path to save gridding result
        '''
        self.observation_target_name = observation_target_name
        self.save_gridding_path = save_gridding_path

        self.obs_cmd_x = np.load(observation_data_path+'/'+self.observation_target_name+'_obs_cmdx.npy')
        self.obs_cmd_y = np.load(observation_data_path+'/'+self.observation_target_name+'_obs_cmdy.npy')
        filter1_template_list = np.load(template_path+'/template_'+self.observation_target_name+'_filter1.npy')
        filter2_template_list = np.load(template_path+'/template_'+self.observation_target_name+'_filter2.npy')
        
        self.obs_cmd, self.temp_cmd = np.array([self.obs_cmd_x, self.obs_cmd_y]), np.array([filter1_template_list - filter2_template_list, filter2_template_list]).transpose(1, 0, 2)

    def uniform_bin(self, bin_num_CMDx = 100, bin_num_CMDy = 100):
        '''
        uniform_bin, The edge is from observation CMD 
        INPUT:
            bin_num_CMDx: int, the number of bins in color axis
            bin_num_CMDy: int, the number of bins in magnitude axis
        OUTPUT:
            uniform bin result of observation and template CMD
        '''
        print('#### uniform bin start ####')
        uniform_bins = self.__get_uniform_bins(self.obs_cmd_x, self.obs_cmd_y, bin_num_CMDx, bin_num_CMDy)
        obs_uniform_data, temp_uniform_data = self.__get_uniform_data(self.obs_cmd, self.temp_cmd, uniform_bins)

        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_obs_uniform_bins.npy', obs_uniform_data)
        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_temp_uniform_bins.npy', temp_uniform_data)
        print('#### uniform bin end ####')
        return None# return obs_uniform_data, temp_uniform_data


    def __get_uniform_bins(self, obs_x, obs_y, bin_num_x, bin_num_y):
    
        bins = [
            np.linspace(np.floor(obs_x.min()), np.ceil(obs_x.max()), bin_num_x + 1), 
            np.linspace(np.floor(obs_y.min()), np.ceil(obs_y.max()), bin_num_y + 1)
        ]
        
        return bins

    def __get_uniform_data(self, obs, mod, bins):
        targ               = np.hstack((np.histogram2d(obs[0], obs[1], bins=bins)[0]))
        temp               = np.zeros((mod.shape[0], (len(bins[0])-1)*(len(bins[1])-1)))
        for i in range(mod.shape[0]):
            mod_xi, mod_yi = mod[i, 0], mod[i, 1]
            temp[i]        = np.hstack((np.histogram2d(mod_xi[~np.isnan(mod_xi)], mod_yi[~np.isnan(mod_xi)], bins=bins)[0]))
        
        return targ, temp

        
    def quad_bin(self, thresh = 5):
        '''
        quadtree bin, based on https://github.com/jradavenport/qthist2d
        INPUT:
            thresh: the threshold number of stars, typical 5
        '''
        print('#### quadtree bin start ####')
        obs_quad_data, xmin, xmax, ymin, ymax = qthist(self.obs_cmd[0], self.obs_cmd[1], N=10, thresh=thresh, density=False)
        
        xmin[xmin==xmin.min()] = xmin.min() - 1
        xmax[xmax==xmax.max()] = xmax.max() + 1
        ymin[ymin==ymin.min()] = ymin.min() - 1
        ymax[ymax==ymax.max()] = ymax.max() + 1
        self.__plot_quad(xmin, xmax, ymin, ymax)
        
        temp_quad_data = np.zeros((self.temp_cmd.shape[0], len(obs_quad_data)))
        for i in range(self.temp_cmd.shape[0]):
            temp_quad_data[i] = qtcount(self.temp_cmd[i][0], self.temp_cmd[i][1], xmin, xmax, ymin, ymax, density=False)
        
        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_obs_quadtree_bins.npy', obs_quad_data)
        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_temp_quadtree_bins.npy', temp_quad_data)
        print('#### quadtree bin end ####')
        return None #return obs_quad_data, temp_quad_data
    
    def __plot_quad(self, xmin, xmax, ymin, ymax):
        obs_x, obs_y = self.obs_cmd
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_subplot(111)
        plt.scatter(obs_x, obs_y, alpha=0.5, facecolor='black', lw=0.1, s=1)
        ax.invert_yaxis() 
        for k in range(len(xmin)):
            ax.add_patch(plt.Rectangle((xmin[k], ymin[k]), xmax[k]-xmin[k], ymax[k]-ymin[k], fc ='none', ec='k', alpha=0.5, lw=0.2))
        plt.savefig(self.save_gridding_path + '/'+self.observation_target_name+'_quad_bin.jpg', dpi=150, bbox_inches='tight')
        plt.close()
        return None
    
    def voronoi_bin(self, thresh):
        '''
        voronoi bin
        INPUT:
            thresh: the threshold number of stars
        '''
        print('#### voronoi bin start ####')
        voronoi = AutoBin(self.obs_cmd_x, self.obs_cmd_y, len(self.obs_cmd_y)/thresh, 
                          [np.nanmin(self.obs_cmd_x),np.nanmax(self.obs_cmd_x)], [np.nanmin(self.obs_cmd_y),np.nanmax(self.obs_cmd_y)])
        plt.savefig(self.save_gridding_path + '/'+self.observation_target_name+'_voronoi_bin.jpg', dpi=150, bbox_inches='tight')
        plt.close()

        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_obs_voronoi_bins.npy', np.array(voronoi.data()))

        temp_specmockbin = []
        for numlist in range(len(self.temp_cmd)):
            test_x2, test_y2 = self.temp_cmd[numlist]
            flt = np.isnan(test_x2) | np.isnan(test_y2)
            test_x2 = test_x2[~flt]
            test_y2 = test_y2[~flt]
            step = voronoi.temp(test_x2,test_y2)
            temp_specmockbin.append(step)
            plt.close()

        np.save(self.save_gridding_path+'/'+self.observation_target_name+'_temp_voronoi_bins.npy', np.array(temp_specmockbin))
        print('#### voronoi bin end ####')
        return None



class AutoBin:
    def __init__(self, cmdx, cmdy, bin_num ,x_range, y_range):
        self.cmdx = np.array(cmdx)
        self.cmdy = np.array(cmdy)
        # bin_num  and  average  star number/bin
        flt = (min(x_range)<self.cmdx)&(self.cmdx<max(x_range))&(min(y_range)<self.cmdy)&(self.cmdy<max(y_range))
        self.cmdx = np.array(cmdx)[flt]
        self.cmdy = np.array(cmdy)[flt]
        self.bin_num = bin_num
        self.star_num = len(self.cmdx) / self.bin_num
        # set first bin
        self.resoxx = 0.01
        self.resoyy = 0.05
        self.resox = int(np.ceil((max(x_range) - min(x_range)) / self.resoxx))
        self.resoy = int(np.ceil((max(y_range) - min(y_range)) / self.resoyy))
        print('star number',len(self.cmdy))
        print('average',self.star_num)
        print('small ceil', self.resox, self.resoy)
        # first bin
        self.binet = np.histogram2d(self.cmdx, self.cmdy, bins=[self.resox, self.resoy], range=[x_range, y_range])[0]
        print('max ceil num', np.max(self.binet))
        # generate empty count vector storing the number of stars in each bin # generate empty bin method storing the shape of each bin
        self.bin_count = []
        self.bin_method = []
        self.bin_cen_mass = []
        #mask
        self.Num_masked_bin = 0
        self.Num_masked_pos = []
        #limit the bin area
        self.magallow = 1.
        self.areaallow = self.magallow**2/self.resoxx/self.resoyy

        #Save bright, bright range
        self.bricolor = 0.8
        self.brimag = 24.

        # the range and the size of binet
        self.xmin = min(x_range)
        self.xmax = max(x_range)
        self.ymin = min(y_range)
        self.ymax = max(y_range)
        self.xrang = self.xmax - self.xmin
        self.x_range = x_range
        self.xpx = self.xrang / self.resox
        self.y_range = y_range
        self.yrang = self.ymax - self.ymin
        self.ypx = self.yrang / self.resoy
        # bin process
        # First bin
        self.bin_count_irr()
        print('bin done')
        #merger left upper part(save the bright star)
        #self.Save_bright()
        self.vis()
        return 
    
    def bin_count_irr(self):
        queue = deque()#around
        visited = deque()# binned
        remn = [[i, j] for i in range(self.resox) for j in range(self.resoy)] # pixel base

        # put the most dense pixel into the queue, As the start position of first bin
        ind = int(np.argmax(self.binet))#dense pixel position
        queue.append([ind // self.resoy, ind % self.resoy])
        self.cent_mass = np.array([ind // self.resoy, ind % self.resoy])

        # two adjacent directions are adjacent in coordination
        dire =  [[0, 1], [-1, 0], [1, 0], [0, -1]]
        suc = 0 #print the process of bin Num

        while len(remn)>0: #suc < self.bin_num and
            # count the summed stars
            summed = 0
            # store bin coordinates
            thisbin = []
            Roundness = 0
            while summed < self.star_num and len(queue) != 0 and Roundness<=0.5 and len(thisbin) < self.areaallow and len(remn)>0:
                #sort by distance
                queue = deque(sorted(queue, key=self.calc_dis))
                popped = queue.popleft()
                # sum at the picking moment, not the exploring moment
                summed = summed + self.binet[popped[0], popped[1]]
                # mark the summed binnet as visited, to prevent repetitive searching
                visited.append(popped)
                remn.remove(popped)
                thisbin.append(popped)
                #judge by roundness
                self.cent_mass = np.sum(np.array(thisbin),axis=0)/len(thisbin)
                bin_r_max = np.sqrt(self.lis_dis(np.array(thisbin).T)).max()
                bin_r_eff = np.sqrt(len(thisbin)/np.pi)
                Roundness = bin_r_max/bin_r_eff-1

                for i in range(4):
                    mov = dire[i]
                    new_x = popped[0] + mov[0]
                    new_y = popped[1] + mov[1]
                    new_cor = [new_x, new_y]
                    if 0 <= new_x < self.resox and 0 <= new_y < self.resoy and (visited.count(new_cor) == 0) and (queue.count(new_cor) == 0):
                        queue.append(new_cor)

            suc = suc + 1
            #print(suc)
            self.bin_method.append(thisbin)
            self.bin_count.append(summed)

            # recursive : use the nearest pixel in remn
            self.cent_mass = np.sum(np.array(thisbin),axis=0)/len(thisbin)
            self.bin_cen_mass.append(self.cent_mass)
            queue = deque()
            sorted_remn = sorted(remn, key=self.calc_dis)
            if len(sorted_remn)>0:
                self.cent_mass = sorted_remn[0]
                queue.append(sorted_remn[0])

        print('Num',len(self.bin_count))
        #dealing the remn
        if len(remn)>0:
            for rest in range(len(remn)):
                remn_dis_2 = np.array([np.sum((np.array(remn[rest])-self.bin_cen_mass[i])**2) for i in range(len(self.bin_cen_mass))])
                Add_tobin = int(np.argmin(remn_dis_2))
                self.bin_method[Add_tobin].append(remn[rest])
                self.bin_count[Add_tobin] = self.bin_count[Add_tobin] + self.binet[remn[rest][0], remn[rest][1]]
                self.bin_cen_mass[Add_tobin] = np.sum(np.array(self.bin_method[Add_tobin]),axis=0)/len(self.bin_method[Add_tobin])

    def vis(self):
        vor_bin_list = np.ones((self.resox, self.resoy))
        draw_num = np.random.rand() * 1000
        for binm in range(len(self.bin_count)):
            for binn in range(len(self.bin_method[binm])):
                vor_bin_list[self.bin_method[binm][binn][0], self.bin_method[binm][binn][1]] = draw_num

            draw_num = np.random.rand() * 1000

        for binm in range(len(self.Num_masked_pos)):
            for binn in range(len(self.Num_masked_pos[binm])):
                vor_bin_list[self.Num_masked_pos[binm][binn][0], self.Num_masked_pos[binm][binn][1]] = -np.inf

        figure = plt.figure(figsize=(3,3))
        ax = figure.add_subplot(111)
        ax.scatter((self.cmdx - self.xmin) / self.xpx, (self.cmdy - self.ymin) / self.ypx,color= 'black', alpha=0.5,s=0.1)
        ax.imshow(vor_bin_list.T, alpha=0.9, aspect='auto',cmap='Reds')
        ax.set_xticks(ticks=np.linspace(0, self.resox, 2),
                   labels=np.round((self.xmin + np.linspace(0, self.resox, 2) * self.xpx), 2))
        ax.set_yticks(ticks=np.linspace(0, self.resoy, 10),
                   labels=np.round((self.ymin + np.linspace(0, self.resoy, 10) * self.ypx), 2))
        ax.set_xlim([0,self.resox])
        ax.set_ylim([self.resoy,0])

    def data(self):
        return self.bin_count

    def temp(self, tcmdx, tcmdy):
        temp_bin = np.histogram2d(tcmdx, tcmdy, bins=[self.resox, self.resoy], range=[self.x_range, self.y_range])[0]
        temp_bin_count = []
        for i in range(len(self.bin_method)):
            summed = 0
            for j in range(len(self.bin_method[i])):
                indx = self.bin_method[i][j][0]
                indy = self.bin_method[i][j][1]
                summed = summed + temp_bin[indx][indy]
            temp_bin_count.append(summed)
        return temp_bin_count

    def calc_dis(self, coor):
        return (coor[0]-self.cent_mass[0])**2+(coor[1]-self.cent_mass[1])**2
        #return np.sum((np.array(coor)-self.cent_mass)**2)

    def lis_dis(self,coor):
        return np.sqrt((coor[0]-self.cent_mass[0])**2+(coor[1]-self.cent_mass[1])**2)