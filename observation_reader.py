import numpy as np
import matplotlib.pyplot as plt

class observation_reader:
    def __init__(self, save_obsdata_path, save_obsdata_name, obj_obs_data, obj_artifical_data):
        '''
        INPUT:
            save_obsdata_path: str, the path to save data
            save_obsdata_name: str, the name to save data, canbe 'None'
            obj_obs_data: a dict including 2 1d N1 arrays, 
                    filter1: 1d N1 array, photometry filter1 magnitude
                    filter2: 1d N1 array, photometry filter2 magnitude
            obj_artifical_data: a dict including 4 1d N2 arrays
                    flux_filter1_input: 1d N2 array, artifical stars input true filter1 magnitude
                    flux_filter1_output: 1d N2 array, artifical stars output filter1 magnitude
                    flux_filter2_input: 1d N2 array, artifical stars input true filter2 magnitude
                    flux_filter2_output: 1d N2 array, artifical stars output filter2 magnitude
        '''
        self.save_obsdata_path = save_obsdata_path
        self.save_obsdata_name = save_obsdata_name
        self.obj_obs_data = obj_obs_data
        self.obj_artifical_data = obj_artifical_data
        
    def __parameter_constrain_s_to_n(self, s_to_n, thr_s_to_n):
        flt = (s_to_n[0] > thr_s_to_n) & (s_to_n[1] > thr_s_to_n)
        return flt
    def __parameter_constrain_sharpness(self, sharpness, thr_sharpness):
        flt = (np.abs(sharpness[0]) < thr_sharpness) & (np.abs(sharpness[1]) < thr_sharpness)
        return flt
    def __parameter_constrain_crowding(self, crowding, thr_crowding):
        flt = (crowding[0] < thr_crowding) & (crowding[1] < thr_crowding)
        return flt
    def __parameter_constrain_objects(self, objects, thr_objects):
        flt = (objects[0] == thr_objects) & (objects[1] == thr_objects)
        return flt
    def __parameter_constrain_flag(self, flag, thr_flag):
        flt = (flag[0] <= thr_flag) & (flag[1] <= thr_flag)
        return flt
    def __parameters_constrain(self, N_demin, parameters_data_threshold, parameters_data):
        thr_s_to_n,thr_sharpness,thr_crowding,thr_objects,thr_flag = parameters_data_threshold['thr_s_to_n'],parameters_data_threshold['thr_sharpness'],parameters_data_threshold['thr_crowding'],parameters_data_threshold['thr_objects'],parameters_data_threshold['thr_flag']
        s_to_n,sharpness,crowding,objects,flag = parameters_data['s_to_n'],parameters_data['sharpness'],parameters_data['crowding'],parameters_data['objects'],parameters_data['flag']

        flt_T_obs = np.ones((N_demin)).astype('bool')

        if s_to_n is not None:
            flt = self.__parameter_constrain_s_to_n(s_to_n, thr_s_to_n)
            flt_T_obs = flt_T_obs & flt
        if sharpness is not None:
            flt = self.__parameter_constrain_sharpness(sharpness, thr_sharpness)
            flt_T_obs = flt_T_obs & flt
        if crowding is not None:
            flt = self.__parameter_constrain_crowding(crowding, thr_crowding)
            flt_T_obs = flt_T_obs & flt
        if objects is not None:
            flt = self.__parameter_constrain_objects(objects, thr_objects)
            flt_T_obs = flt_T_obs & flt
        if flag is not None:
            flt = self.__parameter_constrain_flag(flag, thr_flag)
            flt_T_obs = flt_T_obs & flt
        return flt_T_obs
    
    def __draw_obs_CMD(self, filter1, filter2):
        f = plt.figure(figsize=(3,3))
        ax = f.add_subplot(111)
        ax.scatter(filter1-filter2, filter2, s=1,color='black')
        ax.set_xlabel('Color')
        ax.set_ylabel('Magnitude')
        ax.invert_yaxis() 
        f.savefig(self.save_obsdata_path+'/'+self.save_obsdata_name+'_obs_cmd.jpg',dpi=150,bbox_inches='tight')

    def __draw_flux_err(self, flux_filter1_input, flux_filter1_output, flux_filter2_input, flux_filter2_output):
        f = plt.figure(figsize=(3,3))
        ax = f.add_subplot(211)
        ax.scatter(flux_filter1_input, flux_filter1_output-flux_filter1_input, s=1,color='black', label='filter1')
        ax.set_xlabel('Input')
        ax.set_ylabel('Out-In')
        ax.legend()

        ax = f.add_subplot(212)
        ax.scatter(flux_filter2_input, flux_filter2_output-flux_filter2_input, s=1,color='black', label='filter2')
        ax.set_xlabel('Input')
        ax.set_ylabel('Out-In')
        ax.legend()
        
        f.savefig(self.save_obsdata_path+'/'+self.save_obsdata_name+'_flux_error.jpg',dpi=150,bbox_inches='tight')


    def read(self, if_flag = False, parameters_data_threshold = None, obs_parameters_data = None, artifical_parameters_data = None, if_draw = True):
        '''
        read and resave observation data and artifical stars. 
        INPUT:
            if_flag: bool, if true mask the obsdata through parameter and threshold.
            parameters_data_threshold: a dict including 5 parameters
                    thr_s_to_n: float, signal-to-noise ratio, typical 5-10
                    thr_sharpness: float, typical 0.2
                    thr_crowding: float, typical 0.5
                    thr_objects: float, typical 1 (good stars)
                    thr_flag: float, photometry quality flag, typical 0
                    color_range: [float, float], the cutoff color range in mag
                    mag_range: [float, float], the cutoff magnitude range in mag
            obs_parameters_data: a dict including 5 (2xN1) arrays, 2 is 2 filters
                    s_to_n: (2xN1) array, signal-to-noise ratio
                    sharpness: (2xN1) array
                    crowding: (2xN1) array
                    objects: (2xN1) array
                    flag: (2xN1), photometry quality flag
            artifical_parameters_data: a dict including 5 (2xN2) arrays, 2 is 2 filters
                    s_to_n: (2xN2) array, signal-to-noise ratio
                    sharpness: (2xN2) array
                    crowding: (2xN2) array
                    objects: (2xN2) array
                    flag: (2xN2) array, photometry quality flag
        OUTPUT:
            6 npy files
        '''
        print('#### observation restore start ####')
        filter1, filter2 = self.obj_obs_data['filter1'], self.obj_obs_data['filter2']
        flux_filter1_input, flux_filter1_output, flux_filter2_input, flux_filter2_output = self.obj_artifical_data['flux_filter1_input'], self.obj_artifical_data['flux_filter1_output'], self.obj_artifical_data['flux_filter2_input'], self.obj_artifical_data['flux_filter2_output']
        flux_filter1_output[flux_filter1_output>90] = np.nan
        flux_filter2_output[flux_filter2_output>90] = np.nan

        if if_flag == True:
            flt_T_obs_flag = self.__parameters_constrain(len(filter1), parameters_data_threshold, obs_parameters_data)
            color_mag_list = filter1 - filter2
            flt_range = (color_mag_list >= parameters_data_threshold['color_range'][0]) & (color_mag_list <= parameters_data_threshold['color_range'][1]) & (filter2 >= parameters_data_threshold['mag_range'][0]) & (filter2 <= parameters_data_threshold['mag_range'][1])
            flt_T_obs_flag = flt_T_obs_flag & (filter1<90) & (filter2<90) & flt_range
            filter1, filter2 = filter1[flt_T_obs_flag], filter2[flt_T_obs_flag]

            flt_T_artifical_flag = self.__parameters_constrain(len(flux_filter1_output), parameters_data_threshold, artifical_parameters_data)
            flux_filter1_output[~flt_T_artifical_flag] = np.nan
            flux_filter2_output[~flt_T_artifical_flag] = np.nan
        else:
            flt_T_obs = (filter1<90) & (filter2<90)
            filter1, filter2 = filter1[flt_T_obs], filter2[flt_T_obs]

        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_obs_cmdx.npy',filter1-filter2)
        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_obs_cmdy.npy',filter2)

        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_filter1_input.npy',flux_filter1_input)
        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_filter1_output.npy',flux_filter1_output)
        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_filter2_input.npy',flux_filter2_input)
        np.save(self.save_obsdata_path+'/'+self.save_obsdata_name+'_filter2_output.npy',flux_filter2_output)
        if if_draw:
            self.__draw_obs_CMD(filter1, filter2)
            self.__draw_flux_err(flux_filter1_input, flux_filter1_output,flux_filter2_input, flux_filter2_output)
        
        print('#### observation restore finish ####')



