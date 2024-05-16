##### path and name #####
observation_data_path = './example/ddo210/outputs'
observation_target_name = 'ddo210'
isochrone_path = './isochrone/HST-WFPC2/'
template_path = './example/ddo210/outputs'

##### task selection #####
run_observation_reader = False
run_PARSEC_restore = False
run_template_builder = True

######      observation_reader taske          ######
###### read observational data and artificial ######
## In this example, 
## the observational photometry and artificial comes from 
## http://astronomy.nmsu.edu/logphot
## saved in './example/ddo210/photometry/'
## ddo210_u67203.out and ddo210_u67203.comp, respectively
import pandas as pd
import numpy as np
p = pd.read_table('./example/ddo210/photometry/ddo210_u67203.out',skiprows=3,sep='\s+')
p_err = pd.read_table('./example/ddo210/photometry/ddo210_u67203.comp',skiprows=1,sep='\s+')

# save_obsdata_path: str, the path to save data
# save_obsdata_name: str, the name to save data, canbe 'None'
obj_obs_data =  {'filter1': np.array(p['f606w']),
                 'filter2': np.array(p['f814w'])}
obj_artifical_data = {'flux_filter1_input': np.array(p_err['F(f606w)']),
                      'flux_filter1_output': np.array(p_err['f606w']),
                      'flux_filter2_input': np.array(p_err['F(f814w)']),
                      'flux_filter2_output': np.array(p_err['f814w'])}
        
if_flag = True
parameters_data_threshold = {'thr_s_to_n': 5,
                             'thr_sharpness': 0.2,
                             'thr_crowding': 0.5,
                             'thr_objects':  1,
                             'thr_flag': 2}
obs_parameters_data = {'s_to_n': None,
                       'sharpness': None,
                       'crowding': np.array((p['CROWD1'],p['CROWD2'])),
                       'objects': None,
                       'flag': np.array((p['#FLAG'],p['#FLAG']))}
artifical_parameters_data = {'s_to_n': None,
                             'sharpness': None,
                             'crowding': None,
                             'objects': None,
                             'flag': None}

######      PARSEC_restore taske             ######
###### prepare PARSEC isochrone (selectable) ######
# PARSEC_data_path: str, the path that save the PARSEC data


######      template_builder taske          ######
######      template build                 #######
# isochrone_path: str, the isochrone path
# flux_error_path: str, the flux error path
# obsdata_name: str, the name to save data, canbe 'None'
# save_template_path: str, the template path
necessary_parameters = {'Mm': 24.85, 
                        'A_v': 0.14,
                        'filter1': 606, 
                        'filter2': 814, 
                        'xrange': [-0.5889999999999986,1.3240000000000016], 
                        'yrange': [20.292,25.248]}
IMFpath = './IMF/IMF_Kroupa.npy'
binary_fra = 0.35
metal_input = np.hstack((-2.1917,np.arange(-2.1,0.11,0.1)))
logage_input = np.hstack((np.arange(6.6,8.7,0.1), np.arange(8.7,10.16,0.05)))
######



######