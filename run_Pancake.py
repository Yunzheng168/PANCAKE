import argparse
import observation_reader, PARSEC_restore, template_builder, gridding, fitting

parser = argparse.ArgumentParser(usage= 'python run_Pancake.py [-h] [-c] ini_para',description = 'run the pancake, CMD fitting')

parser.add_argument('-c',help='the config file name')

args = parser.parse_args()

file_name=args.c

try:
    ini_para = __import__(file_name)
    # print(module)
except Exception as e:
    print(f"Import failed: {e}")


if ini_para.run_observation_reader == True:
    observation = observation_reader.observation_reader(save_obsdata_path = ini_para.observation_data_path, 
                                                        save_obsdata_name = ini_para.observation_target_name, 
                                                        obj_obs_data = ini_para.obj_obs_data, 
                                                        obj_artifical_data = ini_para.obj_artifical_data)
    observation.read(if_flag = ini_para.if_flag, 
                    parameters_data_threshold = ini_para.parameters_data_threshold, 
                    obs_parameters_data = ini_para.obs_parameters_data, 
                    artifical_parameters_data = ini_para.artifical_parameters_data)

if ini_para.run_PARSEC_restore == True:
    PARSEC = PARSEC_restore.PARSEC_restore(PARSEC_data_path = ini_para.isochrone_path)

if ini_para.run_template_builder == True:
    template = template_builder.template_builder(isochrone_path = ini_para.isochrone_path, 
                                                flux_error_path = ini_para.observation_data_path, 
                                                obsdata_name = ini_para.observation_target_name, 
                                                save_template_path = ini_para.template_path,)
    template.build(necessary_parameters = ini_para.necessary_parameters, 
                    IMFpath = ini_para.IMFpath, 
                    binary_fra = ini_para.binary_fra,
                    metal_input = ini_para.metal_input,
                    logage_input = ini_para.logage_input)
    
if ini_para.run_gridding == True:
    grider = gridding.gridding(observation_data_path = ini_para.observation_data_path, 
                               observation_target_name = ini_para.observation_target_name, 
                               template_path = ini_para.template_path, 
                               save_gridding_path = ini_para.gridding_path)
    if 'uniform' in ini_para.bin_type:
        grider.uniform_bin(bin_num_CMDx = ini_para.uniform_bin_cmdx, bin_num_CMDy = ini_para.uniform_bin_cmdy)
    if 'quadtree' in ini_para.bin_type:
        grider.quad_bin(thresh = ini_para.quadtree_bin_thresh)
    if 'voronoi' in ini_para.bin_type:
        grider.voronoi_bin(thresh = ini_para.voronoi_bin_thresh)


if ini_para.run_fitting == True:
    fiter = fitting.fitting(observation_target_name = ini_para.observation_target_name, 
                            isochrone_path = ini_para.isochrone_path, 
                            gridding_path = ini_para.gridding_path, 
                            save_fitting_path = ini_para.fitting_path)
    fiter.multi_fit(fitting_bin_type = ini_para.fitting_bin_type,  
                    metal_input = ini_para.metal_input,
                    logage_input = ini_para.logage_input)
