import PANCAKE
import ini_para

if ini_para.run_observation_reader == True:
    observation = PANCAKE.observation_reader()
    observation.data_and_fluxerr_save(save_obsdata_path = ini_para.observation_data_path, 
                                    save_obsdata_name = ini_para.observation_target_name, 
                                    obj_obs_data = ini_para.obj_obs_data, 
                                    obj_artifical_data = ini_para.obj_artifical_data, 
                                    if_flag = ini_para.if_flag, 
                                    parameters_data_threshold = ini_para.parameters_data_threshold, 
                                    obs_parameters_data = ini_para.obs_parameters_data, 
                                    artifical_parameters_data = ini_para.artifical_parameters_data)

if ini_para.run_PARSEC_restore == True:
    PARSEC = PANCAKE.PARSEC_restore()
    PARSEC.read_restore(PARSEC_data_path = ini_para.isochrone_path)

if ini_para.run_template_builder == True:
    template = PANCAKE.template_builder()
    template.template_builder(isochrone_path = ini_para.isochrone_path, 
                            flux_error_path = ini_para.observation_data_path, 
                            obsdata_name = ini_para.observation_target_name, 
                            save_template_path = ini_para.template_path, 
                            necessary_parameters = ini_para.necessary_parameters, 
                            IMFpath = ini_para.IMFpath, 
                            binary_fra = ini_para.binary_fra,
                            metal_input = ini_para.metal_input,
                            logage_input = ini_para.logage_input)