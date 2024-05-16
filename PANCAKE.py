import numpy as np
import glob, os
from extinction import odonnell94
import pandas as pd
from scipy import integrate, interpolate

class observation_reader:
    def __init__(self) -> None:
        pass
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

    def data_and_fluxerr_save(self, save_obsdata_path, save_obsdata_name, obj_obs_data, obj_artifical_data, if_flag = False, parameters_data_threshold = None, obs_parameters_data = None, artifical_parameters_data = None):
        '''
        read and resave observation data and artifical stars. 
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
            if_flag: bool, if true mask the obsdata through parameter and threshold.
            parameters_data_threshold: a dict including 5 floats
                    thr_s_to_n: float, signal-to-noise ratio, typical 5-10
                    thr_sharpness: float, typical 0.2
                    thr_crowding: float, typical 0.5
                    thr_objects: float, typical 1 (good stars)
                    thr_flag: float, photometry quality flag, typical 0
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
        filter1, filter2 = obj_obs_data['filter1'], obj_obs_data['filter2']
        flux_filter1_input, flux_filter1_output, flux_filter2_input, flux_filter2_output = obj_artifical_data['flux_filter1_input'], obj_artifical_data['flux_filter1_output'], obj_artifical_data['flux_filter2_input'], obj_artifical_data['flux_filter2_output']
        flux_filter1_output[flux_filter1_output>90] = np.nan
        flux_filter2_output[flux_filter2_output>90] = np.nan

        if if_flag == True:
            flt_T_obs_flag = self.__parameters_constrain(len(filter1), parameters_data_threshold, obs_parameters_data)
            flt_T_obs_flag = flt_T_obs_flag & (filter1<90) & (filter2<90)
            filter1, filter2 = filter1[flt_T_obs_flag], filter2[flt_T_obs_flag]

            flt_T_artifical_flag = self.__parameters_constrain(len(flux_filter1_output), parameters_data_threshold, artifical_parameters_data)
            flux_filter1_output[~flt_T_artifical_flag] = np.nan
            flux_filter2_output[~flt_T_artifical_flag] = np.nan
        else:
            flt_T_obs = (filter1<90) & (filter2<90)
            filter1, filter2 = filter1[flt_T_obs], filter2[flt_T_obs]

        np.save(save_obsdata_path+'/'+save_obsdata_name+'_obs_cmdx.npy',filter1-filter2)
        np.save(save_obsdata_path+'/'+save_obsdata_name+'_obs_cmdy.npy',filter2)

        np.save(save_obsdata_path+'/'+save_obsdata_name+'_filter1_input.npy',flux_filter1_input)
        np.save(save_obsdata_path+'/'+save_obsdata_name+'_filter1_output.npy',flux_filter1_output)
        np.save(save_obsdata_path+'/'+save_obsdata_name+'_filter2_input.npy',flux_filter2_input)
        np.save(save_obsdata_path+'/'+save_obsdata_name+'_filter2_output.npy',flux_filter2_output)
        print('#### observation restore finish ####')

class PARSEC_restore:
    def __init__(self) -> None:
        pass
    def read_restore(self, PARSEC_data_path):
        '''
        read and resave PARSEC isochrones and output the age and metallicity lists. 
        INPUT:
            PARSEC_data_path: str, the path that save the PARSEC data
        '''
        print('#### PARSEC restore start ####')
        datapath = PARSEC_data_path
        outpath = PARSEC_data_path+'/newstore/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        files = glob.glob(datapath+'*.dat')
        metal = []
        logage = []
        for file in files:
            with open(file,'r') as f:
                content = f.read()

            content_sp1 = content.split('# Kind of output: isochrone tables')[1]
            content_sp2 = content_sp1.split('#')
    
            for i in range(len(content_sp2)):
                if 'Zini' in content_sp2[i]:
                    metal0 = content_sp2[i].split('\n')[3].split(' ')[1]
                    logage0 = content_sp2[i].split('\n')[3].split(' ')[2]
                    metal.append(metal0)
                    logage.append(logage0)
                    with open(outpath+'/M'+metal0.split('.')[0]+'.'+metal0.split('.')[1][:4]+'A'+logage0.split('.')[0]+'.'+logage0.split('.')[1][:4], 'w') as f:
                        f.write(content_sp2[i])

        metal = np.sort(np.array(list(set(metal))).astype('float'))
        logage = np.sort(np.array(list(set(logage))).astype('float'))  #, logage
        np.save(outpath+'/metal_list.npy',metal)
        np.save(outpath+'/logage_list.npy',logage)
        print('#### PARSEC restore finish ####')

class template_builder:
    def __init__(self) -> None:
        pass

    def __calculate_extinction(self, A_v, filters):
        R_v = 3.1
        wave = 10. * np.array([float(filters[0]), float(filters[1])])
        extin = odonnell94(wave, A_v, R_v)
        return extin
        
    def __number_to_mass(self, Mini, Mass, Mini_min, Mini_max, IMFname, IMF_min = 0.08, IMF_max = 150):
        p1 = interpolate.interp1d(Mini, Mass)
        if IMFname == 'Kroupa':
            def IMF_fun(m):
                # Define the Kroupa initial mass function
                if m <= 0.08:
                    return 53.34 * (m / 0.08) ** -0.3
                if 0.08 < m <= 0.5:
                    return 53.34 * (m / 0.08) ** -1.3
                if 0.5 < m <= 150.0:
                    return 53.34 * 6.25 ** -1.3 * (m / 0.5) ** -2.3

            def massxIMF_fun(m):
                # Calculate mass times Kroupa IMF
                if m <= 0.08:
                    return m * 53.34 * (m / 0.08) ** -0.3
                if 0.08 < m <= 0.5:
                    return m * 53.34 * (m / 0.08) ** -1.3
                if 0.5 < m <= 150.0:
                    return m * 53.34 * 6.25 ** -1.3 * (m / 0.5) ** -2.3
                
            def AmassxIMF_fun(m):
                # Calculate mass times Kroupa IMF
                if m <= 0.08:
                    return p1(m) * 53.34 * (m / 0.08) ** -0.3
                if 0.08 < m <= 0.5:
                    return p1(m) * 53.34 * (m / 0.08) ** -1.3
                if 0.5 < m <= 150.0:
                    return p1(m) * 53.34 * 6.25 ** -1.3 * (m / 0.5) ** -2.3
        
        elif IMFname == 'Individual':
            def IMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return 15.8*np.e**(-(logm-np.log10(0.079))**2./(2.*0.69*0.69))/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return m**(-2.3)

            def massxIMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return m * 15.8*np.e**(-(logm-np.log10(0.079))**2./(2.*0.69*0.69))/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return m * m**(-2.3)
                
            def AmassxIMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return p1(m) * 15.8*np.e**(-(logm-np.log10(0.079))**2./(2.*0.69*0.69))/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return p1(m) * m**(-2.3)
                
        elif IMFname == 'System':
            def IMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return 8.6*np.e**(-(logm-np.log10(0.22))**2/2/0.57/0.57)/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return m**(-2.3)

            def massxIMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return m * 8.6*np.e**(-(logm-np.log10(0.22))**2/2/0.57/0.57)/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return m * m**(-2.3)
                
            def AmassxIMF_fun(m):
                logm = np.log10(m)
                if logm <= 0:
                    return p1(m) * 8.6*np.e**(-(logm-np.log10(0.22))**2/2/0.57/0.57)/1.9/m/np.log(10)
                if 0.<logm <= 2.:
                    return p1(m) * m**(-2.3)

        mxn_IMF, err = integrate.quad(massxIMF_fun, IMF_min, IMF_max)
        mxn_ACT, err = integrate.quad(AmassxIMF_fun,Mini_min, Mini_max, limit = 100)
        N_IMF_ACT, err = integrate.quad(IMF_fun,Mini_min, Mini_max)
        
        return mxn_IMF/N_IMF_ACT, mxn_ACT/N_IMF_ACT
    
    def __get_IMF(self, IMFpath):
        # Read the initial mass function (IMF) library
        IMF_base = np.load(IMFpath)
        IMF_base = IMF_base[1:]  # Remove the first element
        return IMF_base
    
    def __read_isochrone_save(self, datapath, outpath, metal, age, filters, extin, Mm, xrange, yrange, IMFpath):
        Mini_min_list, Mini_max_list, inpl_filter1_list, inpl_filter2_list = [],[],[],[]
        Mini_b_min_list, Mini_b_max_list = [],[]
        isochrone_name_list = []
        IMF_mass, ACT_mass = np.ones((len(metal),len(age))), np.ones((len(metal),len(age)))
        for n in range(len(metal)):
            for i in range(len(age)):
                #read isochrone
                p = pd.read_csv(datapath+'/M'+'%.04f'%metal[n]+'A'+'%.04f'%age[i], sep='\s+')
                
                Mini = np.array(p['Mini'])
                Mass = np.array(p['Mass'])
                Filter1=np.array(p['F'+str(filters[0])+'Wmag'])
                Filter2=np.array(p['F'+str(filters[1])+'Wmag'])
                label = np.array(p['label'])
                
                Filter1 = Filter1 + extin[0]+Mm
                Filter2 = Filter2 + extin[1]+Mm
                color = Filter1 - Filter2
                FLT_select_bin_range = (color>xrange[0]) & (color<xrange[1]) & (Filter2>yrange[0]) & (Filter2<yrange[1])
                Mini_min, Mini_max = np.nanmin(Mini[FLT_select_bin_range]), np.nanmax(Mini[FLT_select_bin_range])

                #doing interpolate
                inpl_lin_filter1 = interpolate.interp1d(Mini,Filter1)
                inpl_lin_filter2 = interpolate.interp1d(Mini,Filter2)
                #save mass range main star
                Mini_min_list.append((Mini[FLT_select_bin_range]).min())
                Mini_max_list.append((Mini[FLT_select_bin_range]).max())
                #save mass range binary star
                FLT_select_binary_range = label == 1.
                Mini_b_min_list.append((Mini[FLT_select_binary_range]).min())
                Mini_b_max_list.append((Mini[FLT_select_binary_range]).max())
                
                #save interpolate information
                inpl_filter1_list.append(inpl_lin_filter1)
                inpl_filter2_list.append(inpl_lin_filter2)
                #save isochrone name = 'metal+age'
                isochrone_name_list.append('M'+'%.04f'%metal[n]+'A'+'%.04f'%age[i])

                #numtostellar
                IMFname = IMFpath.split('_')[-1].split('.')[0]
                IMF_mass[n][i], ACT_mass[n][i] = self.__number_to_mass(Mini, Mass, Mini_min, Mini_max, IMFname, IMF_min = 0.08, IMF_max = 150)

        np.save(outpath+'/number_to_IMF_mass.npy',IMF_mass)
        np.save(outpath+'/number_to_ACT_mass.npy',ACT_mass)

        return isochrone_name_list,Mini_min_list,Mini_max_list,inpl_filter1_list,inpl_filter2_list,Mini_b_min_list,Mini_b_max_list

    def __bin_flux_error(self, flux_error_path, obsdata_name):
        flux_filter1_input = np.load(flux_error_path+'/'+obsdata_name+'_filter1_input.npy')
        flux_filter1_output = np.load(flux_error_path+'/'+obsdata_name+'_filter1_output.npy')
        flux_filter2_input = np.load(flux_error_path+'/'+obsdata_name+'_filter2_input.npy')
        flux_filter2_output = np.load(flux_error_path+'/'+obsdata_name+'_filter2_output.npy')

        Fluxerr_filter1_delta = flux_filter1_output-flux_filter1_input
        Fluxerr_filter2_delta = flux_filter2_output-flux_filter2_input
        bin_number = 200
        bin_value, bin_edge = np.histogram(flux_filter1_input, bins = bin_number)
        is_zero_num = len(bin_value[bin_value==0])
        while is_zero_num > 0:
            bin_number = bin_number - is_zero_num
            bin_value, bin_edge = np.histogram(flux_filter1_input, bins = bin_number)
            is_zero_num = len(bin_value[bin_value==0])

        binlist_filter1_1 = bin_edge[:-1]
        binlist_filter1_2 = bin_edge[1:]
        binlist_filter1 = np.array([(binlist_filter1_1[i],binlist_filter1_2[i]) for i in range(len(binlist_filter1_2))])

        bin_number = 200
        bin_value, bin_edge = np.histogram(flux_filter2_input, bins = bin_number)
        is_zero_num = len(bin_value[bin_value==0])
        while is_zero_num > 0:
            bin_number = bin_number - is_zero_num
            bin_value, bin_edge = np.histogram(flux_filter2_input, bins = bin_number)
            is_zero_num = len(bin_value[bin_value==0])

        binlist_filter2_1 = bin_edge[:-1]
        binlist_filter2_2 = bin_edge[1:]
        binlist_filter2 = np.array([(binlist_filter2_1[i],binlist_filter2_2[i]) for i in range(len(binlist_filter2_2))])

        return binlist_filter1, flux_filter1_input, Fluxerr_filter1_delta, binlist_filter2, flux_filter2_input, Fluxerr_filter2_delta

    def __model_CMD(self, flux_error_path, obsdata_name, perct_Binary, Mini_min, Mini_max, Mini_min_b, Mini_max_b, inpl_filter1, inpl_filter2, IMFpath, number = 10000):
        numbZ = int(number)
        binnumZ = int(number*perct_Binary)
        IMF_base = self.__get_IMF(IMFpath)
        if numbZ>1:
            #IMF:
            ## bin the (nomal and flat)IMF database into isochrone mass range
            lar_than_min = IMF_base[IMF_base >= Mini_min]
            sm_t_max = lar_than_min[lar_than_min <= Mini_max]
            
            ## randomly choose numbZ number of masses
            Mod_imf = np.random.choice(sm_t_max, numbZ)

            #binary:
            ## randomly choose binaries mass
            lar_than_min = IMF_base[IMF_base >= Mini_min_b]
            sm_t_max = lar_than_min[lar_than_min <= Mini_max_b]
            bin_imf = np.random.choice(sm_t_max, binnumZ)
            flt_binary = np.ones(numbZ)
            flt_binary[:binnumZ] = 0.
            np.random.shuffle(flt_binary)

            ###close binaries  get magnitudes through mass (apply interpolate)
            filter1_mag = inpl_filter1(Mod_imf)
            filter2_mag = inpl_filter2(Mod_imf)
            filter1_b_mag = -2.5*np.log10(10**(-0.4*filter1_mag[flt_binary==0.])+10**(-0.4*inpl_filter1(bin_imf)))
            filter2_b_mag = -2.5*np.log10(10**(-0.4*filter2_mag[flt_binary==0.])+10**(-0.4*inpl_filter2(bin_imf)))

            filter1_tot_mag = np.hstack((filter1_b_mag,filter1_mag[flt_binary==1.]))
            filter2_tot_mag = np.hstack((filter2_b_mag,filter2_mag[flt_binary==1.]))

            ### add fluxerr
            filter1_finial_mag = np.ones(len(filter1_tot_mag))
            filter2_finial_mag = np.ones(len(filter2_tot_mag))

            binlist_filter1, flux_filter1_input, Fluxerr_filter1_delta, binlist_filter2, flux_filter2_input, Fluxerr_filter2_delta = self.__bin_flux_error(flux_error_path, obsdata_name)
            
            for (bin1,bin2) in binlist_filter1:
                    fltrin=(bin1<=filter1_tot_mag)&(filter1_tot_mag<bin2)
                    fltrout=(bin1<=flux_filter1_input)&(flux_filter1_input<bin2)
                    filter1_finial_mag[fltrin]=np.random.choice(Fluxerr_filter1_delta[fltrout],len(filter1_tot_mag[fltrin]))+filter1_tot_mag[fltrin]

            for (bin1,bin2) in binlist_filter2:
                    fltrin=(bin1<=filter2_tot_mag)&(filter2_tot_mag<bin2)
                    fltrout=(bin1<=flux_filter2_input)&(flux_filter2_input<bin2)
                    filter2_finial_mag[fltrin]=np.random.choice(Fluxerr_filter2_delta[fltrout],len(filter2_tot_mag[fltrin]))+filter2_tot_mag[fltrin]

            #calculate complateness
            color = filter1_finial_mag-filter2_finial_mag
            nanvalue = np.isnan(color)
            complete = 1.-len(color[nanvalue])/len(color)

            return filter1_finial_mag, filter2_finial_mag, complete
            
        
    def template_builder(self, isochrone_path, flux_error_path, obsdata_name, save_template_path, necessary_parameters, IMFpath = './IMF/IMF_Kroupa.npy', binary_fra = 0.35, metal_input = None, logage_input = None):
        '''
        isochrone template build.
        INPUT:
            isochrone_path: str, the isochrone path
            flux_error_path: str, the flux error path
            obsdata_name: str, the name to save data, canbe 'None'
            save_template_path: str, the template path
            necessary_parameters: a dict include 6 parameters
                    Mm: float, distance modulu
                    A_v: float, extinction
                    filter1: float, filter1 wavelength in 0.1 Augstrom
                    filter2: float, filter2 wavelength in 0.1 Augstrom
                    xrange: [float,float], color range
                    yrange: [float,float], magnitude range
            IMFpath: initial mass function, typical './IMF/IMF_Kroupa.npy'
            binary_fra: binary fraction, typical 35%
            metal_input: 1d array, isochrone metal list 
            logage_input: 1d array, isochrone log age list 
        OUTPUT:
            template filter1 and filter2 in magnitude
        '''
        print('#### template build start ####')
        Mm, A_v, filter1, filter2, xrange, yrange = necessary_parameters['Mm'], necessary_parameters['A_v'], necessary_parameters['filter1'], necessary_parameters['filter2'], necessary_parameters['xrange'], necessary_parameters['yrange']
        extin = self.__calculate_extinction(A_v, [filter1,filter2])
        isochrone_path_restore = isochrone_path+'/newstore/'
        if metal_input is None:
            metal = np.load(isochrone_path_restore+'/metal_list.npy')
        else:
            metal = metal_input
        if logage_input is None:
            logage = np.load(isochrone_path_restore+'/logage_list.npy')
        else:
            logage = logage_input

        filter1_template_list = []
        filter2_template_list = []
        completeness_list = []
        # IMF, binary_fra

        isochrone_name_list,Mini_min_list,Mini_max_list,inpl_filter1_list,inpl_filter2_list,Mini_b_min_list,Mini_b_max_list = self.__read_isochrone_save(isochrone_path_restore, save_template_path, metal, logage, [filter1,filter2], extin, Mm, xrange, yrange, IMFpath)
        for NUM in range(len(logage)*len(metal)):
            iso_name,Mini_min,Mini_max,inpl_filter1,inpl_filter2 = isochrone_name_list[NUM],Mini_min_list[NUM],Mini_max_list[NUM],inpl_filter1_list[NUM],inpl_filter2_list[NUM]
            Mini_min_b,Mini_max_b = Mini_b_min_list[NUM],Mini_b_max_list[NUM]
            filter1_finial_mag, filter2_finial_mag, complete = self.__model_CMD(flux_error_path, obsdata_name, binary_fra, Mini_min, Mini_max, Mini_min_b, Mini_max_b, inpl_filter1, inpl_filter2, IMFpath, number = 10000)
            filter1_template_list.append(filter1_finial_mag)
            filter2_template_list.append(filter2_finial_mag)
            completeness_list.append(complete)
            print(iso_name+' finish')

        np.save(save_template_path+'/template_'+obsdata_name+'_filter1.npy',np.array(filter1_template_list))
        np.save(save_template_path+'/template_'+obsdata_name+'_filter2.npy',np.array(filter2_template_list))
        print('#### template build finish ####')

