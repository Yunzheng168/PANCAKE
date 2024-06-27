import numpy as np
import glob, os

class PARSEC_restore:
    def __init__(self, PARSEC_data_path):
        '''
        read and resave PARSEC isochrones and output the age and metallicity lists. 
        INPUT:
            PARSEC_data_path: str, the path that save the PARSEC data
        '''
        self.restore(PARSEC_data_path)

    def restore(self, PARSEC_data_path):
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
