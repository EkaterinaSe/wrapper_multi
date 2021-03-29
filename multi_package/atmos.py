
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic
from scipy.signal import savgol_filter
from numexpr import evaluate as nev

class atmos(object):
    """
    Atmoshere class
    """

    def __init__(self, parent, file=None, output=True):
        """
        Reads in a Multi3d atmosphere, takes the parameteres from multi3d.input

        input:
        (string) file: filename for the population (default:multi3d.input)
        (string) directory: where the file is located (default:../run/output)

        You can provide a different size for the population matrix
        if you want to look at a different grid-size
        (int) nx:
        (int) ny:
        (int) nz:

        (boolean) output: if the user want output info (default:True)
        (boolean) bifrost: default: False
        """

        if file == None:
            file = "out_atm"

        fname = parent.dir + file
                        
        nx = parent.input['nx']
        ny = parent.input['ny']
        nz = parent.input['nz']
        
        
        if parent.input['usestagger'] == 1:
            self.name = parent.input['stagger_snap']
            
        else:
            self.name = parent.input['atmosid']

        #TODO
        atom_nline = 6
        #Reads the binary file

        if output: print("reading " + fname)
            
        data = np.memmap(fname, dtype=np.float32).reshape((nx, ny, nz, 7 + atom_nline), order='F') #,count=nx*ny*nz*5)
        
        self.ne    = data[:, :, :, 0]
        self.temp  = data[:, :, :, 1]
        self.vx    = data[:, :, :, 2]
        self.vy    = data[:, :, :, 3]
        self.vz    = data[:, :, :, 4]
        self.rho   = data[:, :, :, 5]
        self.nh    = data[:, :, :, 6:6+atom_nline]
        self.vturb = data[:, :, :, 6+atom_nline]
        
        self.tau = np.memmap(parent.dir + 'tau500', dtype='<f4')[:nx*ny*nz].reshape([nx, ny, nz], order='F')
        
        
    def dav(self, all_attr, nfin=None, limits=[-6, 2]):
        print('Computing optical depth averages')
        if type(all_attr) == list:
            all_attr = np.array(all_attr)
        
        if nfin == None:
            nfin = self.tau.shape[-1]
        
        if len(all_attr.shape) > 3:
            all_attr = all_attr.ravel().reshape(len(all_attr), -1)
        else:
            all_attr = all_attr.ravel()
            
        #limits =10.0**np.array(limits)
        tau = self.tau.ravel()
        ltau = nev('log10(tau)')
        
        all_av_att, _, _ = binned_statistic(ltau, all_attr, statistic='mean', range=limits, bins=nfin)
        
        return all_av_att
    
        
          
    def to_1D(self, name=None, ofolder='.', nfin=None, limits=[-6, 2]):
        # Creates atmos. and dscale. files as input for Multi1D
        
        if name == None:
            name = self.name
            
        if nfin == None:
            nfin = self.tau.shape[-1]
            
        ux, uy, uz = self.vx, self.vy, self.vz
        u2 = ux**2 + uy**2 + uz**2
        
        all_attr = np.array([self.tau, self.temp, self.ne, u2, ux, uy, uz])
        
        [tau, tt, ne, u2, ux, uy, uz] = self.dav(all_attr, nfin=nfin, limits=limits)
        
        ltau = nev('log10(tau)')
        vz = np.zeros(len(ltau))
        vturb = 1/3 * (u2 - ux**2 - uy**2 - uz**2)**0.5
        
        [tt, ne] = savgol_filter([tt, ne], 31, 3) # window size 31, polynomial order 3
        
        df = pd.DataFrame({'*LG(TAU)':ltau, 'TEMPERATURE':tt, 'NE':ne, 'V':vz, 'VTURB':vturb}, index=None)
        
        print('Writing .csv file: ' + ofolder + '/atmos.' + name)
        f = open(ofolder + '/atmos.' + name, 'w+')
        
        f.write('  ' + name + '\n')
        f.write('  TAU\n')
        f.write('*\n')
        f.write('* LG G\n')
        f.write('  4.4\n')
        f.write('*\n')
        f.write('* NDEP\n')
        f.write('{:4}\n'.format(len(ltau)))
        #f.write('*')
        
        
        #df.to_csv(f, index=False, sep='\t', float_format='%12.4E')
        f.write(df.to_string(index=False, float_format='%12.4E'))
        
        
        f.close()

        print('Writing .csv file: ' + ofolder + '/dscale.' + name)
        f = open(ofolder + '/dscale.' + name, 'w+')
        
        f.write('  ' + name + '\n')
        f.write('  TAU\n')
        f.write('{:4}{:12.4E}\n'.format(len(ltau), df['*LG(TAU)'].values[0]))
        
        df['*LG(TAU)'].to_csv(f, index=False, sep='\t', float_format='%12.4E', header=None)
        
        f.close()
