import numpy as np
from scipy.io import FortranFile
from IPython.display import clear_output, display
from multi_package.auxfuncs import vacuum2obs
from scipy import integrate

ee = 1.602189e-12  # electron volt [erg]
hh = 6.626176e-27  # Planck's constant [erg s]
cc = 2.99792458e10 # Speed of light [cm/s]
em = 9.109534e-28  # Electron mass [g] 
uu = 1.6605655e-24 # Atomic mass unit [g]
bk = 1.380662e-16  # Boltzman's cst. [erg/K]
pi = 3.14159265359 # Pi
ec = 4.80325e-10   # electron charge [statcoulomb]

cm_to_aa = 1e+8


def f77_string(f, dtype='a20'):
    data = f.read_record(dtype)
    
    if len(data) > 1:
        return np.array(list(map( lambda s: str(s).split("'")[1].rstrip(), data)))
    else:
        return str(data).split("'")[1].rstrip()


class bfline():
    def __init__(self, f, kr, ncont):
        clear_output(wait=True)
        print('Reading bound-free transition ' + str(kr+1) + ' of ' + str(ncont))
        
        self.bf_type = f77_string(f)
        
        self.j      = f.read_ints(np.int32)[0]
        self.i      = f.read_ints(np.int32)[0]
        self.ntrans = f.read_ints(np.int32)[0]
        self.nnu    = f.read_ints(np.int32)[0]
        self.ired   = f.read_ints(np.int32)[0]
        self.iblue  = f.read_ints(np.int32)[0]
        
        self.nu0    = f.read_reals('float64')[0]
        self.numax  = f.read_reals('float64')[0]
        self.alpha0 = f.read_reals('float64')[0]
        
        self.alpha  = f.read_reals('float64')
        self.nu     = f.read_reals('float64') # frequency index [0:nq[kr]-1] or [0:nq[kr]]
        self.wnu    = f.read_reals('float64')

        self.laa = cc / self.nu * cm_to_aa


                
class bbline():
    def __init__(self, f, kr, parent):
        clear_output(wait=True)
        print('Reading bound-bound transition ' + str(kr+1) + ' of ' + str(parent.nline))
                
        self.line_profiletype = list(filter(None, f77_string(f, dtype='a8')))[0]
        
        self.ga      = f.read_reals('float64')[0] # radiative damping parameter
        self.gw      = f.read_reals('float64')[0] # van der waals damping parameter
        self.gq      = f.read_reals('float64')[0]
        self.lambda0 = f.read_reals('float64')[0] * cm_to_aa
        self.nu0     = f.read_reals('float64')[0]
        self.Aij     = f.read_reals('float64')[0]
        self.Bji     = f.read_reals('float64')[0]
        self.Bij     = f.read_reals('float64')[0]
        self.f       = f.read_reals('float64')[0] # oscillator strength
        self.qmax    = f.read_reals('float64')[0]
        self.Grat    = f.read_reals('float64')[0]
        
        self.ntrans  = f.read_ints(np.int32)[0]
        self.j       = f.read_ints(np.int32)[0]
        self.i       = f.read_ints(np.int32)[0]
        self.nnu     = f.read_ints(np.int32)[0]
        self.ired    = f.read_ints(np.int32)[0]
        self.iblue   = f.read_ints(np.int32)[0]
        self.io      = f.read_ints(np.int32)[0]
        
        self.nu      = f.read_reals('float64') # frequency index [0:nq[kr]-1] or [0:nq[kr]]
        self.q       = f.read_reals('float64') # wavelength from line center in doppler space
        self.wnu     = f.read_reals('float64')
        self.wq      = f.read_reals('float64')
        
        self.dlaa = cc * cm_to_aa * (1/self.nu - 1/self.nu0)
        self.laa = cc / self.nu * cm_to_aa
        
        self.kr = self.ntrans - 1
        
        self.dir = parent.dir
        
        if kr in parent.printed_krs:
            self.flux, self.cntm, self.i3, self.c3, self.w3 = self.ldflux(parent.input)
            
            self.nflux = self.flux / self.cntm
            self.lam = vacuum2obs(self.laa)[::-1] 
            self.lam0 = vacuum2obs(self.lambda0)
            
            
        
    def get(self, keys):
        output = []
        
        for key in keys:
            output.append(self.__dict__[key])
            
        return output
    
    
    def readie(self, inp, ang=0, quiet=False):
        # Reading output/out_nu file
        f = FortranFile(self.dir + 'out_nu', 'r')
        
        outnnu = f.read_ints(np.int32)
        outff  = f.read_ints(np.int32)
        
        f.close()
                
        
        muxout = inp['muxout']
        muyout = inp['muyout']
        muzout = inp['muzout']
        
        if not isinstance(muxout, np.ndarray):
            infile = directory + 'ie_allnu'
            
        else:
            mx = '{:+.2f}'.format(muxout[ang])
            my = '{:+.2f}'.format(muyout[ang])
            mz = '{:+.2f}'.format(muzout[ang])
            
            infile = self.dir + 'ie_' + mx + '_' + my + '_' + mz + '_allnu'
            
            
        if not quiet: print('reading from ' + infile)
            
        ie = np.memmap(infile, dtype='<f4')
        
        nx = inp['nx']
        ny = inp['ny']
        
        ired = self.ired
        
        try:
            rcrd = np.where(outff == ired)[0][0]
        except:
            print('No intensities written for line kr = ' + str(self.kr) + '\n'
                 + 'Note that kr is id-1!\n'
                 + 'Try kr = ' + str(inp['lin1'] - 1))
            
            raise
        
        if not quiet: print('Reading intensities of line kr = ' + str(self.kr))
        
        start = nx*ny*rcrd
        stop  = start + nx*ny*self.nnu
        
        ie = ie[start:stop].reshape((nx, ny, -1), order='F')
        
        return ie
    
    
    
    def ldflux(self, inp):
        nmuout = inp['nmuout']
        ntheta = inp['ntheta']
        nphi   = inp['nphi']
    
        lam = vacuum2obs(self.laa)[::-1] 
        nlam = len(lam)
        
        i3   = np.empty([nlam, nmuout])
        c3   = np.empty([nlam, nmuout])
        
        for i in range(nmuout):
            ie = self.readie(inp, ang=i, quiet=True)
            i3[:, i] = np.mean(ie, axis=(0,1))[::-1]
            
            # continuum is now a straight line with a
            # gradient computed from the first and index
            # of the intensity
            
            c3[:, i] = np.linspace(i3[ 0, i], i3[-1, i], len(i3[:, i]))
            
        if inp['quad_scheme'] == 'lobatto':
            wts = inp['wts']
    
            #flux = np.broadcast_to(wts, i3.shape) * i3
            flux = np.sum(wts*i3, axis=1)
            
        elif inp['quad_scheme'] == 'custom':
            muz  = inp['muzout']
            
            mus = np.append(0, np.unique(muz))
            
            i1 = np.zeros((nlam, len(mus)))
            
            for i, mu in enumerate(mus):
                mask = muz == mu
                
                if any(mask):
                    i1[:, i] = np.mean(i3[:, mask], axis=1)
                    
                    
            flux = integrate.simps(i1 * 2*mus, x=mus)
                          
        else:
            print('Unsupported quad scheme: ' + inp['quad_scheme'])
            
    
        # setting up new continuum array for flux
        cntm = np.linspace(flux[0], flux[-1], len(flux))
        
        w3 = np.trapz(1-flux/cntm, x=lam) * 1e3
        
        
        return flux, cntm, i3, c3, w3
        
        
            



