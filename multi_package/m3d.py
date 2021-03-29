import numpy as np
from scipy.io import FortranFile
from IPython.display import clear_output, display
from multi_package.atmos import atmos
from multi_package.lines import bbline, bfline


def f77_string(f, dtype='a20'):
    data = f.read_record(dtype)
    
    if len(data) > 1:
        return np.array(list(map( lambda s: str(s).split("'")[1].rstrip(), data)))
    else:
        return str(data).split("'")[1].rstrip()


class m3d(object):
    def __init__(self, directory=None, file=None, lines=None, conts=None, output=True, ratmos=True, rpop=True):
        """
        Reads in a Multi3d parameteres from out_par

        input:
        (string) file: filename for the population (default:multi3d.input)
        (string) directory: where the file is located (default:../run/output)
        (boolean) output: if the user want output info (default:True)
        """

        
        #default
        if directory == None:
            directory = "../run/output/"
            
        self.dir = directory
        
        self.input = {}
        
        # --------------------------------------------------
        # Reading multi3d.input file
        # --------------------------------------------------
        print(directory + 'multi3d.input')
        
        for line in open(directory + 'multi3d.input', 'r'):
            li=line.strip()
            if not li.startswith(";") and li != '':
                    key, val = li.split('=')
                    
                    key, val = key.strip(), val.strip()
                    
                    if val.startswith("'"):
                        self.input[key] = val[1:-1]
                    elif val.startswith("["):
                        self.input[key] = eval('np.array(' + val + ')')
                    elif '.' in val:
                        self.input[key] = float(val)
                    else:
                        self.input[key] = int(val)
            
        # --------------------------------------------------
        # Reading out_par file
        # --------------------------------------------------
        if file == None:
            file = "out_par"
            
        fname = directory + file
        
        f = FortranFile(fname, 'r')
        
        self.nmu = f.read_ints(np.int32)[0] # number of angles
        nx = f.read_ints(np.int32)[0]
        ny = f.read_ints(np.int32)[0]
        nz = f.read_ints(np.int32)[0]
        
        self.widthx = f.read_reals('float64')
        self.widthy = f.read_reals('float64')
        self.widthz = f.read_reals('float64')
        
        self.mux = f.read_reals('float64')
        self.muy = f.read_reals('float64')
        self.muz = f.read_reals('float64')
        self.wmu = f.read_reals('float64')
        
        self.nnu = f.read_ints(np.int32)[0]
        self.maxac = f.read_ints(np.int32)[0]
        self.maxal = f.read_ints(np.int32)[0]
        
        self.nu = f.read_reals('float64')
        self.wnu = f.read_reals('float64')
        
        self.ac = f.read_ints(np.int32)
        self.al = f.read_ints(np.int32)
        self.nac = f.read_ints(np.int32)
        self.nal = f.read_ints(np.int32)
        
        self.nrad = f.read_ints(np.int32)[0]   # number of radiative transitions treated in detail
        self.nrfix = f.read_ints(np.int32)[0]  # number of transitions with fixed rates
        self.ncont = f.read_ints(np.int32)[0]  # number of bound-free transitions
        self.nline = f.read_ints(np.int32)[0]  # number of radiative bound-bound transitions
        self.nlevel = f.read_ints(np.int32)[0]
        
        self.id = f77_string(f)    # 4 character identification of atom
        self.crout = f77_string(f) 
        
        self.label = f77_string(f) # 20 character identification of level
        
        self.ion = f.read_ints(np.int32) # ionization stage of level, 1=neutral
        
        self.ilin = f.read_ints(np.int32)
        self.icon = f.read_ints(np.int32)
        
        self.abnd = f.read_reals('float64')[0] # atomic abundance, log scale with hydrogen=12
        self.awgt = f.read_reals('float64')[0] # atomic weight. input in atomic units, converted to cgs
        
        self.ev = f.read_reals('float64') # energy above ground state. input in cm-1, converted to ev
        self.g  = f.read_reals('float64') # statistical weight of level
            
        line_keys = ['lin1', 'lin2', 'lin3', 'lin4']
        self.input_ids = np.array([self.input[key] for key in line_keys])
        self.printed_krs = self.input_ids[self.input_ids > 0] - 1
            
        self.line = [None] * self.nline
        self.cont = [None] * self.ncont
            
        for kr in range(self.ncont):
            if conts == None or kr in conts:
                self.cont[kr] = bfline(f, kr, self.ncont)
                    
            else:
                for i in range(13):
                    l = f._read_size()
                    f._fp.seek(l, 1) # moving read pointer forward relative to position
                    l = f._read_size()
                    
                    
            
        for kr in range(self.nline):
            if lines == None or kr in lines:
                self.line[kr] = bbline(f, kr, self)
            else:
                for i in range(23):
                    l = f._read_size()
                    f._fp.seek(l, 1) # moving read pointer forward relative to position
                    l = f._read_size()
                
        
        clear_output(wait=True)
        print('Completed reading parameters from ' + self.dir)
        
        f.close()
        
        # --------------------------------------------------
        # Reading out_atm file
        # --------------------------------------------------
        if ratmos:
            self.atmos = atmos(self, output=False)
            self.atmosid = self.atmos.name
        
        self.atomid  = self.input['atom'].split('atom.')[-1]
        
        if self.input['maxiter'] > 0:
            self.mode = 'NLTE'
        else:
            self.mode = 'LTE'
            
        self.dim = '3D'
        
        if rpop:
            file = "out_pop"
            fname = directory + file
            
            data = np.memmap(fname, dtype=np.float32).reshape((nx,ny,nz,-1), order="F")
            
            self.pop_LTE  = data[:, :, :, :self.nlevel]
            self.pop_NLTE = data[:, :, :, self.nlevel:-1]
            
            
    def depart(self, nx=None, ny=None, level=None):
        if level is None:
            print('Level has to be defined! For example: m3d.depart(level=0)')
            
        else:
            if nx is None and ny is None:
                NLTE = self.pop_NLTE[:, :, :, level]
                LTE  = self.pop_LTE[:, :, :, level]
            
            else:
                NLTE = self.pop_NLTE[nx, ny, :, level]
                LTE  = self.pop_LTE[nx, ny, :, level]
                
            return NLTE/LTE
    
