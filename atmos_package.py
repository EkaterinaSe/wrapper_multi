import numpy as np

"""
    Read and manipulate model atmospheres
"""

def read_atmos_marcs(self, file):
    """
    Read model atmosphere in standart MARCS format i.e. *.mod
    input:
    (string) file: path to model atmosphere
    """
    # Boltzmann constant
    k_B = 1.38064852E-16

    data = []
    for line in  open(file, 'r').readlines():
        data.append(line.strip())
    # MARCS model atmosphere are by default strictly formatted
    self.id = data[0]
    self.teff = float(data[1].split()[0])
    self.flux = float(data[2].split()[0])
    self.logg = np.log10( float(data[3].split()[0]) )
    self.vturb = float(data[4].split()[0])
    self.feh, self.alpha = np.array(data[6].split()[:2]).astype(float)
    self.X, self.Y, self.Z = np.array(data[10].split()[:3]).astype(float)

    # read structure
    for line in data:
        if 'Number of depth points' in line:
            self.ndep = int(line.split()[0])
    self.tau500, self.height, self.temp, self.ne = [], [], [], []
    for line in data[25:25+self.ndep]:
        spl = np.array( line.split() ).astype(float)
        self.tau500.append(spl[2])
        self.height.append(spl[3])
        t = spl[4]
        self.temp.append(t)
        pe = spl[5]
        ne =  pe / t / k_B
        self.ne.append(ne)

    self.vturb = np.full(self.ndep, self.vturb )
    self.vmac = np.zeros(self.ndep)
    # add comments
    self.header = "Converted from MARCS formatted model atmosphere %s" %( file.split('/')[-1].strip() )

    return


def read_atmos_m1d(self, file):
    """
    Read model atmosphere in MULTI 1D input format, i.e. atmos.*
    M1D input model atmosphere is strictly formatted
    input:
    (string) file: path to model atmosphere file
    """
    data = []
    # exclude comment lines, starting with *
    for line in open( file , 'r').readlines():
        if not line.startswith('*'):
            data.append(line.strip())
    # read header
    self.id = data[0]
    self.depth_scale_type = data[1]
    self.logg = float(data[2])
    self.ndep = int(data[3])
    # read structure
    self.depth_scale, self.temp, self.ne, self.vmac, self.vturb = [],[],[],[],[]
    for line in data[ 4 : ]:
        spl = np.array(line.split()).astype(float)
        self.depth_scale.append( spl[0] )
        self.temp.append( spl[1] )
        self.ne.append( spl[2] )
        self.vmac.append( spl[3] )
        self.vturb.append( spl[4] )
    # info that's not provided in the model atmosphere file:
    self.teff   = np.nan
    self.feh    = np.nan
    self.alpha  = np.nan
    self.X      = np.nan
    self.Y      = np.nan
    self.Z      = np.nan
    # add comments here
    self.header = 'Read from M1D formatted model atmosphere %s' %( file.split('/')[-1].strip() )

    return


def write_atmos_m1d(atmos, file):
    """
    Write model atmosphere in MULTI 1D input format, i.e. atmos.*
    input:
    (object of class model_atmosphere): atmos
    (string) file: path to output file
    """
    with open(file, 'w') as f:
        # write header with comments
        f.write("* %s \n" %(atmos.header) )
        # write formatted header
        f.write("%s \n" %(atmos.id) )
        f.write("* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H)\n %s \n" %(atmos.depth_scale_type) )
        f.write("* log(g) \n %.3f \n" %(atmos.logg) )
        f.write("* Number of depth points \n %.0f \n" %(atmos.ndep) )
        # write structure
        f.write("* depth scale, temperature, N_e, Vmac, Vturb \n")
        for i in range(len(atmos.depth_scale)):
            f.write("%15.5E %15.5f %15.5E %10.3f %10.3f\n" \
                %( atmos.depth_scale[i], atmos.temp[i], atmos.ne[i], atmos.vmac[i], atmos.vturb[i] ) )

def write_dscale_m1d(atmos, file):
    """
    Write MULTI1D DSCALE input file with depth scale to be used for NLTE computations
    """
    with open(file, 'w') as f:
        # write formatted header
        f.write("%s \n" %(atmos.id) )
        f.write("* Depth scale: log(tau500nm) (T), log(column mass) (M), height [km] (H)\n %s \n" %(atmos.depth_scale_type) )
        f.write("* Number of depth points, top point \n %.0f %10.4E \n" %(atmos.ndep, atmos.depth_scale[0]) )
        # write structure
        for i in range(len(atmos.depth_scale)):
            f.write("%15.5E \n" %( atmos.depth_scale[i] ) )



    return


class model_atmosphere(object):
    def __init__(self, file='atmos.sun', format='m1d'):
        """
        Model atmosphere for NLTE calculations
        input:
        (string) file: file with model atmosphere, default: atmos.sun
        (string) format: m1d, marcs, stagger, see function calls below
        """
        if format.lower() == 'marcs':
            read_atmos_marcs(self, file)
            print("Setting depth scale to tau500")
            self.depth_scale_type = 'TAU500'
            self.depth_scale = self.tau500
        elif format.lower() == 'm1d':
            read_atmos_m1d(self, file)
        else:
            print("Unrecognized format of model atmosphere: %s" %(format) )
            exit(1)



if __name__ == '__main__':
    atmos = model_atmosphere('./atmos.sun_marcs_t5777_4.44_0.00_vmic1_new', format='m1d')
    write_atmos_m1d(atmos, file='atmos.test_out')
    write_dscale_m1d(atmos, file='dscale.test_out')
    atmos = model_atmosphere('./sun.mod', format='Marcs')
    write_atmos_m1d(atmos, file='atmos.test_out')
    write_dscale_m1d(atmos, file='dscale.test_out')
    exit(0)
