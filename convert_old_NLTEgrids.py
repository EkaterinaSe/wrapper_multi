import sys
import os
import time
import numpy as np
from datetime import date
from parallel_worker import addRec_to_NLTEbin

"""
A script to convert exisiting NLTE grids to a new format
NOTE: not all grids have header written in them, check with h_old
"""


if __name__ == '__main__':
    today = date.today().strftime("%b-%d-%Y")

    """ Which files to reformat? """
    binFile = sys.argv[1]
    auxFile = sys.argv[2]
    print("Reading binary NLTE grid from %s" %(binFile) )
    print("Reading aux data from %s" %(auxFile) )

    ma3d = False
    """ are those mean stagger 3D models? """
    if len(sys.argv) > 3:
        if sys.argv[3]=='3d' or sys.argv[3]=='3D':
            ma3d = True


    """ Open new files (to be written in new format), write header """
    binFile_new = binFile.replace('.bin','') + '_reformat_%s.bin' %(today)
    with open(binFile_new, 'wb') as f_new :
        header = "NLTE grid (grid of departure coefficients) in TurboSpectrum format. \nAccompanied by an auxilarly file and model atom. \n" + \
                "Reformatted from the existing grid %s on %s: \nby Ekaterina Semenova (semenova at mpia dot de) \n" %(binFile.split('/')[-1], today)
        header = str.encode('%1000s' %(header) )
        f_new.write(header)

    auxFile_new = auxFile.replace('.txt','') + '_reformat_%s.txt' %(today)
    faux_new = open(auxFile_new, 'w')

    faux_new.write("# atmos ID, Teff [K], log(g) [cgs], [Fe/H], [alpha/Fe], mass, Vturb [km/s], A(X), pointer \n")


    """ Open the input NLTE binary in the old format, read header """
    f = open(binFile, 'rb')

    # h_old = f.readline(1000).decode('utf-8', 'ignore')

    """ Read the input aux file in the old format """
    auxData = {'atmosID':[], 'teff':[], 'logg':[], 'feh':[], 'A(X)':[] }
    auxData['atmosID'] = np.loadtxt(auxFile, comments='#', dtype=str, usecols=(0))
    auxData['teff'], auxData['logg'], auxData['feh'], auxData['A(X)'], auxData['pointer']  =  \
            np.loadtxt(auxFile, usecols=(1,2,3,4,5), unpack=True, comments='#')
    """ Get more parameters from the atmos ID """
    # unless  those are stagger avareged models
    if not ma3d:
        auxData.update({ 'alpha/fe':[], 'mass':[], 'vturb':[] })
        for atm in auxData['atmosID']:
            mass = float(atm.split('_m')[-1].split('_t')[0])
            alpha = float(atm.split('_a')[-1].split('_c')[0])
            vturb = float(atm.split('_t')[-1].split('_')[0])

            auxData['mass'].append(mass)
            auxData['alpha/fe'].append(alpha)
            auxData['vturb'].append(vturb)
    else:
        N = len(auxData['atmosID'])
        auxData.update({ 'alpha/fe':N*[np.nan], 'mass':N*[np.nan], 'vturb':N*[np.nan] })



    """ Iteratively read old binary and aux file """
    auxCount = 0
    # a pointer starts with 1, because Fortran starts with 1 while Python starts with 0
    pointer = len(header) + 1

    for i in range(10**12):
        atmosID = f.readline(500).decode('utf-8', 'ignore').strip()


        if len(atmosID) > 0:
            ndep = np.fromfile(f, dtype='i4', count=2)[0]
            nk = np.fromfile(f, dtype='i4', count=2)[0]
            tau = np.fromfile(f, dtype='f8', count=ndep)
            depart = np.fromfile(f, dtype='f8', count=ndep*nk)
            depart = depart.reshape(nk, ndep)

            if atmosID != auxData['atmosID'][auxCount]:
                print("!!! records in the aux file and NLTE grid do not match: %s, %s" %(atmosID, auxData['atmosID'][auxCount]) )
            else:
                # something went wrong and the first letter was lost in the atmosID
                if ma3d and not atmosID.startswith('t'):
                    atmosID = 't'+atmosID
                if not ma3d and not (atmosID.startswith('p') or atmosID.startswith('s') ) :
                    if auxData['mass'][auxCount] == 0.0:
                        atmosID = 'p'+atmosID
                    else:
                        atmosID = 's'+atmosID


                """ Add each record to a binary in new format, accompanied by a record in a new aux file """
                faux_new.write(" '%s' %10.4f %10.4f %10.4f %10.4f %10.2f %10.2f %10.4f %10.0f \n" \
                        %( atmosID, auxData['teff'][auxCount], auxData['logg'][auxCount], auxData['feh'][auxCount],  \
                        auxData['alpha/fe'][auxCount], auxData['mass'][auxCount], auxData['vturb'][auxCount], auxData['A(X)'][auxCount], pointer  ) )

                # call to the function below insures that reformatted grids are written identically to the ones computed in the new format
                record_len = addRec_to_NLTEbin(binFile_new, atmosID, ndep, nk, tau, depart)
                pointer = pointer + record_len
                auxCount += 1

        else:
            # reached end of grid
            f.close()
            faux_new.close()
            break



    print("total of %.0f records written into a new grid, %s, %s" %(auxCount, binFile_new, auxFile_new) )
