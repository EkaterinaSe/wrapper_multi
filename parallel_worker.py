import multiprocessing
import sys
import subprocess as sp
import os
import shutil
import numpy as np
from atom_package import model_atom, write_atom
from atmos_package import model_atmosphere, write_atmos_m1d, write_dscale_m1d
from m1d_output import m1d, m1dline


def mkdir(s):
    if os.path.isdir(s):
        shutil.rmtree(s)
    os.mkdir(s)
    return

def setup_multi_job(setup, job):
    """
    Setting up and running an individual serial job of NLTE calculations
    for a set of model atmospheres (stored in setup.jobs[k]['atmos'])
    Several indidvidual jobs can be run in parallel, set ncpu in the config. file
    to the desired number of processes
    Note: Multi 1D expects all input files to be named only in upper case

    input:
    # TODO:
    (string) directory: common working directory, default: "./"
    (integer) k: an ID of an individual job within the run
    (object) setup: object of class setup, regulates a setup for the whole run
    """

    """ Make a temporary directory """
    mkdir(job.tmp_wd)

    """ Link input files to a temporary directory """
    for file in ['absmet', 'abslin', 'abund', 'absdat']:
        os.symlink( setup.m1d_input + '/' + file, job.tmp_wd + file.upper() )

    """ Link INPUT file (M1D input file complimenting the model atom) """
    os.symlink( setup.m1d_input_file, job.tmp_wd +  '/INPUT' )

    """ Link executable """
    os.symlink(setup.m1d_exe, job.tmp_wd + 'multi1d.exe')


    """
    find a smarter way to do all of this...
    It doesn't need to be here, but..
    What kind of output from M1D should be saved?
    Read from the config file, passed here throught the object setup
    """
    job.output = { 'write_ew':setup.write_ew, 'write_profiles':setup.write_profiles, 'write_ts':setup.write_ts }

    """ Save EWs """
    if job.output['write_ew'] == 1 or job.output['write_ew'] == 2:
        # create file to dump output
        job.output.update({'file_ew' : job.tmp_wd + '/output_EW.dat' } )
        with open(job.output['file_ew'], 'w') as f:
            f.write("# Lambda, temp, logg.... \n")
    elif job.output['write_ew'] == 0:
        pass
    else:
        print("write_ew flag unrecognised, stoppped")
        exit(1)

    """ Output for TS? """
    if job.output['write_ts'] == 1:
        header = "departure coefficients from serial job # %.0f" %(job.id)
        header = str.encode('%1000s' %(header) )

        # create a file to dump output from this serial job
        job.output.update({'file_4ts' : job.tmp_wd + '/output_4TS.bin', 'pointer':1 } )
        with open(job.output['file_4ts'], 'wb') as f:
            f.write(header)
    elif job.output['write_ts'] == 0:
        pass
    else:
        print("write_ts flag unrecognised, stoppped")
        exit(1)



    return


def run_multi( job, atom, atmos):
    """
    Run MULTI1D
    input:
    (string) wd: path to a temporary working directory,
        created in setup_multi_job
    (object) atom:  object of class model_atom
    (object) atmos: object of class model_atmosphere
    """

    """ Create ATOM input file for M1D """
    write_atom(atom, job.tmp_wd +  '/ATOM' )

    """ Create ATMOS input file for M1D """
    write_atmos_m1d(atmos, job.tmp_wd +  '/ATMOS' )
    write_dscale_m1d(atmos, job.tmp_wd +  '/DSCALE' )

    """ Go to directory and run MULTI 1D """
    os.chdir(job.tmp_wd)
    sp.call(['multi1d.exe'])

    """ Read MULTI1D output and print to the common file """
    if job.output['write_ew'] > 0:
        out = m1d('./IDL1')
        if job.output['write_ew'] == 1:
            mask = np.arange(out.nline)
        elif job.output['write_ew'] == 2:
            mask = np.where(out.nq[:out.nline] > min(out.nq[:out.nline]))[0]

        with open(job.output['file_ew'], 'a')as f:

            # print(out.nline[mask])
            for kr in mask:
                line = out.line[kr]
                print(line)
                f.write('%10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f %10.4f\n' \
                    %(atmos.teff, atmos.logg, atmos.feh, out.abnd, out.g[kr], out.ev[kr],\
                        line.lam0, out.f[kr], out.weq[kr], out.weqlte[kr], np.mean(atmos.vturb)) )
    """ Read MULTI1D output and save in a common binary file in the format for TS """
    if job.output['write_ts'] == 1:
        out = m1d('./IDL1')
        with open(job.output['file_4ts'], 'ab') as fbin:
            atmosID = str.encode('%500s' %atmos.id)
            job.output['pointer'] = job.output['pointer'] + 500
            fbin.write(atmosID)

            # ndep = np.array([int(out.ndep)])
            # print(out.ndep, out.nk)
            job.output['pointer'] = job.output['pointer'] + 8
            # ndep.tofile(fbin, format='i4')
            print(out.ndep.to_bytes())
            fbin.write(out.ndep.to_bytes())

            nk = np.array([int(out.nk)])
            job.output['pointer'] = job.output['pointer'] + 8
            # nk.tofile(fbin, format='i4')
            fbin.write(nk.tobytes())

            tau500 = np.array(out.tau, dtype='f8')
            fbin.write(tau500.tobytes())
            # tau500.tofile(fbin, format='f64')
            job.output['pointer'] = job.output['pointer'] + ndep[0] * 8
            #
            depart = np.array((out.n/out.nstar).reshape(out.ndep, out.nk), dtype='f8')
            depart.tofile(fbin, format='f8')
            job.output['pointer'] = job.output['pointer'] + ndep[0] * nk[0] * 8

    os.chdir(job.common_wd)
    return

def read_m1d_output():
    return



def run_serial_job(setup, job):
        setup_multi_job( setup, job )
        print("job # %5.0f: %5.0f M1D runs" %( job.id, len(job.atmos) ) )
        for i in range(len(job.atmos)):
            # model atom is only read once
            atom = setup.atom
            atom.abund  =  job.abund[i]
            atmos = model_atmosphere(file = job.atmos[i], format = setup.atmos_format)
            run_multi( job, atom, atmos)
            # read output
        # shutil.rmtree(job['tmp_wd'])



    # """ Start individual jobs """
    # workers = []
    # for k in set.jobs.keys():
    #     p = multiprocessing.Process( target=run_multi( k, set ) )
    #     workers.append(p)
    #     p.start()
