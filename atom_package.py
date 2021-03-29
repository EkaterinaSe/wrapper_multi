import sys
import os
import numpy as np


"""
Only for model atoms in MULTI format
"""

def read_atom(self, file):
    data = []
    for line in open(file, 'r'):
        line=line.strip()
        data.append(line)

    """ Read the header """
    c_noncom = 0 # a counter for non-commented lines
    c_all = 0 # a count for all lines, including commented lines
    self.header = []
    for li in data:
        c_all += 1
        if not li.startswith('*') and li != '':
            c_noncom += 1
            if c_noncom == 1:
                self.element = li
            elif c_noncom == 2:
                self.abund, self.atomic_weight = np.array(li.split()).astype(float)
            elif c_noncom == 3:
                self.nk, self.nline, self.ncont, self.nrfix = np.array(li.split()).astype(int)
                break
        else:
            self.header.append(li)

    """ Read the energy levels """
    self.en, self.g, self.label, self.ion = [], [], [], []
    for li in data[c_all+1 : ]:
        if c_noncom < 3 + self.nk:
            c_all += 1
            if not li.startswith('*') and li != '':
                lsp = li.split()
                self.en.append( float( lsp[0] ) )
                self.g.append( int( float(lsp[1]) ) )
                self.label.append( li.split("'")[1].strip() )
                self.ion.append( int( li.split("'")[-1].split()[0] ) )
                c_noncom += 1
        else:
            c_all += 1
            break
    """ Read the b-b transitions """
    self.bb = {}
    kr = 0
    for i in range(len(data[ c_all : ])):
        if c_noncom < 3 + self.nk + self.nline:
            li = data[ c_all  ]
            # preceeding line
            if not li.startswith('*') and li != '':
                # if no comment line before this b-b, pass empty "comment" line
                if not data[ c_all - 1 ].startswith('*'):
                    bb = bbline("* ", li )
                # else pass preceeding comment line
                else:
                    bb = bbline(data[ c_all - 1 ], li)
                self.bb.update( { kr : bb } )
                kr += 1
                c_noncom += 1
            c_all += 1
        else:
            break

    """ Read the b-f transitions """
    self.bf = {}
    kr = 0
    for i in range(len(data[ c_all : ])):
        li = data[ c_all ]
        c_all += 1
        if not li.startswith('*') and li != '':
            if c_noncom < 3 + self.nk + self.nline + self.ncont:
                # header line :
                if  len(li.split()) != 2:
                    c_noncom += 1
                    # how many frequency points for this transition?
                    nq = int(li.split()[3])
                    self.bf[kr] = bfline(data[ c_all - 1 : c_all+nq ])
                    kr += 1
                    c_all = c_all + nq - 1
            else:
                break
    """ Read a name of the collisional routine """
    # the first not comment line after b-f is the name of the collisional routine
    for li in data[ c_all : ]:
        c_all += 1
        if not li.startswith('*') and li != '':
            self.col_routine = li.strip()
            break
    """ Read number of temperature points and points themselves """
    # for li in data[ c_all : ]:
    #     c_all += 1
    #     if not li.startswith('*') and li != '':
    #         if 'TEMP' in li:
    #             # atom.temp = []
    #             atom.n_temp = 0
    #             break
    # for li in data[ c_all : ]:
    #     if not li.startswith('*') and li != '':
    #         atom.n_temp = int(li.split()[0])
    #         t_points = np.array(li.split()[1:]).astype(float)
    #         atom.temp = t_points
    #         c_all += 1
    #         break
    # for li in data[ c_all + 1 : ]:
    #     if not li.startswith('*') and li != '':
    #         if len(atom.temp) < atom.n_temp:
    #             t_points = np.array(li.split()).astype(float)
    #             atom.temp = np.hstack((atom.temp, t_points))
    #             c_all += 1
    #         else:
    #             c_all += 1
    #             break
    # # self.col = {}
    # # for now I'll just keep the collisions as a bunch of lines
    self.col = data[ c_all : -1 ]
    # print(self.col)
    # TO BE FINISHED
    return

def write_atom(self, file):
    with open(file, 'w') as f:
        for li in self.header:
            f.write(li + '\n')
        f.write("%s \n" %(self.element))
        f.write("%10.2f %10.3f \n" %(self.abund, self.atomic_weight))
        f.write("%.0f %.0f %.0f %.0f \n" %(self.nk, self.nline, self.ncont, self.nrfix) )
        # energy system
        f.write("* Energy, Stat. weight / multiplicity, Label, Ion\n")
        for i in range(len( self.en )):
            f.write(" %10.5f %10.2f '%s' %.0f \n" %(self.en[i], self.g[i], self.label[i], self.ion[i]) )
        # b-b transitions
        for kr in range(self.nline):
            bb = self.bb[kr]
            f.write("%s \n" %bb.comment_line )
            f.write("%4.0f %4.0f %10.4E %6.0f %6.1f %4.1f %4.0f %10.4E %10.4f %10.4E %s\n" \
                %(bb.j, bb.i, bb.f_osc, bb.nq, bb.qmax, bb.q0, bb.iwide, bb.ga, bb.gvw, bb.gs, bb.profile_type) )
        # b-f transitions
        for kr in range(self.ncont):
            bf = self.bf[kr]
            f.write("%4.0f %4.0f %10.4E %5.0f %3.0f %3.0f \n" \
                %(bf.j, bf.i, bf.x[0], bf.nq, -1.0, 0.0) ) # what's -1.0 and 0.0 ?
            for i in range(len(bf.x)):
                f.write(" %10.4f %10.4E \n" %(bf.wave[i], bf.x[i]) )
        # write collisional data
        f.write("%s \n" %self.col_routine)
        # f.write("TEMP \n")
        # temporary solution for collisional rates
        for line in self.col:
            f.write("%s \n" %line)
        # signal end
        f.write("END")
    return


class model_atom(object) :
    def __init__(self, file, comment=''):
        read_atom(self, file)
        """
        A small comment line from the config file.
        Written to the header of the NLTE binary grid
        """
        if len(comment) > 100:
            print("Please, use a shorter comment for atom_comment. Stopped")
            exit(1)
        else:
            self.info = comment

class bbline():
    def __init__(self, com_line, data_line):
        """
        initialised by reading data from the two lines in the model atom, e.g.:
        *  CA II 3P6 4P 2PO 1/2  CA II 3P6 4S 2SE             3968.486
        4  1  3.16E-01 251  300.  3. 0  1.42E08  234.223  5.458E-07
        """
        self.comment_line = com_line.strip()
        data_line = data_line.strip()
        self.j, self.i = np.array(data_line.split()[0:2]).astype(int)
        self.f_osc = float(data_line.split()[2])
        self.nq = int(data_line.split()[3])
        self.qmax, self.q0 = np.array(data_line.split()[4:6]).astype(float)
        self.iwide = int(data_line.split()[6])
        self.ga, self.gvw, self.gs = np.array(data_line.split()[7:10]).astype(float)
        if len(data_line.split()) > 10:
            self.profile_type = data_line.split()[-1].strip()
        else:
            print("Unknown profile for transition %.0f -- > %.0f, setting to VOIGT" %(self.i, self.j))
            self.profile_type = 'VOIGT'


class bfline():
    def __init__(self, data):
        """
        *  UP  LO  F        NQ   QMAX  Q0
        6   5 3.06e-18  4 -1.0 0.0
          1420.40  3.06e-18
          1402.77  2.94e-18
          1335.00  2.48e-18
          1239.78  1.91e-18
        """
        li = data[0]
        self.j, self.i = np.array(li.split()[0:2]).astype(int)
        # self.x_max = float(li.split()[2])
        self.nq = int(li.split()[3])
        self.wave, self.x = [], []
        for li in data[1:]:
            wave, x = np.array(li.split()).astype(float)
            self.wave.append(wave)
            self.x.append(x)


if __name__ == '__main__':
    atom = model_atom('/Users/Semenova/phd/nlte/ca/atom.ca6')
    # one can change basically anything in the model atom at this point
    atom.abund = 10.0
    write_atom(atom, './test_atom.dat')


    # change_abund('/Users/Semenova/phd/nlte/ca/atom.ca6', './test.atom', 5)
