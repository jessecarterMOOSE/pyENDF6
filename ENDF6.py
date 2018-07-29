"""
Tools to read an ENDF-6 data file.

From https://t2.lanl.gov/nis/endf/intro05.html
An ENDF-format nuclear data library has an hierarchical structure by tape, material, file, and section, denoted by numerical identifiers.

Tape is a data-file that contains one or more ENDF materials in increasing order by MAT.
    Each material contains several files in increasing order by MF.
    Each file contains several sections in increasing order by MT.

MAT labels an the target material as ZZXX, where XX starts from 25 for the lightest common isotope and increases in steps of 3 to allow for isomers.
    MAT=125   H-1
    MAT=128   H-2
    MAT=2625  Fe-54

MF labels an ENDF file to store different types of data:
    MF=1  descriptive and miscellaneous data,
    MF=2  resonance parameter data,
    MF=3  reaction cross sections vs energy,
    MF=4  angular distributions,
    MF=5  energy distributions,
    MF=6  energy-angle distributions,
    MF=7  thermal scattering data,
    MF=8  radioactivity data
    MF=9-10  nuclide production data,
    MF=12-15  photon production data, and
    MF=30-36  covariance data.

MT labels an ENDF section, usually used to hold different reactions, e.g.
    MT=1   total cross section
    MT=2   elastic scattering
    MT=3  Total photo-absorption cross section
    MT=16  (n,2n) reaction
    MT=18  fission
    MT=102 radiative capture
"""

import numpy as np


slices = {
    'MAT': slice(66, 70),
    'MF': slice(70, 72),
    'MT': slice(72, 75),
    'line': slice(75, 80),
    'content': slice(0, 66),
    'data': (slice(0, 11), slice(11, 22), slice(22, 33), slice(33, 44), slice(44, 55), slice(55, 66))}


def read_float(v):
    """
    Convert ENDF6 string to float
    """
    if v.strip() == '':
        return 0.
    try:
        return float(v)
    except ValueError:
        # ENDF6 may omit the e for exponent
        return float(v[0] + v[1:].replace('+', 'e+').replace('-', 'e-'))  # don't replace leading negative sign


def read_line(l):
    """Read first 6*11 characters of a line as floats"""
    return [read_float(l[s]) for s in slices['data']]


def read_table(lines):
    """
    Parse tabulated data in a section
    https://t2.lanl.gov/nis/endf/intro07.html
    https://t2.lanl.gov/nis/endf/intro08.html
    https://t2.lanl.gov/nis/endf/intro09.html
    """
    # header line 1: (100*Z+A), mass in [m_neutron]
    # [MAT, 3, MT/ ZA, AWR, 0, 0, 0, 0] HEAD

    # header line 2: Q-value and some counts
    # [MAT, 3, MT/ QM, QI, 0, LR, NR, NP/ EINT/ S(E)] TAB1
    f = read_line(lines[1])
    nS = int(f[4])  # number of interpolation sections
    nP = int(f[5])  # number of data points

    # header line 3: interpolation information
    # [MAT, 3, 0/ 0.0, 0.0, 0, 0, 0, 0] SEND
    # 1   y is constant in x (constant, histogram)
    # 2   y is linear in x (linear-linear)
    # 3   y is linear in ln(x) (linear-log)
    # 4   ln(y) is linear in x (log-linear)
    # 5   ln(y) is linear in ln(x) (log-log)
    # 6   y obeys a Gamow charged-particle penetrability law

    # data lines
    x = []
    y = []
    for l in lines[3:]:
        f = read_line(l)
        x.append(f[0])
        y.append(f[1])
        x.append(f[2])
        y.append(f[3])
        x.append(f[4])
        y.append(f[5])
    return np.array(x[:nP]), np.array(y[:nP])


def find_file(lines, MF=1):
    """Locate and return a certain file"""
    v = [l[slices['MF']] for l in lines]
    n = len(v)
    cmpstr = '%2s' % MF       # search string
    i0 = v.index(cmpstr)            # first occurrence
    i1 = n - v[::-1].index(cmpstr)  # last occurrence
    return lines[i0: i1]


def find_section(lines, MF=3, MT=3):
    """Locate and return a certain section"""
    v = [l[70:75] for l in lines]
    n = len(v)
    cmpstr = '%2s%3s' % (MF, MT)       # search string
    i0 = v.index(cmpstr)            # first occurrence
    i1 = n - v[::-1].index(cmpstr)  # last occurrence
    return lines[i0: i1]


def list_content(lines):
    """Return set of unique tuples (MAT, MF, MT)"""
    s0 = slices['MAT']
    s1 = slices['MF']
    s2 = slices['MT']
    content = set(((int(l[s0]), int(l[s1]), int(l[s2])) for l in lines))

    # remove section delimiters
    for c in content.copy():
        if 0 in c:
            content.discard(c)
    return content


def get_num_subsections(lines):
    """Return number of subsections within the file, taken as the 5th column of the first line"""
    values = read_line(lines[0])
    return int(values[4])


def parse_header(line):
    """Return header information - generally integers except for second column which may be float"""
    a, b, c, d, e, f = read_line(line)
    return int(a), b, int(c), int(d), int(e), int(f)


def find_yields(lines):
    """Find available reaction products and return a dict with the product ZAP ID as key"""

    # yield data is in MF6/MT5
    lines = find_section(lines, MF=6, MT=5)

    # return whole subsection if thre are no subsections
    if get_num_subsections(lines) == 0:
        return lines

    # if multiple subsections exist ,make a dict with the product as key
    yield_dict = {}

    # parse the section into subsections
    # unfortunately, there are no good delimiters that tell when a subsections starts and stops, so we will go through
    # the section line by line and use the number of data points in each subsection as a guide
    line_num = 1  # keeping track of the FILE line number, not list index, so we will always need to subtract 1

    # we are using the read_table function but it assumes 3 header rows for the section, but the yield subsection
    # only has 2 header lines, so we will save the first line and tack it top of each subsections
    head_line = lines[line_num-1]

    line_num += 1  # move up

    # start the loop
    while line_num <= len(lines):
        ZAP, AWR, LIP, LAW, NR, NP = parse_header(lines[line_num-1])  # parse TAB1 line

        # check values
        if NR > 1:
            print("error: can not yet handle multiple interpolation ranges")
            quit()

        if LAW > 0:
            print("error: can only handle LAW=0 (no angular data) for now, file had LAW:", LAW)
            quit()

        # first subsection will be the yield data, so get the lines and save in the dict
        data_lines = int(np.ceil(NP/3.0))  # 3 data pairs per line
        yield_section = lines[line_num-1:line_num-1 + data_lines+2]  # include 2 TAB1 lines
        # dict key in the form of "ZAP(m)", e.g. "41091m" for Nb-91m, "41091" fo Nb-91 (ground state)
        key = str(ZAP)
        if LIP > 0:
            key = key + 'm'
        yield_dict[key] = [head_line] + yield_section  # add section to dict with header

        line_num += data_lines + 2  # move up past yield data

    return yield_dict


if __name__ == "__main__":
    # run an example
    import matplotlib.pyplot as plt
    import os.path

    # open library file and read it in
    f = open(os.path.join('library_files', '40090'))
    lines = f.readlines()
    f.close()

    product = '41091m'

    # get ready to plot it
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # get total reaction cross section
    x1, y1 = read_table(find_section(lines, MF=3, MT=5))
    x1 = x1*1.0e-6  # convert energy to MeV

    # get yield section
    yield_dict = find_yields(lines)
    print("found these products:")
    for i, key in enumerate(sorted(yield_dict.keys())):
        print(i+1, key)
    # check if yield of product exists before trying to evaluate it
    if yield_dict.get(product):
        x2, y2 = read_table(yield_dict.get(product))
        x2 = x2*1.0e-6  # convert to MeV
        ax2.plot(x2, y2, 'o--', label='yield data for ' + product)

    ax1.plot(x1, y1, 'o-', label='total reaction cross section')
    ax1.set_xlabel('energy (MeV)')
    ax1.set_ylabel('cross section (barns)')
    ax2.set_ylabel('yield fraction')
    ax1.set_xlim([0, 10])
    ax1.set_yscale('symlog', linthreshy=1e-10)
    ax2.set_yscale('symlog', linthreshy=1e-4)
    ax1.legend(loc='upper left', fontsize=8)

    fig.tight_layout()
    plt.show()
