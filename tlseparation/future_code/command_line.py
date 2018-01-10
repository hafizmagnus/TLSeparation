# Copyright (c) 2017, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.2.1.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import sys
from glob import glob
import numpy as np
from separation import main as sep
import itertools
import os
import ast


def main():

    """
	Usage: tlseparation config_file.txt output_path
	"""

    with open(sys.argv[1], 'r') as f:
        contents = f.readlines()

    try:
        output_path = sys.argv[2]
    except:
        output_path = ''

    vars_ = {}

    for i in contents:
        key, value = i.split('\n')[0].split('=')

        if len(value) > 0:
            try:
                value = [float(value)]
            except:
                try:
                    value = ast.literal_eval(value)
                except:
                    pass
            vars_[key] = value

    if os.path.isfile(vars_['file']) is True:
        filelist = [vars_['file']]
    elif os.path.isdir(vars_['file']) is True:
        filelist = glob(vars_['file'] + '*.txt')
    else:
        raise Exception('Please, insert a valid file or folder to import\
 the data from.')

    vars_.pop('file')
    for f in filelist:

        filename = f.split(os.sep)[-1].split('.txt')[0]
        print(filename)

        try:
            arr = np.loadtxt(f, delimiter=' ')
        except:
            arr = np.loadtxt(f, delimiter=',')

        for p in dict_product(vars_):
            print p

            wood, leaf, params = sep(arr, **p)

            np.savetxt(output_path + filename + '_' +
                       '_'.join(map(str, params)) + '_wood.txt', wood)
            np.savetxt(output_path + filename + '_' +
                       '_'.join(map(str, params)) + '_leaf.txt', leaf)


def dict_product(dicts):
    return (dict(itertools.izip(dicts, x)) for x in
            itertools.product(*dicts.itervalues()))


if __name__ == '__main__':

    main()
