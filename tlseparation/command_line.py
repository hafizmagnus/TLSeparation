#!/usr/bin/env python
"""
Command line parser for the tlseparation package.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

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
