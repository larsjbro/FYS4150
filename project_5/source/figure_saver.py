"""
Created on 8. des. 2017

@author: ljb
"""
from __future__ import division, absolute_import
import matplotlib.pyplot as plt

_DPI = 250
_SIZE = 0.7


def figure_set_default_size():
    F = plt.gcf()
    DefaultSize = [8 * _SIZE, 6 * _SIZE]
    print "Default size in Inches", DefaultSize
    print "Which should result in a %i x %i Image" % (_DPI * DefaultSize[0],
                                                      _DPI * DefaultSize[1])
    F.set_size_inches(DefaultSize)
    return F


def my_savefig(filename):
    F = figure_set_default_size()
    F.tight_layout()
    F.savefig(filename, dpi=_DPI)
