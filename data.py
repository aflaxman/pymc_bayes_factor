""" Load Human Development Index and Total Fertility Rate data from Nature Paper csv

Methods/Members
---------------
all - generate list of dicts of data

to_arrays(data) - generate numpy arrays with hdi[i] and tfr[i]
  corresponding to the same country-year

plot(data) - generate summary plot of the data
"""

import csv
from numpy import nan, isnan, array


def plot_1(data=all):
    from pylab import plot, axis, legend
    
    hdi75 = [float(d['HDI.1975'] or -1) for d in data]
    hdi05 = [float(d['HDI.2005'] or -1) for d in data]
    tfr75 = [float(d['TFR.1975'] or -1) for d in data]
    tfr05 = [float(d['TFR.2005'] or -1) for d in data]

    plot(hdi75, tfr75, 'bs', alpha=.75, markeredgecolor='b', label='1975')
    plot(hdi05, tfr05, 'r^', alpha=.75, markeredgecolor='r', label='2005')
    axis([.3, 1, 1, 8])
    legend()
    xlabel('Human development index')
    ylabel('Total fertility rate')
    
    
def plot_2(data=all):
    from pylab import plot, axis, legend, xlabel, ylabel
    
    hdi75 = [float(d['HDI.1975'] or -1) for d in data]
    hdi05 = [float(d['HDI.2005'] or -1) for d in data]
    tfr75 = [float(d['TFR.1975'] or -1) for d in data]
    tfr05 = [float(d['TFR.2005'] or -1) for d in data]

    plot(hdi75, tfr75, 'b.', alpha=1., label='1975')
    plot(hdi05, tfr05, 'r.', alpha=1., label='2005')
    
    hdi = [[float(d['HDI.%d'%y] or -1) for y in range(1976,2005)] for d in data]
    tfr = [[float(d['TFR.%d'%y] or -1) for y in range(1976,2005)] for d in data]

    for x, y in zip(hdi, tfr):
        plot(x, y, 'k.', alpha=.3)

    axis([.3, 1, 1, 8])
    legend()
    xlabel('Human development index')
    ylabel('Total fertility rate')


def to_arrays(data):
    """ Convert list of data dicts to array

    Parameters
    ----------
    data : list of dicts
      each d in data is expected to contain keys of the form
      'HDI.YYYY' and 'TFR.YYYY', which are either '' or strings that
      can be converted to floats, for YYYY in range(1976, 2006)
    """
    
    hdi = []
    tfr = []
    for d in data:
        for y in range(1976, 2006):
            if d['HDI.%d'%y] == '' or d['TFR.%d'%y] == '':
                continue
            hdi.append(float(d['HDI.%d'%y]))
            tfr.append(float(d['TFR.%d'%y]))
    hdi = array(hdi)
    tfr = array(tfr)

    return hdi, tfr


all = [d for d in csv.DictReader(open('nature08230-s2.csv'))]
hdi, tfr = to_arrays(all)

