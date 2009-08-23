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


def plot(data):
    countries = [d['country'] for d in data]
    hdi = [[float(d['HDI.%d'%y] or 'nan') for y in range(1976,2006)] for d in data]
    tfr = [[float(d['TFR.%d'%y] or 'nan') for y in range(1976,2006)] for d in data]

    for ii, c in enumerate(country):
        plot(hdi[ii],tfr[ii],'.-', linewidth=2, alpha=.5)
        text(hdi[ii][-1],tfr[ii][-1],country[ii],alpha=.5,size=8)


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

