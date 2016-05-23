#!/usr/bin/python

"""
Handles plotting of data created with an WithOutput-object.

It creates plots of spike trains for 50 random neurons,
a plot of average rates for each population,
a plot of CV,
a plot of the phases of 5 randomly chosen neurons.

Usage: from command line call 'python plot_output_total.py /directory/where/the/data/is'.
The plots will be saved in the directory of the data.
By default, it will only save the plots, and not show them while running the script.
You can, however, pass the option '-show' when calling the script,
in which case the plots will only be shown, not saved.
If you also add the option '-save', e.g. '-show -save' or '-save -show',
the plots will be shown and saved. 
"""



import matplotlib.pyplot as plt
#plt.ion()
import numpy as np
import random as r

import sys
sys.path.append('..')
sys.path.append('../../Simulation')

from sparsenetworks import output_analyzer as ana

folder=sys.argv[1]

options={'show':False,'save':True}

if len(sys.argv)>2:
    options['save']=False
    if sys.argv[2:].count('-show'):
        options['show']=True
    if sys.argv[2:].count('-save'):
        options['save']=True

a=ana.Analyzer(folder)

sys.stdout.write('spike trains ...')

a.read_spikes()

f,ax=plt.subplots(1,1)

a.plot_spike_trains(ax,r.sample(np.arange(1,a.spike_array.shape[1]-1),50))

ax.set_xlabel('$t\;[T]$',fontsize=18)
ax.set_ylabel('spike trains',fontsize=18)

#plt.show(block=True)

if options['save']:
    f.savefig(folder+'/spike_trains.png')
if options['show']:
    f.show()

sys.stdout.write('\r')
sys.stdout.write('spike trains done\n')



sys.stdout.write('rates ...')

rates=a.compute_rates(0.2)

f,ax=plt.subplots(1,1)

a.plot_rates(ax,rates)

#plt.show(block=True)

if options['save']:
    f.savefig(folder+'/rates.png')
if options['show']:
    f.show()


sys.stdout.write('\r')
sys.stdout.write('rates done\n')

del rates

#try:
if True:

    sys.stdout.write('CV ...')

    CV=a.compute_CV()

    #print CV.shape
    #print VMR.shape

    f,ax=plt.subplots(1,1)

    bins=np.linspace(0.0,2.0,40)
    a.plot_CV(ax,CV,bins)
    ax.set_title('')
    ax.set_xlabel('CV of ISI',fontsize=18)
    ax.set_ylabel('$N_{\text{neurons}}$',fontsize=18)
    if options['save']:
        f.savefig(folder+'/CV.png')
    if options['show']:
        f.show()


    del CV

    del a.spike_array

    sys.stdout.write('\r')
    sys.stdout.write('CV done\n')


#except:
    #pass


sys.stdout.write('phases ...')

a.read_phases()

f,ax=plt.subplots(1,1)

indices=r.sample(np.arange(1,a.phase_array.shape[1]),5)
for ind in indices:
    a.plot_single_phase_dynamics(ax,ind)

if options['save']:
    f.savefig(folder+'/phases.png')
if options['show']:
    f.show()



sys.stdout.write('\r')
sys.stdout.write('phases done\n')
