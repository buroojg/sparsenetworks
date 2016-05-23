#!/usr/bin/python

"""
Handles plotting of data created with an WithOutput-object.

It creates similar plots as plot_output_total.py, but for separate populations.

Usage: from command line tou can call 'python plot_output_separate.py /directory/where/the/data/is'.
It is recommended, however, to do it with ipython (because it causes less problems keeping the 
plots open and doesn't close them immediately again.)
In ipython call 'run plot_output_separate.py /directory/where/the/data/is'.
The plots will be saved in the directory of the data.
By default, it will only save the plots, and not show them while running the script.
You can, however, pass the option '-show' when calling the script,
in which case the plots will only be shown, not saved.
If you also add the option '-save', e.g. '-show -save' or '-save -show',
the plots will be shown and saved. 
"""



import matplotlib.pyplot as plt
import numpy as np
import random as r

import sys
sys.path.append('..')
sys.path.append('../../simulation')

from sparsenetworks import output_analyzer as ana


# path to folder where output files are stored
folder=sys.argv[1]

# dictionary that keeps track wether plots should be showed or saved or both
options={'show':False,'save':True}

# reads in options put in via command line
if len(sys.argv)>2:
    options['save']=False
    if sys.argv[2:].count('-show'):
        options['show']=True
    if sys.argv[2:].count('-save'):
        options['save']=True

a=ana.Analyzer(folder)



## ---------------- spike trains plot -----------------------
sys.stdout.write('spike trains ...')

a.read_spikes()

trains_indices=[]
for i in range(0,len(a.parameters['N'])):
    trains_indices.append(r.sample(np.arange(a.parameters['N'][:i].sum()+1,a.parameters['N'][:i+1].sum()),30))

f,ax=plt.subplots(len(a.parameters['N']),1,sharex=True)
f.subplots_adjust(left=0.1,right=0.99,bottom=0.1,top=0.9,hspace=0.01)

for i in range(0,len(ax)):
    a.plot_spike_trains(ax[i],trains_indices[i])
    ax[i].set_ylabel('pop '+str(i+1))

ax[-1].set_xlabel('$t\;[T]$',fontsize=18)
ax[0].set_title('spike trains',fontsize=18)

#plt.show(block=True)

if options['save']:
    f.savefig(folder+'/spike_trains_separate.png')
if options['show']:
    f.show()

sys.stdout.write('\r')
sys.stdout.write('spike trains done\n')

# ------------------------------------------

# ---------- rates plot --------------------
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

#-----------------------------------





#-------------- CV plot ------------

sys.stdout.write('CV ...')

CV=a.compute_CV()

#print CV.shape


f,ax=plt.subplots(1,1)

bins=np.linspace(0.0,2.0,40)


for i in range(0,len(a.parameters['N'])):

    a.plot_CV(ax,CV[a.parameters['N'][:i].sum():a.parameters['N'][:i+1].sum()],bins,hist_kwargs={'alpha':1.0-i*0.8/len(a.parameters['N']),'label':'pop '+str(i+1)})


ax.legend()

if options['save']:
    f.savefig(folder+'/CV_separate.png')
if options['show']:
    f.show()


del CV


del a.spike_array

sys.stdout.write('\r')
sys.stdout.write('CV done\n')

# ----------------------------------------------------


## ---------- phases plot ----------------------------
sys.stdout.write('phases ...')

a.read_phases()

f,ax=plt.subplots(len(a.parameters['N']),1,sharex=True)

#phases_indices=[]
for i in range(0,len(a.parameters['N'])):
    phases_indices=r.sample(np.arange(a.parameters['N'][:i].sum()+1,a.parameters['N'][:i+1].sum()+1),5)

    for ind in phases_indices:
        a.plot_single_phase_dynamics(ax[i],ind)
        ax[i].set_ylabel('pop '+str(i+1))

f.subplots_adjust(left=0.1,right=0.99,bottom=0.1,top=0.9,hspace=0.05)

if options['save']:
    f.savefig(folder+'/phases.png')
if options['show']:
    f.show()



sys.stdout.write('\r')
sys.stdout.write('phases done\n')

# ------------------------------------------------------------------
