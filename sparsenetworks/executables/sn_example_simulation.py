#!/usr/bin/python

"""
@author Ilyas Kuhlemann
@email ilyasp.ku@gmail.com

This is an example of a set-up for a simulation with two inhibitory networks.
It creates the folder 'out_example' and writes data (neurons' phases and emitted spikes) to it.
These data can be visualized, e.g. by using example_plot.py.
"""

## numpy is required to define arrays of connection strength
import numpy as np

import sys
## adds the upper folder to python path for this session.
sys.path.append('..')

## system holds the class WithOutput, our main class to run a simulation and write its output to files.
from sparsenetworks import system as S

## set the number of neurons per population
# (in this case, two populations with 800 neurons each)
N=[800,800]

## specify neurons' properties
I=[4,4]
gamma=[1,1]


## J_int sets connection strengths among internal populations
J_int=np.ones((2,2))
# (indices [i,j]: from population j to population i)
J_int[0,0]=-0.6
J_int[1,1]=-0.6
J_int[0,1]=-0.3
J_int[1,0]=-0.3


## set number of neurons per external population
# (one external population with 1000 neurons)
N_ext=[800]
## rates of external neurons, one rate per external population
rates=[.1] 
## connection strengths from external to internal
J_ext=np.ones((2,1))*0.2


## K sets the average number of connection one neuron receives from neurons of any population.
K=26
## delay between emitting and receiving (internal) spikes in the network
tau=0.05

 
## Create a WithOutput object with previously specified parameters
s=S.WithOutput(N=N,J_int=J_int,I=I,gamma=gamma,K=K,tau=tau,N_ext=N_ext,J_ext=J_ext,rates=rates)

## set for how many periods the simulation should run
t_end=8

## run simulation and write output to folder 'out_example'
s.run(t_end,'out_example')
