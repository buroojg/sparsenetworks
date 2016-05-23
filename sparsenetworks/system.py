#-*- coding: utf-8 -*-

## @author Diemut Regel
# @author Francesca Schönsberg
# @author Burooj Ghani
# @author Ilyas Kuhlemann
# @contact ilyasp.ku@gmail.com
# @date 25.03.2015


"""
@package system

The class System represents a model of a system of one or more populations of leaky integrate and fire neurons.
"""

import numpy as np
from scipy.sparse import csr_matrix,lil_matrix


## Defines the maximum memory that's available for the phases-array. When it's full it gets written to a file and emptied.
# We never explored the limits, but you shouldn't reserve more than half you computer's memory for this.
OUTPUT_MEMORY=1.5*10**9


class System:
    """
    System of one or more populations of leaky integrate and fire neurons.
    Neurons share the same statistical properties among one population; they are modeled as leaky integrate and fire neurons in phase representation as in 'How chaotic is the balanced state' by Jahnke, Memmesheimer and Timme.
    """

    
    ##Initializes a 'System'-object.
    # @param N One-dimensional array or list containing the number of individual neurons for each population.
    # @param l List of firing rates of external inputs (one constant rate per external population/source).
    # @param J_int Two-dimensional array containing connection strengths J_int[k,l] for connections of neurons from population l to neurons of population k.
    # @param J_ext Two-dimensional array containing connection strengths J_ext[k,m] for connections from external source m to neurons of population k.
    # @param I List containing parameters (currents, one value per population) defining properties of the leaky integrate and fire neurons.
    # @param gamma List containing parameters (leak-factor?, one value per population) defining properties of the leaky integrate and fire neurons.
    # @param K Average number of connections all neurons of one population receive from any other population.
    # @param tau Delay between sending and receiving an (internal) spike.
    def __init__(self,N=np.array([400,100]),J_int=None,I=[1.,1.],gamma=[0.2,0.2],K=80,tau=0.05,N_ext=[],J_ext=np.array([]),rates=[]):
        self.N=np.array(N)
        self.N_ext=np.array(N_ext)
        self.tau=tau

        ## Holds an array of size N (total number of neurons) containing paraters I and gamma for each neuron.
        self.I_gamma=np.ones((2,self.N.sum()))

        #for each population...
        for i in range(0,self.N.shape[0]):
            if i==0:
                index_i=0
            else:
                index_i=self.N[:i].sum()
            # ... set value in first column to I[i] and in second column to gamma[i]
            self.I_gamma[0,index_i:index_i+self.N[i]]=self.I_gamma[0,index_i:index_i+self.N[i]]*I[i]
            self.I_gamma[1,index_i:index_i+self.N[i]]=self.I_gamma[1,index_i:index_i+self.N[i]]*gamma[i]
            
        self.t=0
        self.events=[]

        ## one rate value per external population
        self.rates=rates

        ## for each external neuron, holds the (arrival) time of its next spike  
        self.external_events=np.ones(self.N_ext.sum())
        
        # compute initial inter-spike intervals (for each neuron, time when next spike arrives at internal population)
        self.get_initial_ISI()


        # create random initial phases between 0 and 1
        self.create_phases()

        ## weight matrix, representing the connections and their strengths from any neuron to any other neuron
        self.weight_matrix=csr_matrix(np.concatenate([self.create_weight_matrix(J_int,K),self.create_ext_weight_matrix(J_ext,K)],1))



    def jump_to_next_event(self):
        """
        One step of simulation; searches for and handles the next event.

        Checks whether the next event is the arrival of one or more internal/external spikes or the reset and spike of one or more neurons.
        Updates system time self.t by the difference dt between current time and the event found to happen next.
        """


        # location holds the information of which spikes are found to be the next event
        location=0 # 0 for no spike, 1 for internal, 2 for external, 3 for internal and external simultaneously

        ## set dt (time to next event) to a random too large value ... this will only be used for comparison with time to next emission of spike if there are no other (already emitted) spikes to consider at all
        dt=500.0

        if len(self.events): # events=[] --> events=[(time_of_arrival,spike_vector),....,]
            dt=self.events[0][0]-self.t
            location=1


        ##check for external event
        if self.external_events.any(): # if there is at least one external event ...
            min_dt=self.external_events.min()-self.t # check when it will occur
            if min_dt<dt:
                dt=min_dt
                location=2
            elif min_dt==dt:
                location=3
            
        # from here on: min_dt = time until the next phase reaches the threshold without any incoming spikes.
        min_dt=1-self.phases.max()
        
        if min_dt<dt: # True if any phase reaches threshold before a spike arrives.
            
            
            # min_dt can be smaller than 0, if a previous spike caused a phase to exceed 1
            dt=max(min_dt,0) 

            # update phases with temporal difference dt
            
            self.phases=self.phases+dt 
            
            
            # find phases that exceed 1
            spike_id=np.where(self.phases>=1.0)
            # create an array with 1 for each neuron that spikes, 0 else
            spikes=np.zeros_like(self.phases)
            spikes[spike_id]=1.0

            # set phases of neurons that spiked to 0        
            self.phases=self.phases-spikes*self.phases
            # append spike vector to list of events
            self.events.append([self.tau+self.t+dt,spikes]) #the spike vector [which refers to spikes that happen at self.t+dt] is saved at time self.t+dt+tau [which is already the receiving time]
            
        
        elif min_dt==dt: # True if time of next spike arrivals equals time of next phase reaching threshold
            
            # indices of neurons that reach thresholds next
            neuron_id=np.where(self.phases==self.phases.max())[0] 
            
            # min_dt can be smaller than 0, if a previous spike caused a phase to exceed 1
            dt=max(min_dt,0)
            self.phases=self.phases+dt
            
            # find phases that exceed 1
            spike_id=np.where(self.phases>=1.0)
            # create an array with 1 for each neuron that spikes, 0 else
            spikes=np.zeros_like(self.phases)
            spikes[spike_id]=1.0

            # set phases of neurons that spiked to 0
            self.phases=self.phases-spikes*self.phases
            # append spike vector to list of events
            self.events.append([self.tau+self.t+dt,spikes]) #the spike vector [which refers to spikes that happen at self.t+dt] is saved at time self.t+dt+taum [which is already the receiving time]

            
            if location==1: # if next event is an internal spike

                # get the whole spike vector (concatenated from internal and external spikes)
                # (in this case, external spikes is a 0-array)
                spike_vector=np.concatenate([self.events.pop(0)[1],np.zeros(self.N_ext.sum())])

            elif location==2: # if next event is an external spike

                
                ext_vect=np.zeros(self.N_ext.sum()) # create an empty array to contain possible external spikes
                indices=np.where(self.external_events==self.t+dt)[0] # find all external spikes arriving at t+dt (their indices in the external populations)
                for index in indices: # for each external neurons that is spiking ...
                    ext_vect[index]=1 # ... set value to one in the vector of external spikes 
                    
                    # keeps track with which rate to draw a new ISI
                    rate_index=None 
                    for i in range(1,len(self.rates)+1):
                        if index < self.N_ext[:i].sum():
                            rate_index=i-1
                            break
                    
                    # draw new ISI for external neuron index
                    self.external_events[index]=self.get_InterSpikeInterval(self.rates[rate_index])
                
                # get the whole spike vector (concatenated from internal and external spikes)
                # (in this case, internal spikes is a 0-array)
                spike_vector=np.concatenate([np.zeros(self.N.sum()),ext_vect])

            elif location==3: # if external and internal spikes arrive at the same time

                # create an empty array to contain possible external spikes
                ext_vect=np.zeros(self.N_ext.sum()) 
                # find all external spikes arriving at t+dt (their indices in the external populations)
                indices=np.where(self.external_events==self.t+dt)[0] 

                for index in indices: # for each external neurons that is spiking ...
                    ext_vect[index]=1 # ... set value to one in the vector of external spikes 
                    
                    # keeps track with which rate to draw a new ISI
                    rate_index=None
                    for i in range(1,len(self.rates)+1):
                        if index < self.N_ext[:i].sum():
                            rate_index=i-1
                            break
                    
                    # draw new ISI
                    self.external_events[index]=self.get_InterSpikeInterval(self.rates[rate_index])

                # get the whole spike vector (concatenated from internal and external spikes)
                spike_vector=np.concatenate([self.events.pop(0)[1],ext_vect])
                
            # calculate change in voltage (epsilon) for each neuron
            epsilon=self.epsilon(spike_vector)   
            # update phases using transfer function h
            self.phases=self.h(epsilon)

            
            for i in range(len(neuron_id)):
                if self.phases[neuron_id[i]]>1:  #if the received spike is elicting another spike immediatley we neglet this further spike
                    self.phases[neuron_id[i]]=0
                    

        else: # next spikes arrive before a neuron reaches its threshold on its own
            

            self.phases=self.phases+dt
            
            

            # next spike is an internal spike ... 
            if location==1: 
                # ... so the external part is a 0-array
                spike_vector=np.concatenate([self.events.pop(0)[1],np.zeros(self.N_ext.sum())])

            # next spike is an external spike ...
            elif location==2:

                
                ext_vect=np.zeros(self.N_ext.sum())
                indices=np.where(self.external_events==self.t+dt)[0]
                for index in indices:
                    ext_vect[index]=1
                    
                    rate_index=None

                    for i in range(1,len(self.rates)+1):
                        if index < self.N_ext[:i].sum():
                            rate_index=i-1
                            break
                    
                    self.external_events[index]=self.get_InterSpikeInterval(self.rates[rate_index])
                
                # ... so the internal part is empty
                spike_vector=np.concatenate([np.zeros(self.N.sum()),ext_vect])


            # next external and internal spikes arrive at same time
            elif location==3:

                ext_vect=np.zeros(self.N_ext.sum())
                indices=np.where(self.external_events==self.t+dt)[0]
                for index in indices:
                    ext_vect[index]=1
                    
                    rate_index=0
                    for i in range(1,len(self.rates)+1):
                        if index < self.N_ext[:i].sum():
                            rate_index=i-1
                            break
                    
                    self.external_events[index]=self.get_InterSpikeInterval(self.rates[rate_index])
                spike_vector=np.concatenate([self.events.pop(0)[1],ext_vect])
                

            epsilon=self.epsilon(spike_vector)   
            self.phases=self.h(epsilon)



        # update system time by difference dt between this event and the last event (old system time).
        self.t+=dt 



    ## Run the simulation until system time self.t exceeds t_end.
    # @param t_end Ending time of the run.
    def run(self,t_end=1):
        while self.t<t_end:
            self.jump_to_next_event()
     
            
    ## Update phases according to function H_epsilon(phi) as in 'How chaotic is the balanced state' by Jahnke, Memmesheimer and Timme.

    # H(phi,epsilon)=U^-1[U(phi)+epsilon]
    # @param epsilon Change/jump in potential for each neuron (vector/array with one entry per neuron).
    # @return Updated phases.
    def h(self,epsilon):
        

        # compute the argument to the logarithm
        log_arg=np.exp(-self.I_gamma[1,:]*self.phases)-self.I_gamma[1,:]/self.I_gamma[0,:]*epsilon
        # find those that are invalid arguments for the logarithm
        too_large=np.where(log_arg<=0)[0]
        # set those invalid to some valid value
        log_arg[too_large]=1.0        
        # update phases according to H(.)
        updated_phases=-1./self.I_gamma[1,:]*np.log(log_arg)

        # change those that were invalid before to a value above threshold
        updated_phases[too_large]=1.1

        return updated_phases


    ## Gets the change of the potential for each neuron caused by internal spikes.
    # @param spike_vector Vector of spikes; one entry per neuron (=1 if the neuron spikes, =0 if it does not).
    # @return epsilon=change in potential for each neuron.
    def epsilon(self,spike_vector):
        return self.weight_matrix.dot(spike_vector)

    ## Draws a random inter-spike interval according to given rate, and already adds it up to current system time.
    # @param l_i Rate of the poissonian firing of the neuron for which the next spiking time is drawn.
    # @return Time of next spike.
    def get_InterSpikeInterval(self,l_i):
        return self.t-1./l_i*np.log(np.random.rand())


    def get_initial_ISI(self):
        """
        Fills self.external_events with initial (arrival) times of each external neurons' spikes.
        """
        for n in range(0,len(self.N_ext)):
            if n==0:
                index_i=0
            else:
                index_i=self.N_ext[:n].sum()
            for i in range(0,self.N_ext[n]):
                self.external_events[index_i+i]=self.get_InterSpikeInterval(self.rates[n])




    def create_phases(self):
        """
        Creates an initial random phase for each neuron.
        """
        N=self.N.sum()
        self.phases=np.random.rand(N)


    def create_ext_weight_matrix(self,J_ext,K):
        """
        Creates the external weight matrix, that is, weight of connections from each external neuron to each internal neuron.
        
        First, a matrix of random values between 0 and 1 of appropriate size is created. Then, depending on K and the resulting probability to have a connection, the values in the random matrix are set either to 1 or 0 (connection or no connection).
        The connections' weights are then set according to J_ext.
        @param J_ext Array of connection strength, J_ext[j,i] refers to connections from external population i to internal population j.
        @param K Average number of connections one neuron receives from neurons from any other population.
        @return External weight matrix.
        """
        r=np.random.rand(self.N.sum(),self.N_ext.sum())

        for i in range(0,self.N.shape[0]):

            if i==0:
                index_i=0
            else:
                index_i=self.N[:i].sum()
            p_i=float(K)/self.N[i]
        
            r[index_i:index_i+self.N[i],:]=np.ones(r[index_i:index_i+self.N[i],].shape)-(r[index_i:index_i+self.N[i],]+(1-p_i)).astype(int)
        
            
            for j in range(0,self.N_ext.shape[0]):

                if j==0:
                    index_j=0
                else:
                    index_j=self.N_ext[:j].sum()
                
                r[index_i:index_i+self.N[i],index_j:index_j+self.N_ext[j]]=r[index_i:index_i+self.N[i],index_j:index_j+self.N_ext[j]]*J_ext[i,j]
        return r
        

    def create_weight_matrix(self,J_int,K):
        """
        Creates the weight matrix, that is, weight of connections from any internal neuron to each internal neuron.

        First, a matrix of random values between 0 and 1 of appropriate size is created. Then, depending on K and the resulting probability to have a connection, the values in the random matrix are set either to 1 or 0 (connection or no connection).
        The connections' weights are then set according to J_ext.
        @param J_ext Array of connection strength, J_ext[j,i] refers to connections from external population i to internal population j.
        @param K Average number of connections one neuron receives from neurons from any other population.
        @return Weight matrix.
        """
        r=np.random.rand(self.N.sum(),self.N.sum())

        for i in range(0,self.N.shape[0]):

            if i==0:
                index_i=0
            else:
                index_i=self.N[:i].sum()
            p_i=float(K)/self.N[i]
            #print 'p_i='+str(p_i)
            r[index_i:index_i+self.N[i],:]=np.ones(r[index_i:index_i+self.N[i],].shape)-(r[index_i:index_i+self.N[i],]+(1-p_i)).astype(int)

            
            for j in range(0,self.N.shape[0]):

                if j==0:
                    index_j=0
                else:
                    index_j=self.N[:j].sum()
                
                r[index_i:index_i+self.N[i],index_j:index_j+self.N[j]]=r[index_i:index_i+self.N[i],index_j:index_j+self.N[j]]*J_int[i,j]

        return r
                
           


class WithOutput(System):
    """
    WithOutput inherits the class System. It is very similar, but has some functionalities implemented to write data generated during a simulation to an output folder. Furthermore, it displays some more output on the command line when the simulation is running (progress bar).
    """
    def __init__(self,N=np.array([400,100]),J_int=np.array([]),I=[1.,1.],gamma=[0.2,0.2],K=50,tau=0.05,N_ext=[],J_ext=np.array([]),rates=[]):
        """
        Initializes a 'WithOutput'-object.
        @param N One-dimensional array or list containing the number of individual neurons for each population.
        @param l List of firing rates of external inputs (one constant rate per external population/source).
        @param J_int Two-dimensional array containing connection strengths J_int[k,l] for connections of neurons from population l to neurons of population k.
        @param J_ext Two-dimensional array containing connection strengths J_ext[k,m] for connections from external source m to neurons of population k.
        @param I List containing parameters (currents, one value per population) defining properties of the leaky integrate and fire neurons.
        @param gamma List containing parameters (leak-factor?, one value per population) defining properties of the leaky integrate and fire neurons.
        @param K Average number of connections all neurons of one population receive from any other population.
        @param tau Delay between sending and receiving an (internal) spike.
        """

        self.parameters={'N':np.array(N),'J_int':J_int,'I':I,'gamma':gamma,'K':K,'tau':tau,'N_ext':N_ext,'J_ext':J_ext,'rates':rates}
        
        self.n_files=0
        System.__init__(self,N,J_int,I,gamma,K,tau,N_ext,J_ext,rates)
        

    def run(self,t_end,output_dir):
        """
        Run the simulation until system time self.t exceeds t_end, creates folder output_dir and writes output to it.
        @param t_end Ending time of the run.
        @param output_dir A string specifying the folder to which the output should be written.
        """


        import sys
        import os
        os.system('mkdir -p '+output_dir)

        #if check_dir(output_dir):
        if True:

            import pickle
            with open(output_dir+'/parameters.pickle','wb') as f:
                pickle.dump(self.parameters,f)

            output_size=int(OUTPUT_MEMORY/(self.N.sum()+1)/8)

            #print output_size

            self.outputs=np.zeros((output_size,self.N.sum()+1))

            spikes=lil_matrix((output_size,self.N.sum()+1))

            i=0
            i_spike=0
            i_progess=0
            n_progress=100

            last_t=0
            
            progress=0
            sys.stdout.write('[%-20s] %d%% of t_end' % ('='*(progress/5), progress))

            while self.t<t_end:

                self.jump_to_next_event()

                if len(self.events):
                    if self.events[-1][0]>last_t:
                        spikes[i_spike,0]=self.events[-1][0]-self.tau
                        spikes[i_spike,1:]=self.events[-1][1]
                        last_t=self.events[-1][0]
                        #print str(self.t)+':  '+str(i_spike)
                        i_spike+=1

                self.outputs[i,1:]=self.phases
                self.outputs[i,0]=self.t
                i+=1
                if i==output_size:
                    np.save(output_dir+'/phases'+str(self.n_files)+'.npy',self.outputs)
                    del self.outputs

                    out_spikes=spikes[:i_spike,:].todense()
                    del spikes
                    np.save(output_dir+'/spikes'+str(self.n_files)+'.npy',out_spikes)
                    del out_spikes
                    spikes=lil_matrix((output_size,self.N.sum()+1))

                    self.outputs=np.zeros((output_size,self.N.sum()+1))
                    self.n_files+=1
                    i=0
                    i_spike=0
                    
                if i_progess%n_progress==0:
                    progress=int(self.t/t_end*100)
                    sys.stdout.write('\r')
                    sys.stdout.write('[%-20s] %d%% of t_end' % ('='*(progress/5), progress))
                    sys.stdout.flush()

                    

            np.save(output_dir+'/phases'+str(self.n_files)+'.npy',self.outputs[:i])
            del self.outputs

            out_spikes=spikes[:i_spike,:].todense()
            del spikes
            np.save(output_dir+'/spikes'+str(self.n_files)+'.npy',out_spikes)
            del out_spikes
                    

            self.n_files+=1
                
        else:
            print "ERROR: something wrong with given directory"
            
    """
    def get_hist(self,i,n_bins=10):
        return self.outputs[i][0],np.histogram(self.outputs[i][1],n_bins)
    """


def check_dir(d):
    import os
    try:
        f=open(d+'/test_check_dir.txt','w')
        f.write('test test test')
        f.close()
        os.system('rm '+d+'/test_check_dir.txt')
        return True
    except:
        return False
