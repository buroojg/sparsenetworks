This is a small framework for simulation and analysis of sparse neural networks.  
For lack of a better name we jokingly sometimes refer to this module as BFID model, for our initials 

* **B**urooj,

* **F**rancesca,

* **I**lyas,

* **D**iemut.

### To install ...

... copy/clone this repository's contents, navigate to the new directory (to the level where the file `setup.py` is), and run the command:
```
$ sudo python setup.py install
```  

You can now import the `sparsenetworks` module in python files.
```python
from sparsenetworks import system
# ...
```

### Example

Have a look at the script `sn_example_simulation.py`. You can run it from command line, it should have been added as executable to your path upon install.  
```
$ sn_example_simulation.py
```
You should also have a look at that file's content, to see how to import and use this module.  
After you ran the example simulation, use the plot scripts in folder `scripts` to plot the generated data. They are designed to be run from the ipython command line environment (otherwise the plots show only for a tiny moment).  
ADD EXAMPLE HOW TO USE PLOT FILES!