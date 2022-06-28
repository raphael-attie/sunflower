![picture](figures/Flow_map_paraview.png)
# README #

Sun Flower is a python framework implementing the Balltracking ang Magnetic Balltracking algorithms for tracking plasma flows at the solar surface.
These algorithms track plasma motions from intensity continuum image series (Balltracking) and series of magnetograms (Magnetic Balltracking).

### How do I get set up? ###

Necessary packages are given in requirements.txt. Check the requirements file for the exact version number. 
In particular, you will need the following packages: 

- Numpy
- Astropy
- Cython
- mahotas
- matplotlib

For people not familiar with Cython, I have provided compiled binaries for different architectures. 
However, actual compatibility with your architecture is not guaranteed, thus I recommend 
you compile the necessary balltracking binaries written with Cython as follows:

- go to the *[cython_modules](https://github.com/raphael-attie/sunflower/blob/master/balltracking/balltrack.py)* directory
- If present (e.g. from a past installation) remove ``interp.c`` file
- run ``python setup_cbinterp.py build_ext --inplace`` 

### How do I get the code running
For Balltracking, the main function is `balltrack_all` in [balltrack.py](https://github.com/raphael-attie/sunflower/blob/master/balltracking/balltrack.py). 
For Magnetic Balltracking, the main function is `track_all_frames` in [mballtrack.py](https://github.com/raphael-attie/sunflower/blob/master/balltracking/balltrack.py).

**Come back soon for self-reproducible examples.** 


### Who do I talk to for help? ###

Dr. Raphael Attie, contractor at NASA/Goddard Space Flight Center with George Mason University
Emails: raphael dot attie at nasa.gov, please CC your inquiries to attie dot raphael at gmail.com
