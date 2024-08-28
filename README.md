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
- Matplotlib
- Pandas
- Scipy
- Setuptools
- Scikit-image

For maximum compatibility, I recommend to use the requirements.txt file that shows at what version that code is tested. 

Some Cython code is also present that requires compilation for your architecture:

- go to the *[cython_modules](https://github.com/raphael-attie/sunflower/blob/master/balltracking/balltrack.py)* directory
- If present (e.g. from a past installation) remove ``interp.c`` file
- run ``python setup_cbinterp.py build_ext --inplace`` 

### How do I get the code running
Some generic scripts and example scripts are provided in the "scripts" directory. 

### Who do I talk to for help? ###

Dr. Raphael Attie, contractor at NASA/Goddard Space Flight Center with George Mason University
Emails: raphael dot attie at nasa.gov, and CC to rattie at gmu.com
