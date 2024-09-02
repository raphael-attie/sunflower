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

In addition, please install Jupyter Lab or Jupyter Notebook. They are required for analyzing the output interactively during the hands-on session. 
They are not necessary to run Balltracking, thus it is not present in the requirements.txt


Some Cython code is also present that requires compilation for your architecture:

- go to the *[cython_modules](https://github.com/raphael-attie/sunflower/blob/master/balltracking/balltrack.py)* directory
- If present (e.g. from a past installation) remove ``interp.c`` file
- run ``python setup_cbinterp.py build_ext --inplace`` 

# Download Example SDO/HMI data
Please download the data that will be used during the hands-on session at:
https://drive.google.com/file/d/10C0wICYds_nG2eRUXb-Gik45ez9gIUTY/view?usp=sharing
It is a ~1GB tar file. Untar it. You will see a few FITS file which are 3D data cube containing time series of 2D images and magnetograms, 
all sliced at the same time, but at different latitudes. These are SDO/HMI data remapped using Postel projection. 
Please contact me at rattie at gmu dot edu if you have trouble downloading. 


### How do I get the code running
Some generic scripts and example scripts are provided in the "scripts" directory, or during the hands-on session. 

### Who do I talk to for help? ###

Dr. Raphael Attie, contractor at NASA/Goddard Space Flight Center with George Mason University
Emails: raphael dot attie at nasa.gov, and CC to rattie at gmu.com
