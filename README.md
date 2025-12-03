![picture](figures/Flow_map_paraview.png)
# README #

Sun Flower is a python framework implementing the Balltracking ang Magnetic Balltracking algorithms for tracking plasma flows at the solar surface.
These algorithms track plasma motions from intensity continuum image series (Balltracking) and series of magnetograms (Magnetic Balltracking).

This code is for research purposes. It is not a final product. As such, it evolves based on latest research findings. 

The use of this code implies that you have read the relevant publications detailing the Balltracking and Magnetic Balltracking algorithms. 
E.g.:
- [Attie et al. 2018](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018SW001939)
- [Attie et al. 2016](https://www.aanda.org/articles/aa/full_html/2016/12/aa27798-15/aa27798-15.html)
- [Attie et al. 2015](https://www.aanda.org/articles/aa/full_html/2015/02/aa24552-14/aa24552-14.html)


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
- `run_balltrack_template.py` is a generic wrapper for HMI data that you would adapt to your specific HMI dataset, the only one what was comprehensively tested with an
optimized set of input parameters that maximizes tracking precision. It works with `inputs.py` that contains the optimized parameter set. 
- Other wrappers named `run_balltrack_*` are provided for examplifying other possible uses cases, but using any dataset different than 
full-resolution HMI data remains untested and therefore, not guaranteed to give best results. 
- An effort is ongoing with an ISSI group to provide more use cases with optimized input tracking parameters for other data. The outcome of that
effort will be shared publically. 

### Who do I talk to for help? ###

Dr. Raphael Attie, contractor at NASA/Goddard Space Flight Center with George Mason University
Emails: raphael dot attie at nasa.gov, and CC to rattie at gmu.edu
