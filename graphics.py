import os
import numpy as np
import fitsio
from scipy.misc import bytescale, imsave

def fits_to_jpeg(file, outputdir, pmin=0.1, pmax=99.9):

    data = fitsio.read(file)
    dmin = np.percentile(data, pmin)
    dmax = np.percentile(data, pmax)
    datab = bytescale(data, cmin=dmin, cmax=dmax)

    filename = os.path.join(outputdir, os.path.splitext(os.path.basename(file))[0] + '.png')
    imsave(filename, datab)

