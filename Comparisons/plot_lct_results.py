import os, glob
import numpy as np
from scipy.io import readsav
import matplotlib.pyplot as plt


lctsaves = sorted(glob.glob('/Users/rattie/Data/SDO/HMI/karin/flctdriftstest1/flct-test1-drift*-dt9min-dx15pix.save'))

# Here we need to exclude the region that got circularly shifted by the Fourier phase shit ~ 23 pixels in both horizontal directions (left-right)
fov_slices = np.s_[23:263-23, 0:263]

vxmeans = []
for i in range(len(lctsaves)):
    idl_dict = readsav(lctsaves[i])
    vx = idl_dict['vx'].mean(axis=0)
    vxmeans.append(vx[fov_slices[1], fov_slices[0]].mean())

vxmeans = np.array(vxmeans)

## Calibration parameters
# Set npts drift rates
npts = 9
vx_rates = np.linspace(-0.2, 0.2, npts)

p = np.polyfit(vx_rates, vxmeans, 1)
a = 1 / p[0]
vxfit = a * (vxmeans - p[1])


fig = plt.figure(0)
plt.plot(vxmeans, vx_rates, 'r.', label='LCT')
plt.plot(vxmeans, vxfit, 'b-', label=r'fit $\alpha_t$ =%0.2f' %a)
plt.legend()
plt.tight_layout()
plt.show()