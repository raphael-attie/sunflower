import os, glob
import numpy as np
import matplotlib
matplotlib.use('TKagg')
import matplotlib.pyplot as plt
from scipy.io import readsav


datadir = '/Users/rattie/Data/Ben/SteinSDO/calibration/sanity_check_shifted_images'
lctfile = os.path.join(datadir, 'vxvy-drift-test-0.2step.save')
idl_dict = readsav(lctfile)
vx = idl_dict['vxms']
print(vx[1,:])

drifts = np.arange(0.1, 2.1, 0.1)

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(1,1, figsize = (8,8))
ax.plot(drifts, drifts, 'k-', label='reference 1:1 fit')
ax.plot(drifts, vx[1,:], 'r+', label = 'FLCT shifts (1)')
ax.grid(True, axis='both')
plt.xlabel('true shift [px]')
plt.ylabel('FLCT shift [px]')
plt.legend()
plt.tight_layout()
plt.show()

plt.savefig(os.path.join(datadir, 'lct_single_image_shifted.png'))


