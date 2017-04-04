import matplotlib
import matplotlib.pyplot as plt
import numpy as np

a = np.random.rand(256, 256)

# Colormap with 256 levels of quantization
cmap256 = matplotlib.cm.get_cmap(name='gray', lut=256)
# Same but with 10 levels
cmap10  = matplotlib.cm.get_cmap(name='gray', lut=10)

plt.figure(0)
plt.subplot(121)
plt.imshow(a, cmap=cmap256)
plt.colorbar()
plt.subplot(122)
plt.imshow(a, cmap=cmap10)
plt.colorbar()

print('done')