import numpy as np
import matplotlib
matplotlib.use('macosx')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def normal_pdf(x, mean, var):
    return np.exp(-(x - mean)**2 / (2*var))


# Generate the space in which the blobs will live
xmin, xmax, ymin, ymax = (0, 100, 0, 100)
n_bins = 100
xx = np.linspace(xmin, xmax, n_bins)
yy = np.linspace(ymin, ymax, n_bins)

# Generate the blobs. The range of the values is roughly -.0002 to .0002
means_high = [20, 50]
means_low = [50, 60]
var = [150, 200]

gauss_x_high = normal_pdf(xx, means_high[0], var[0])
gauss_y_high = normal_pdf(yy, means_high[1], var[0])

gauss_x_low = normal_pdf(xx, means_low[0], var[1])
gauss_y_low = normal_pdf(yy, means_low[1], var[1])

weights_high = np.array(np.meshgrid(gauss_x_high, gauss_y_high)).prod(0)
weights_low = -1 * np.array(np.meshgrid(gauss_x_low, gauss_y_low)).prod(0)
weights = weights_high + weights_low

# We'll also create a grey background into which the pixels will fade
greys = np.ones(weights.shape + (3,)) * 70

# First we'll plot these blobs using only ``imshow``.
vmax = np.abs(weights).max()
vmin = -vmax
cmap = plt.cm.RdYlBu

fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(weights, extent=(xmin, xmax, ymin, ymax), cmap=cmap)
ax.set_axis_off()

# Create an alpha channel of linearly increasing values moving to the right
# alphas = np.ones(weights.shape)
# alphas[:, 30:] = np.linspace(1, 0, 70)
# Create an alpha channel based on weight values
# Any value whose absolute value is > .0001 will have zero transparency [ TYPO in documentation ? mean <0.0001]
alphas = Normalize(0, .3, clip=True)(np.abs(weights))
alphas = np.clip(alphas, .4, 1) # alpha value clipped at the bottom at .4

# Normalize the colors b/w = 0/1, we'll then pass an MxNx4 array to imshow
colors = Normalize(vmin, vmax, clip=True)(weights)
colors = cmap(colors)
# Add the alpha channel
colors[...,-1] = alphas
# Create the figure and image
fig, ax = plt.subplots()
ax.imshow(greys)
ax.imshow(colors, extent=(xmin, xmax, ymin, ymax))
ax.set_axis_off
