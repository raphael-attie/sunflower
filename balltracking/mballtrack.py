import sys
import os
import numpy as np
from numpy import pi, cos, sin
import cython_modules.interp as cinterp
import fitstools
import balltracking.balltrack as blt
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter

DTYPE = np.float32
class MBT:
    def __init__(self, nt=1, rs =2, am=1, dp=0.3, td=5, zdamping=1,
                 ballspacing=10, intsteps=15, mag_thresh=30, mag_thresh_sunspots=400, noise_level=20, polarity=1,
                 track_emergence=False, emergence_box=10, datafiles=None, data=None, prep_function=None, local_min=False):

        self.datafiles = datafiles
        self.data = data
        self.nt = nt
        self.intsteps = intsteps
        self.rs = rs
        self.dp = dp
        # data prep parameters
        self.prep_function = prep_function
        self.local_min = local_min
        # Ballspacing is the minimum initial distance between the balls.
        self.ballspacing = ballspacing
        self.polarity=polarity
        # Load 1st image
        self.image = load_data(self.datafiles, 0)

        self.nx = self.image.shape[1]
        self.ny = self.image.shape[0]
        # Contrary to the Matlab implementation, and given the new way of initialization with locating the local extrema,
        # it seems more sensible to use a coarse grid with a grid size  < ballspacing.
        # A grid size of one ball diameter appears reasonable. That defines the "removal_distance"
        self.removal_distance = 2 * self.rs
        # Dimensions of the coarse grid. Used only to remove balls when they are too close to each other
        self.nxc, self.nyc = np.ceil(self.nx/self.removal_distance).astype(int), np.ceil(self.nx/self.removal_distance).astype(int) #self.coarse_grid.shape
        #self.coarse_grid = np.zeros([self.nyc, self.nxc], dtype=np.uint32)

        # Acceleration factor (used to be 0.6 in Potts implementation)
        self.am = am
        # Force scaling factor
        self.k_force = self.am / (self.dp**2 * pi * self.rs**2)
        # Damping
        self.td = td
        self.zdamping = zdamping
        self.e_td = np.exp(-1/self.td)
        self.e_tdz = np.exp(-1/self.zdamping)
        # Deepest height below surface level at which ball can fall down.
        self.min_ds = 4 * self.rs

        # Maximum number of balls that can possibly used
        self.nballs_max = self.nxc*self.nyc
        # Current position, force and velocity components, updated after each frame
        self.pos = np.full([3, self.nballs_max], -1, dtype=DTYPE)
        self.vel = np.zeros([3, self.nballs_max], dtype=DTYPE)
        self.force = np.zeros([3, self.nballs_max], dtype=DTYPE)
        # Array of the lifetime (age) of the balls
        self.balls_age = np.ones([self.nballs_max], dtype=np.uint32)
        # Storage arrays of the above, for all time steps
        self.ballpos = np.zeros([3, self.nballs_max, self.nt], dtype=DTYPE)
        self.balls_age_t = np.ones([self.nballs_max, self.nt], dtype=np.uint32)
        # Store intermediate positions, force and velocity
        self.ballpos_inter = np.zeros([3, self.nballs_max, self.intsteps])
        self.vel_inter = np.zeros([3, self.nballs_max, self.intsteps])
        self.force_inter = []

        # Ball grid and mesh. Add +1 at the np.arange stop for including right-hand side boundary
        self.ballgrid = np.arange(-self.rs, self.rs + 1, dtype=DTYPE)
        self.ball_cols, self.ball_rows = np.meshgrid(self.ballgrid, self.ballgrid)
        self.bcols = self.ball_cols.ravel()[:, np.newaxis].astype(DTYPE)
        self.brows = self.ball_rows.ravel()[:, np.newaxis].astype(DTYPE)
        self.ds    = np.zeros([self.bcols.shape[0]], dtype=DTYPE)

        # Mask of bad balls
        self.bad_balls_mask = np.zeros(self.nballs_max, dtype=bool)
        # Mask of valid balls
        self.valid_balls_mask_t = np.ones([self.nballs_max, self.nt], dtype=bool)

        # Data dependent parameters
        self.noise_level = noise_level
        self.mag_thresh = mag_thresh
        self.mag_thresh_sunspots = mag_thresh_sunspots
        self.track_emergence = track_emergence
        self.emergence_box = emergence_box

        # Initialization of ball positions
        if prep_function is not None:
            self.surface = prep_function(self.image)
            self.image = self.surface
        else:
            self.surface = prep_data(self.image)

        #self.xstart, self.ystart = get_local_extrema_ar(self.image, self.surface, self.polarity, self.ballspacing, self.mag_thresh, self.mag_thresh_sunspots, local_min=self.local_min)
        self.xstart, self.ystart = get_local_extrema(self.image, self.polarity, self.ballspacing, self.mag_thresh, local_min=self.local_min)
        self.nballs = self.xstart.size
        self.zstart = blt.put_balls_on_surface(self.surface, self.xstart, self.ystart, self.rs, self.dp)

        self.pos[0, 0:self.nballs] = self.xstart.copy()
        self.pos[1, 0:self.nballs] = self.ystart.copy()
        self.pos[2, 0:self.nballs] = self.zstart.copy()

        self.new_valid_balls_mask = np.zeros([self.nballs_max], dtype=bool)
        self.new_valid_balls_mask[0:self.nballs] = True
        self.unique_valid_balls = np.arange(self.nballs)


    def track_all_frames(self):

        for n in range(0, self.nt):

            #print('Frame n=%d'%n)

            self.image = load_data(self.datafiles, n)
            if self.prep_function is not None:
                self.surface = self.prep_function(self.image)
                self.image = self.surface
            else:
                self.surface = prep_data(self.image)

            if self.track_emergence and n > 0:
                self.populate_emergence()
            # The current position "pos" and velocity "vel" are attributes of bt.
            # They are integrated in place.
            if n==0:
                old_surface = self.surface.copy()

            for i in range(self.intsteps):

                #print('intermediate step i=%d'%i)
                #fig_title = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%d_%d.png'%(n,i)
                #plot_balls_over_frame(self.image, self.pos[0, :], self.pos[1, :], fig_title)
                # Interpolate surface
                surface_i = (old_surface*(self.intsteps - i) + self.surface * i)/self.intsteps
                blt.integrate_motion(self, surface_i)
            old_surface = self.surface.copy()
            # fig_title = '/Users/rattie/Data/SDO/HMI/EARs/AR12673_2017_09_01/mballtrack/frame_%d_%d.png'%(n,self.intsteps)
            # plot_balls_over_frame(self.image, self.pos[0, :], self.pos[1, :], fig_title)
            set_bad_balls(self, self.pos)

            # Flag the bad balls with -1
            self.pos[:, self.bad_balls_mask] = -1
            self.vel[:, self.bad_balls_mask] = np.nan
            self.balls_age[self.new_valid_balls_mask] += 1

            self.ballpos[..., n] = self.pos.copy()
            self.balls_age_t[:, n] = self.balls_age.copy()
            self.valid_balls_mask_t[:,n] = self.new_valid_balls_mask

        # Trim the array down to the actual number of balls used so far.
        # That number has been incremented each time new balls were added, in self.populate_emergence
        self.ballpos = self.ballpos[0:self.nballs, ...]
        self.balls_age_t = self.balls_age_t[0:self.nballs, :]
        self.valid_balls_mask_t = self.valid_balls_mask_t[0:self.nballs, :]



    def track_all_frames_debug(self):

        for n in range(3, self.nt-4):

            print('Frame n=%d'%n)

            self.image = load_data(self.datafiles, n)
            self.surface = prep_data(self.image)

            if self.track_emergence and n > 0:
                self.populate_emergence()

            for i in range(self.intsteps):
                blt.integrate_motion(self, self.surface)

            self.ballpos[..., n] = self.pos.copy()


    def track_start_intermediate(self):

        for i in range(self.intsteps):
            pos, vel, force = blt.integrate_motion(self, self.surface, return_copies=True)
            self.ballpos_inter[...,i] = pos
            self.vel_inter[...,i] = vel
            self.force_inter.append(force)


    def populate_emergence(self):

        #flux_posx, flux_posy = get_local_extrema_ar(self.image, self.surface, self.polarity, self.ballspacing, self.mag_thresh, self.mag_thresh_sunspots)
        flux_posx, flux_posy = get_local_extrema(self.image, self.surface, self.polarity, self.ballspacing,
                                                    self.mag_thresh, local_min=self.local_min)

        # TODO: Consider profiling this for optimization
        # Consider getting a view by using tuples of indices...
        # For now we are getting a copy
        # TODO: Consider being consistent with a coarser grid for stronger pixels, in set_bad_balls

        ball_posx, ball_posy = self.pos[0:2, self.new_valid_balls_mask]

        distance_matrix = np.sqrt((flux_posx[:,np.newaxis] - ball_posx[np.newaxis,:])**2 + (flux_posy[:,np.newaxis] - ball_posy[np.newaxis,:])**2)
        distance_min = distance_matrix.min(axis=1)
        populate_flux_mask = distance_min > self.emergence_box

        # Populate only if there's something
        if populate_flux_mask.sum() > 0 :

            newposx = flux_posx[populate_flux_mask].view('int32').copy(order='C')
            newposy = flux_posy[populate_flux_mask].view('int32').copy(order='C')

            within_edges_mask = np.logical_not(blt.get_off_edges(self, newposx, newposy))
            newposx = newposx[within_edges_mask]
            newposy = newposy[within_edges_mask]

            # Emergence detection is pixel-wise. Using interpolation in Matlab was an oversight.
            # only integer coordinates that come out of this. Interpolation is totally useless
            # I can index directly in the array.
            #newposz = self.surface[newposy, newposx]
            newposz = blt.put_balls_on_surface(self.surface, newposx, newposy, self.rs, self.dp)

            # Insert the new positions contiguously in the self.pos array
            # We need to use the number of balls at initialization (self.nballs) and increment it with the number
            # of new balls that will populate and track the emerging flux.
            self.pos[0, self.nballs:self.nballs + newposx.size] = newposx
            self.pos[1, self.nballs:self.nballs + newposx.size] = newposy
            self.pos[2, self.nballs:self.nballs + newposx.size] = newposz
            # Initialize the velocity, otherwise they could be NaN
            self.vel[:, self.nballs:self.nballs + newposx.size] = 0

            # Must add these new balls to self.new_valid_balls_mask and bad_balls_mask
            self.bad_balls_mask[self.nballs:self.nballs + newposx.size] = False
            self.new_valid_balls_mask = np.logical_not(self.bad_balls_mask)
            self.nballs += newposx.size


def mballtrack_main_positive(**kwargs):

    mbt_p = MBT(polarity=1, **kwargs)
    mbt_p.track_all_frames()

    return mbt_p


def mballtrack_main_negative(**kwargs):

    mbt_n = MBT(polarity=-1, **kwargs)
    mbt_n.track_all_frames()

    return mbt_n


def mballtrack_main(**kwargs):

    mbt_p = MBT(polarity=1, **kwargs)
    mbt_n = MBT(polarity=-1, **kwargs)
    mbt_p.track_all_frames()
    mbt_n.track_all_frames()

    return mbt_p, mbt_n


def load_data(datafiles, n):
    _, ext = os.path.splitext(datafiles[0])
    if ext == '.fits':
        image = load_fits(datafiles, n)
        return image
    elif ext == '.npz':
        image = load_npz(datafiles, n)
        return image
    else:
        sys.exit("invalid file extension. Must be either .fits or .npz")


def load_npz(datafiles, n):
    data = np.load(datafiles[n])
    image = data[data.files[0]]
    return image


def load_fits(datafiles, n):
    #image = fitstools.fitsread(mbt.datafiles, tslice=n).astype(DTYPE)
    image = fitstools.fitsread(datafiles, tslice=slice(n,n+1)).astype(DTYPE)
    #image = np.median(image, 2)
    return image


def get_local_extrema(image, polarity, min_distance, threshold, local_min=False):
    """
    Default to finding only local maxima. local_min = True will look only for local minima

    :param image: 2D frame displaying the features to track.
    :param polarity: if data signed (e.g magnetograms), set which polarity is tracked
    :param min_distance: minimum distance to search between local extrema
    :param threshold: values setting the limit for searching for local extrema. Can be a signed value or range of 2 values
    :param local_min: if True, will look for local minima instead of local maxima
    :return: list of x- and y- coordinates of the local extrema
    """

    # Get a mask of where to look for local maxima.
    if len(threshold) == 1:
        if polarity >= 0:
            mask_thresh = image >= threshold
        else:
            mask_thresh = image <= -threshold
    else:
            mask_thresh = (image > min(threshold)) & (image < max(threshold))


    # Look for local maxima, get a list of coordinates. Use a geater grid size for sunspot.
    # Outside sunspot
    #surface_sm = -gaussian_filter(surface, sigma=3)
    #surface_sm2 = surface_sm + np.abs(surface_sm.min())

    if local_min:
        # reverse the scale of the image so the local min are searched as local max
        image2 = image.max() - image
        ystart, xstart = np.array(
        peak_local_max(np.abs(image2), indices=True, min_distance=min_distance, labels=mask_thresh)).T
    else:
        #se = disk(round(min_distance/2))
        #ystart, xstart = np.array( peak_local_max(np.abs(image), indices=True, footprint=se,labels=mask_maxi)).T
        ystart, xstart = np.array(peak_local_max(np.abs(image), indices=True, min_distance=min_distance, labels=mask_thresh)).T

    # Because transpose only creates a view, and this is eventually given to a C function, it needs to be copied as C-ordered
    return xstart.copy(order='C'), ystart.copy(order='C')


def get_local_extrema_ar(image, polarity, min_distance, threshold, threshold2, local_min=False):
    """
    Find the coordinates of local extrema with special treatment of Active Regions.
    Similar to get_local_extrema() but uses a grid size (min_distance) 3x greater
    in regions that exceeds a higher threshold.

    :param image: typically a magnetogram. Can be anything whose larger region have a pixels of higher intensity.
    :param polarity: 0,+1 for positive flux or intensity. -1 for negative flux or intensity
    :param min_distance: minimum distance to consider between local extrema.
    :param threshold: pixels below this value are ignored
    :param threshold2: values that define the regions of higher intensity.
    :return: arrays of coordinates of local extrema
    """

    xstart, ystart = get_local_extrema(image, polarity, min_distance, threshold, local_min=local_min)
    # Get the intensity at these locations
    data_int = image[ystart, xstart]
    # Build a distance-based matrix for coordinates of pixel whose intensity is above threshold2, and keep the maximum
    if polarity >= 0:
        select_mask = np.logical_and(data_int >=0, data_int < threshold2)
        mask_maxi_sunspots = image >= threshold2
    else:
        select_mask = np.logical_and(data_int < 0, data_int > -threshold2)
        mask_maxi_sunspots = image < - threshold2

    xstart1, ystart1 = xstart[select_mask], ystart[select_mask]

    se = disk(round(min_distance/2))
    #se = np.ones([3*min_distance, 3*min_distance])

    # ystart2, xstart2 = np.array(peak_local_max(np.abs(image), indices=True,
    #                                            footprint= se,
    #                                            labels=mask_maxi_sunspots), dtype=DTYPE).T

    ystart2, xstart2 = np.array(peak_local_max(np.abs(image), indices=True,
                                               min_distance=min_distance,
                                               labels=mask_maxi_sunspots), dtype=DTYPE).T

    xstart = np.concatenate((xstart1, xstart2))
    ystart = np.concatenate((ystart1, ystart2))

    return xstart, ystart

# def get_local_extrema_ar2(image, polarity, threshold):
#
#
#     data = image.astype(np.float64)
#     #TODO: Check if this is not overkill; since each value is compared against the threshold, this might not be needed
#     if polarity >= 0:
#         signed_data = np.ma.masked_less(data, 0).filled(0)
#     else:
#         signed_data = np.ma.masked_less(-data, 0).filled(0)
#
#     labels = segmentation.detect_polarity(signed_data, float(threshold))
#
#     return labels


def prep_data(image):

    image2 = np.sqrt(np.abs(image))
    image3 = image2.max() - image2
    surface = (image3 - image3.mean())/image3.std()
    return surface.copy(order='C').astype(DTYPE)



def set_bad_balls(bt, pos, check_polarity=True, check_noise=True, check_sinking=True):
    # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
    # and https://stackoverflow.com/questions/36863404/accumulate-constant-value-in-numpy-array
    #xpos0, ypos0, zpos0 = bt.pos

    # Bad balls are flagged with -1 in the pos array. They will be excluded from the comparison below:

    # It is important to first get rid of the off-edge ones so we can use direct coordinate look-up instead of
    # interpolating the values, which would be troublesome with off-edge coordinates.
    off_edges_mask = blt.get_off_edges(bt, pos[0,:], pos[1,:])
    # Ignore these bad balls in the arrays and enforce continuity principle
    valid_balls_mask = np.logical_not(off_edges_mask)
    valid_balls_idx = np.nonzero(valid_balls_mask)[0]
    pos2 = pos[:, valid_balls_idx].astype(np.int32)
    # Initialize new mask for checking for noise tracking, polarity crossing and sinking balls
    valid_balls_mask2 = np.ones([valid_balls_idx.size], dtype=bool)
    # Forbid crossing flux of opposite polarity and tracking below noise level.

    # The block below checks for polarity crossing, balls tracking in the noise, and sinking balls.
    # It must happens first because the coarse-grid-based decimation that comes next is more expensive
    # with more balls that have no point being in there anyway.
    if check_polarity:
        same_polarity_mask = np.sign(bt.image[pos2[1, :], pos2[0, :]]) * bt.polarity >=0
        valid_balls_mask2 = np.logical_and(valid_balls_mask2, same_polarity_mask)

    if check_noise:
        # Track only above noise level. Balls below that noise level are considered "sinking".
        # Use absolute values to make it independent of the polarity that's being tracked.
        noise_mask = np.abs(bt.image[pos2[1, :], pos2[0, :]]) > bt.noise_level
        valid_balls_mask2 = np.logical_and(valid_balls_mask2, noise_mask)

    if check_sinking:
        # This assumes the vertical position have already been set
        # That is not the case when decimating the balls at the initialization state
        # Thus this check should be set to false during initialization
        unsunk_mask = pos2[2,:] > bt.surface[pos2[1, :], pos2[0, :]] - bt.min_ds
        valid_balls_mask2 = np.logical_and(valid_balls_mask2, unsunk_mask)

    # Get indices in the original array. Remember that valid_balls_mask2 has the same size as pos2
    valid_balls_idx = valid_balls_idx[valid_balls_mask2]
    # Get the valid balls from the input pos array.
    # indexing scheme below returns a copy, just like with boolean index arrays.
    xpos, ypos = pos[0:2, valid_balls_idx]
    balls_age = bt.balls_age[valid_balls_idx]

    ## Decimation based on the coarse grid.
    # Get the 1D position on the coarse grid, clipped to the edges of that grid.
    _, _, coarse_pos = coarse_grid_pos(bt, xpos, ypos)

    # Get ball number and balls age sorted by position, sort positions too, and array of valid balls indices as well!!!
    sorted_balls = np.argsort(coarse_pos)
    balls_age = balls_age[sorted_balls]
    coarse_pos = coarse_pos[sorted_balls]
    valid_balls_idx = valid_balls_idx[sorted_balls]
    # There can be repetitions in the coarse_pos because there can be more than one ball per finegrid cell.
    # The point is to keep only one ball per coarse grid point: the oldest.
    # So we need to sort coarse_pos further using the balls age as weight and extract a unique set where each ball is the oldest
    sidx = np.lexsort([balls_age, coarse_pos])
    # Indices of the valid balls to keep
    bt.unique_valid_balls = valid_balls_idx[ sidx[ np.r_[np.flatnonzero(coarse_pos[1:] != coarse_pos[:-1]), -1] ] ]

    # Now the point is to have a mask or list of balls at overpopulated cells.
    # They are simply the ones not listed by unique_oldest_balls
    bt.bad_balls_mask = np.ones([pos.shape[1]], dtype=bool)
    bt.bad_balls_mask[bt.unique_valid_balls] = False
    # Update mask of valid balls and increment the age of valid balls only
    bt.new_valid_balls_mask = np.logical_not(bt.bad_balls_mask)
    return

def coarse_grid_pos(mbt, x, y):

    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.uint32(np.clip(np.floor(x / mbt.removal_distance), 0, mbt.nxc-1))
    ycoarse = np.uint32(np.clip(np.floor(y / mbt.removal_distance), 0, mbt.nyc-1))
    # Convert to linear (1D) indices. One index per ball
    #coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), mbt.coarse_grid.shape)
    coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), (mbt.nyc, mbt.nxc))
    return xcoarse, ycoarse, coarse_idx


def merge_positive_negative_tracking(mbt_p, mbt_n):

    # Get a view that gets rid of the z-coordinate
    pos_p = mbt_p.ballpos[slice(0,1), ...]
    pos_n = mbt_n.ballpos[slice(0,1), ...]
    # Merge
    pos = np.concatenate((pos_p, pos_n), axis=1)
    return pos


def get_balls_at(x, y, xpos, ypos, tolerance=0.2):

    return np.where((np.abs(xpos - x) < tolerance) & (np.abs(ypos - y) < tolerance))[0]


def label_from_pos(x, y, dims):

    label_map = np.zeros(dims, dtype= np.int32)
    labels = np.arange(x.size, dtype = np.int32)+1
    # This assumes bad balls are flagged with coordinate value of -1 in x (and y)
    valid_mask = x > 0
    label_map[y[valid_mask], x[valid_mask]] = labels[valid_mask]

    return label_map

def marker_watershed(data, x, y, threshold, polarity, invert=True):

    markers = label_from_pos(x, y, data.shape)
    if polarity >=0:
        mask_ws = data > threshold
    else:
        mask_ws = data < -threshold

    wdata = np.abs(data)
    # For magnetograms, need to invert the absolute value so the fragment intensity decreases toward centroid
    if invert:
        wdata -= wdata

    labels = watershed(wdata, markers, mask=mask_ws)
    borders = find_boundaries(labels)
    # Subtract 1 to align with the ball number series. E.g: watershed label 0 corresponds to ball #0
    labels -=1
    return labels, markers, borders

def watershed_series(datafile, nframes, threshold, polarity, ballpos, verbose=False, prep_function=None, invert=True):

    # Load a sample to determine shape
    #data = fitstools.fitsread(datafile, tslice=0)
    data = load_data(datafile, 0)
    if prep_function is not None:
        data = prep_function(data)

    ws_series = np.empty([nframes-7, data.shape[1], data.shape[0]], dtype=np.int32)
    markers_series = np.empty([nframes-7, data.shape[1], data.shape[0]], dtype=np.int32)
    borders_series = np.empty([nframes-7, data.shape[1], data.shape[0]], dtype=np.bool)

    # For parallelization, need to see how to share a proper container, whatever is more efficient
    for n in range(nframes-7):
        if verbose:
            print('Watershed series frame n = %d'%n)
        #data = fitstools.fitsread(datafile, tslice=n)
        data = load_data(datafile, n)
        # Get a view of (x,y) coords at frame #i (use slice instead of fancy insteading). Either with slice(0,1) or 0:2
        # I'll use slice for clarity
        # positions = ballpos[slice(0,1),:,n]
        labels_ws, markers, borders = marker_watershed(data, ballpos[0,:,n], ballpos[1,:,n], threshold, polarity, invert=invert)
        ws_series[n,...] = labels_ws
        markers_series[n,...] = markers
        borders_series[n,...] = borders

    return ws_series, markers_series, borders_series

def merge_watershed(labels_p, borders_p, nballs_p, labels_n, borders_n):
    """
    Merge the results from markers-watershed the positive and negative flux.
    The output borders array for positive flux stay at +1, but borders of negative values are set at -1.

    :param labels_p: Array of watershed labels for positive flux
    :param borders_p: Array of watershed borders for positive flux
    :param labels_n: Array of watershed labels for negative flux
    :param borders_n: Array of watershed borders for negative flux
    :return: Array of same shape as input. +1 on borders of positive flux, -1 for negative flux
    """

    ws_labels = labels_p.copy()
    ws_labels[labels_n >= 0] = nballs_p + labels_n[labels_n >= 0] + 1
    borders = borders_p.copy().astype(np.int8)
    borders[borders_n == 1] = -1

    return ws_labels, borders



def plot_balls_over_frame(frame, x, y, fig_title):
    plt.figure(0, figsize=(11.5, 9))
    plt.imshow(frame, vmin=-100, vmax=100, cmap='gray')
    plt.plot(x, y, ls='none', marker='+', color='green', markerfacecolor='none', ms=2)
    plt.axis([0, frame.shape[1], 0, frame.shape[0]])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_title)
    plt.close()

