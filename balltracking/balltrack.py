""" This module hosts all the necessary functions to run balltracking.
A main program should execute "balltrack_all()".
"""
import sys
import os
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
from numpy import pi, cos, sin
import csv
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from scipy.signal import convolve2d
from cython_modules import interp as cinterp
import filters
import fitstools

DTYPE = np.float32


class BT:

    def __init__(self, rs, dp, ballspacing, intsteps, sigma_factor, fourier_radius, nt, mode, direction, datafiles=None, data=None,
                 output_prep_data=False, outputdir=None, verbose=False):

        """ This is the class hosting all the parameters and intermediate results of balltracking.

        Attributes:

            rs (int): balls radius in pixels
            dp (float): Characteristic percentage depth. 0 < dp < 1
            ballspacing (int): nb of pixels between balls center at the initialization stage
            intsteps (int): nb of intermediate frames using linear interpolation.
            sigma_factor (float): multiplier to the standard deviation
            fourier_radius (float): radius for high-pass image fourier filter,
                                    to be given in period units (pixels) instead of k-space frequency.
            nt (int): nb of frames to track
            mode (str): determines which side of the data surface to track: 'top' or 'bottom'.
            direction (str): determines whether we track 'forward' or 'backward' in time.
            datafiles (list): list of FITS file paths
            data (ndarray): instead of providing a list of files with datafiles, one can directly provide a 3D array
            output_prep_data (str): whether to write out the intermediary surface data. For sanity check.
            outputdir (str): Save the arrays of tracked position to that directory

        """

        if output_prep_data & (outputdir is None):
            sys.exit('Missing output directory to output prep data')

        self.datafiles = datafiles
        self.data = data
        if direction != 'forward' and direction != 'backward':
            raise ValueError

        self.direction = direction
        self.nt = int(nt)

        # Get a sample. 1st of the series in forward direction. last of the series in backward direction.
        # TODO: It could be wiser to get some average value between 1st, middle, and last of the series?
        if self.data is None:
            if self.direction == 'forward':
                self.sample = fitstools.fitsread(self.datafiles, tslice=0).astype(np.float32)
            else:
                self.sample = fitstools.fitsread(self.datafiles, tslice=self.nt - 1).astype(np.float32)
        else:
            if self.direction == 'forward':
                self.sample = self.data[:, :, 0]
            else:
                self.sample = self.data[:, :, -1]

        self.nx = int(self.sample.shape[1])
        self.ny = int(self.sample.shape[0])

        self.intsteps = int(intsteps)
        self.rs = rs
        self.dp = dp
        self.ballspacing = int(ballspacing)
        # Number of balls in a row
        self.nballs_row = int((self.nx - 4 * self.rs) / self.ballspacing + 1)
        # Number of balls in a column
        self.nballs_col = int((self.ny - 4 * self.rs) / self.ballspacing + 1)
        # Total number of balls
        self.nballs = self.nballs_row * self.nballs_col
        # Image coordinates
        self.xcoords = np.arange(self.nx)
        self.ycoords = np.arange(self.ny)
        # Image mesh
        self.meshx, self.meshy = np.meshgrid(self.xcoords, self.ycoords)
        # Initialize ball positions
        self.xstart, self.ystart = initialize_mesh(ballspacing, self.nx, self.ny)
        self.zstart = np.zeros(self.xstart.shape, dtype=np.float32)
        self.nballs = self.xstart.size
        # Initialize the array of the lifetime (age) of the balls
        self.balls_age = np.ones([self.nballs], dtype=np.uint32)

        # Coarse grid
        # Matlab: finegrid=zeros(ceil(BT.nr/BT.finegridspacing),ceil(BT.nc/BT.finegridspacing));
        self.coarse_grid = np.zeros([np.ceil(self.ny/self.ballspacing).astype(int), np.ceil(self.nx/self.ballspacing).astype(int)], dtype=np.uint32)
        # Dimensions of the coarse grid
        self.nyc_, self.nxc_ = self.coarse_grid.shape

        # Acceleration factor (used to be 0.6 in Potts implementation)
        self.am = 1.0
        # Force scaling factor
        self.k_force = self.am / (self.dp**2 * pi * self.rs**2)
        # Damping
        self.td = 1.0
        self.tdx = self.td
        self.tdy = self.td
        self.zdamping = 0.3
        self.e_td_ = np.exp(-1 / self.td)
        self.e_tdx_ = self.e_td_
        self.e_tdy_ = self.e_td_
        self.e_tdz_ = np.exp(-1 / self.zdamping)

        # Rescaling factor for the standard deviation
        self.sigma_factor = sigma_factor
        self.mean = 0
        self.sigma = 0
        self.fourier_radius = fourier_radius

        # Current position, force and velocity components, updated after each frame
        self.pos = np.zeros([3, self.nballs], dtype=DTYPE)
        self.vel = np.zeros([3, self.nballs], dtype=DTYPE)
        self.force = np.zeros([3, self.nballs], dtype=DTYPE)
        self.balls_age_t = np.zeros([self.nballs, self.nt], dtype=np.uint32)
        # Storage arrays of the above, for all time steps
        self.ballpos = np.zeros([3, self.nballs, self.nt], dtype=DTYPE)
        self.ballvel = np.zeros([3, self.nballs, self.nt], dtype=DTYPE)

        # Ball grid and mesh. Add +1 at the np.arange stop for including right-hand side boundary
        self.ballgrid = np.arange(-self.rs, self.rs + 1, dtype=DTYPE)
        self.ball_cols, self.ball_rows = np.meshgrid(self.ballgrid, self.ballgrid)
        self.bcols = self.ball_cols.ravel()[:, np.newaxis].astype(DTYPE)
        self.brows = self.ball_rows.ravel()[:, np.newaxis].astype(DTYPE)
        # Not in use for now
        self.ds = np.zeros([self.bcols.shape[0]], dtype=DTYPE)
        # Hold the current surfance
        self.surface = np.zeros(self.sample.shape)
        # Initialize deepest height at which ball can fall down. Typically it will be set to a multiple of -surface.std().
        self.min_ds = -5
        # Mask of bad balls
        self.bad_balls_mask = np.zeros(self.nballs, dtype=bool)
        # Mask of valid balls
        self.new_valid_balls_mask = np.ones(self.nballs, dtype=bool)

        # Mode and direction
        self.mode = mode
        self.direction = direction

        # Optional parameter for writing the "prepped" data surface in fits files.
        self.output_prep_data = output_prep_data
        self.outputdir = outputdir
        self.verbose = verbose

    def coarse_grid_pos(self, x, y):
        """ x, y coordinates in the image into coarse grid coordinates"""

        # Get the position on the coarse grid, clipped to the edges of that grid.
        xcoarse = np.uint32(np.clip(np.floor(x / self.ballspacing), 0, self.nxc_ - 1))
        ycoarse = np.uint32(np.clip(np.floor(y / self.ballspacing), 0, self.nyc_ - 1))
        # Convert to linear (1D) indices. One index per ball
        coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), self.coarse_grid.shape)
        return xcoarse, ycoarse, coarse_idx

    def initialize(self):
        """Calculate offset (mean) and standard deviation from  a valid surface

        First, filter image to focus on the granulation
        Get mean and standard deviation from the masked array, not affected by bad invalid values (sunspot, ...)
        Generate the data surface from the image with the masked mean and sigma"""

        self.surface, mean, sigma = prep_data2(self.sample, self.sigma_factor)
        self.mean = mean
        self.sigma = sigma
        # Initialize the height of the ball. Only possible if the data surface is given.
        self.pos[0, :] = self.xstart.flatten()
        self.pos[1, :] = self.ystart.flatten()
        self.zstart = put_balls_on_surface(self.surface, self.xstart.ravel(), self.ystart.ravel(), self.rs, self.dp)
        self.pos[2, :] = self.zstart.copy()
        # Setup the coarse grid: populate edges to avoid "leaks" of balls ~ balls falling off.
        # Although can't remember if that was actually ever used in the Matlab implementation
        self.coarse_grid[0,:] = 1
        self.coarse_grid[:, 0] = 1
        self.coarse_grid[-1,:] = 1
        self.coarse_grid[:, -1] = 1
        return

    def track(self):
        """Tracking outer loop

        Loop over the images and update the balls positions
        """

        # Outer loop goes over the data frames.
        # If data is a fits cube, we just access a slice of it

        # Initialize
        self.initialize()

        for n in range(0, self.nt):
            if self.verbose:
                print("Tracking direction {}/{}, frame {:d}".format(self.direction, self.mode, n))

            if self.direction == 'forward':
                if self.data is None:
                    image = fitstools.fitsread(self.datafiles, tslice=n).astype(np.float32)
                else:
                    image = self.data[:, :, n]
            else:
                if self.data is None:
                    image = fitstools.fitsread(self.datafiles, tslice=self.nt - 1 - n).astype(np.float32)
                else:
                    image = self.data[:, :, self.nt - 1 - n]

            # TODO: check the choice of prep_data regarding mean normalization with fixed mean or time-dependent one
            # self.surface = prep_data(image, self.mean, self.sigma, sigma_factor=self.sigma_factor)
            self.surface, _, _ = prep_data2(image, sigma_factor=self.sigma_factor, pixel_radius=self.fourier_radius)
            if self.mode == 'bottom':
                self.surface = -self.surface

            if self.output_prep_data:
                filename_surface = os.path.join(self.outputdir, 'prep_data_{}_{}_{:05d}.fits'.format(self.direction, self.mode, n))
                fitstools.writefits(self.surface, filename_surface)

            # The current position "pos" and velocity "vel" are attributes of bt.
            # They are integrated in place.
            for _ in range(self.intsteps):
                integrate_motion(self, self.surface)

            self.balls_age += 1

            # Get the bad balls at the current frame
            self.get_bad_balls()

            # TODO: check the modifications below where I flag the bad balls before saving the array and relocating the bad balls
            # Flag the bad balls with -1
            self.pos[:, self.bad_balls_mask] = -1
            self.vel[:, self.bad_balls_mask] = np.nan

            # Add the current position array to the time series of position
            self.ballpos[..., n] = self.pos.copy()
            self.ballvel[..., n] = self.vel.copy()

            # The bad balls are assigned new positions (relocated to empty cells)
            # This is done in place in bt.pos. For flexibility with debugging, the array
            # of position is given explicitly as input
            _, _ = self.replace_bad_balls(self.surface)

            self.balls_age_t[:, n] = self.balls_age.copy()

        if self.direction == 'backward':
            # Flip timeline for backward tracking
            self.ballpos = np.flip(self.ballpos, 2)
            self.ballvel = np.flip(self.ballvel, 2)

    def get_bad_balls(self):
        """
        Mask out all the balls that are outside the tracking requirements.

        Returns
        -------

        """
        # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
        # and https://stackoverflow.com/questions/36863404/accumulate-constant-value-in-numpy-array
        # xpos0, ypos0, zpos0 = bt.pos

        # Bad balls are flagged with -1 in the pos array. They will be excluded from the comparisons below
        bad_balls1_mask = self.get_outliers_mask()

        # Ignore these bad balls in the arrays and enforce continuity principle
        valid_balls = np.logical_not(bad_balls1_mask)
        valid_balls_idx = np.nonzero(valid_balls)[0]
        # valid_balls = np.logical_not(bad_balls1_mask)
        # Map back to original balls indices
        # valid_balls_idx = np.nonzero(valid_balls)[0]

        xpos, ypos, zpos = self.pos[:, valid_balls]
        balls_age = self.balls_age[valid_balls]

        # Get the 1D position on the coarse grid, clipped to the edges of that grid.
        _, _, coarse_pos = self.coarse_grid_pos(xpos, ypos)

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
        unique_oldest_balls = valid_balls_idx[sidx[np.r_[np.flatnonzero(coarse_pos[1:] != coarse_pos[:-1]), -1]]]

        # Now the point is to have a mask or list of balls at overpopulated cells.
        # They are simply the ones not listed by unique_oldest_balls
        bad_full_balls = np.ones([self.nballs], dtype=bool)
        bad_full_balls[unique_oldest_balls] = False
        # The bad balls are not just the ones in overpopulated cells, there's also the ones off edges & sunk balls
        # bad_balls_mask = np.logical_or(bad_balls1, bad_full_balls)
        self.bad_balls_mask = np.logical_or(bad_balls1_mask, bad_full_balls)

        return

    def get_outliers_mask(self):
        """
        Create the mask of balls that are falling off the 3D surface.

        Returns
        -------

        """
        x, y, z = self.pos
        sunk = z < self.min_ds_
        off_edges_mask = get_off_edges_mask(self.rs, self.nx, self.ny, x, y)
        outliers_mask = np.logical_or(sunk, off_edges_mask)

        return outliers_mask

    def replace_bad_balls(self, surface):
        """
        Repopulate the data surface with a new set of balls whose number is consistent with the masked out balls.

        Parameters
        ----------
        surface : data surface to be repopulated

        Returns
        -------
        Array of x,y coordinates of the new balls
        """
        nbadballs = self.bad_balls_mask.sum()
        # Get the mask of the valid balls that we are going to keep
        valid_balls_mask = np.logical_not(self.bad_balls_mask)
        # Work more explicitly with views on coordinate and velocity arrays of valid balls for clarity
        # (instead of working with pos[:, :, valid_balls_mask] directly)
        xpos, ypos, zpos = self.pos[:, valid_balls_mask]
        # Get the 1D position on the coarse grid, clipped to the edges of that grid.
        _, _, coarse_pos_idx = self.coarse_grid_pos(xpos, ypos)
        # Set these positions on the coarse grid as filled. Remember that to avoid putting new balls on the edge,
        # the coarse_grid is pre-filled with ones at its edges
        # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
        coarse_grid = self.coarse_grid.copy()
        coarse_grid.ravel()[coarse_pos_idx] = 1
        y0_empty, x0_empty = np.where(coarse_grid == 0)
        nemptycells = x0_empty.size
        # If there are more empty cells than there are bad balls to relocate, we only populate a maximum of nbadballs.
        # That situation does not happen with appropriate choice of ball parameters.
        if nemptycells > nbadballs:
            # TODO: consider a random shuffle of the array prior to selecting a subset
            x0_empty = x0_empty[0:nbadballs]
            y0_empty = y0_empty[0:nbadballs]
        # if nemptycells <= nbadballs:
        # IMPROVEMENT with respect to Matlab version:
        # y0, x0 above are from np.where => integer values! All values in xnew, ynew are integers.
        # => There is no need to interpolate in put_ball_on_surface!
        xnew = self.ballspacing * x0_empty.astype(np.float32)#+ bt.rs
        ynew = self.ballspacing * y0_empty.astype(np.float32)#+ bt.rs
        znew = put_balls_on_surface(surface, xnew, ynew, self.rs, self.dp)
        # Get the indices of bad balls in order to assign them to new positions. If nemptycells > nbadballs,
        # this array indexing automatically truncates indexing limit to the size of bad_balls_mask.
        bad_balls_idx = np.nonzero(self.bad_balls_mask)[0][0:nemptycells]

        # Relocate them in the empty cells. Only take as many bad balls as we have empty cells.

        self.pos[0, bad_balls_idx] = xnew
        self.pos[1, bad_balls_idx] = ynew
        self.pos[2, bad_balls_idx] = znew
        # Reset the velocity and age at these new positions
        self.vel[:, bad_balls_idx] = 0
        self.balls_age[bad_balls_idx] = 0

        # Rest of bad balls. If nemptycells > nbadballs, this gives an empty array.
        bad_balls_remaining_idx = np.nonzero(self.bad_balls_mask)[0][nemptycells:nbadballs]

        # Update the list of valid balls in bt.new_valid_balls_mask
        new_valid_balls_mask = np.ones([self.nballs], dtype = bool)
        new_valid_balls_mask[bad_balls_remaining_idx] = False
        self.new_valid_balls_mask = new_valid_balls_mask

        # else:
        #     # This case means the number of bad balls available for relocation is smaller than the number of
        #     empty cells where they can be relocated.
        #     # This means the continuity principle is not satisfied and needs investigation.
        #     raise SystemExit('The number of empy cells is greater than the number of bad balls.')

        return xnew, ynew


def initialize_mesh(ballspacing, nx, ny):
    """
    Initial horizontal (x,y) positions into a regular, cartesian grid

    Args:
        ballspacing (int): Initial spacing between the balls
        nx (int): size of image horizontal axis in pixels
        ny (int): size of image vertical axis in pixels

    Returns:
        x (horizontal) and y (vertical) coordinates of initial positions

    """

    x_start_points = np.arange(ballspacing, nx - ballspacing + 1, ballspacing, dtype=DTYPE)
    y_start_points = np.arange(ballspacing, ny - ballspacing + 1, ballspacing, dtype=DTYPE)
    xstart, ystart = np.meshgrid(x_start_points, y_start_points)
    return xstart, ystart


def track_instance(params, side_direction, datafiles=None, data=None):
    """
    Wrapper to run balltracking on a given tuple of (side, direction).
    This routine must be executed with 4 of these pairs for minimizing the random error.
    Either the list of files or numpy array of images must be provided.

    Args:
        params (dict): ball parameters for the BT class
        side_direction (tuple): (BT.mode) and (BT.direction) listed as ('top', 'forward') or ('bottom', 'backward')
        datafiles (list): path to data cube file or list of FITS files. Ignored if `data` is provided.
        data (ndarray): Series of 3D Numpy arrays with time on the 3rd index: (y-axis, x-axis, time-axis).

    Returns:
        BT.ballpos : Array storing the positions of the balls at each frame
    """

    print(side_direction)

    params_side = params[side_direction[0]]

    bt_instance = BT(params['nframes'], params_side['rs'], params_side['ballspacing'], params_side['dp'],
                     params_side['intsteps'], params_side['sigma_factor'], params_side['fourier_radius'],
                     side_direction[0], side_direction[1],
                     output_prep_data=params['output_prep_data'],
                     outputdir=params['outputdir'],
                     verbose=params['verbose'],
                     datafiles=datafiles, data=data)

    bt_instance.track()

    return bt_instance.ballpos


def balltrack_all(params, datafiles=None, data=None, write_ballpos=True, ncores=1):
    """
    Run the tracking on the 4 (mode, direction) pairs:
    (('top', 'forward'),
     ('top', 'backward'),
     ('bottom', 'forward'),
     ('bottom', 'backward'))

     Can be executed in a pool of up to 4 parallel workers.

    Args:
        params (dict): all ball parameters relevant to the BT class
        datafiles (str or list): path to data cube file or list of FITS files
        data (ndarray): Series of 3D Numpy arrays with time on the 3rd index: (y-axis, x-axis, time-axis)
        write_ballpos (bool): sets whether to write the array of ball positions
        ncores: number of cores to use for running the 4 modes/directions in parallel
            Default is 1 for sequential processing. There can up to 4 workers for these parallel tasks (2-3x speed-up)

    Returns:
        ballpos_top : 3D array of ball positions for top-side tracking -> [xyz, ball #, time]
        ballpos_bottom : 3D array of ball positions for bottom-side tracking -> [xyz, ball #, time]

    """
    if write_ballpos and 'outputdir' not in params:
        sys.exit('missing outputdir in params to write ball positions')

    # Check user data input
    if (datafiles is None or not isinstance(datafiles, str)) and not isinstance(datafiles, list) and data is None:
        raise ValueError
    # Get a BT instance with the above parameters
    mode_direction_list = (('top', 'forward'),
                           ('top', 'backward'),
                           ('bottom', 'forward'),
                           ('bottom', 'backward'))
    if datafiles is not None:
        partial_track = partial(track_instance, params, datafiles=datafiles)
    else:
        partial_track = partial(track_instance, params, data=data)
    # Only use 1 to 4 workers. 1 means no parallelization.
    if ncores == 1:
        ballpos_top_f, ballpos_top_b, ballpos_bot_f, ballpos_bot_b = list(map(partial_track, mode_direction_list))
    else:
        with Pool(processes=max(min(ncores, 4), 1)) as pool:
            ballpos_top_f, ballpos_top_b, ballpos_bot_f, ballpos_bot_b = pool.map(partial_track, mode_direction_list)

    ballpos_top = np.concatenate((ballpos_top_f, ballpos_top_b), axis=1)
    ballpos_bottom = np.concatenate((ballpos_bot_f, ballpos_bot_b), axis=1)

    if write_ballpos:
        # Create outputdir if does not exist
        os.makedirs(params['outputdir'], exist_ok=True)
        np.savez_compressed(os.path.join(params['outputdir'], 'ballpos.npz'),
                            ballpos_top=ballpos_top, ballpos_bottom=ballpos_bottom)

    return ballpos_top, ballpos_bottom


def calculate_invalid_mask(data, threshold=5):
    """
    Create a masked array where invalid values are where the data values are too small to be coming from granulation
    The default threshold is 5 times the standard deviation below the mean

    Args:
        data (ndarray): balltracking input image, e.g. as loaded from BT.datafiles or BT.data
        threshold (int): multiplier to the standard deviation (sigma)
    Returns:
        out (MaskedArray): Numpy Masked Array with off-threshold values masked out
    """

    mean = data.mean()
    sigma = data.std()
    mind = mean - threshold * sigma
    out = ma.masked_less(data, mind)
    return out


def filter_image(image, pixel_radius=0):
    """
    Filter the image to enhance granulation signal

    Args:
        image (ndarray): input image e.g continuum intensity from SDO/HMI (2D array).
        pixel_radius (int): radius of the fourier filter in spatial domain units (pixels) instead of k-space.

    Returns:
        fdata: filtered data (2D array)
    """

    if image.shape[0] != image.shape[1]:
        print('Image must have equal dimensions')
        sys.exit(1)
    # Make sure to filter on even dimensions
    if image.shape[0] % 2 != 0:
        image2 = np.zeros([image.shape[0]+1, image.shape[1]+1])
        image2[0:image.shape[0], 0:image.shape[1]] = image
    else:
        image2 = image

    ffilter_hpf = filters.han2d_bandpass(image2.shape[0], 0, pixel_radius)
    fdata = filters.ffilter_image(image2, ffilter_hpf)

    if image.shape[0] % 2 != 0:
        fdata = fdata[0:image.shape[0], 0:image.shape[1]]

    return fdata


def rescale_frame(data, offset, norm_factor):
    """
    Rescales the images for Balltracking.

    Subtracts the mean from the data and divide by a scaling factor.
    For balltracking, this scaling factor shall be the standard deviation of the whole data series.
    See http://mathworld.wolfram.com/HanningFunction.html

    Args:
        data (ndarray): 2D frame (e.g: continuum image or magnetogram)
        offset (float): scalar that offsets the data. Typically the mean of the data or of masked data
        norm_factor (float): scalar that divides the data. Typically a multiple of the standard deviation

    Returns:
        rescaled_data (ndarray): rescaled image
    """
    rescaled_data = data - offset
    rescaled_data = rescaled_data / norm_factor

    return rescaled_data


def prep_data(image, mean, sigma, sigma_factor=1):
    """
    Rescale the image into a data surface

    The resulting standard deviation is equal to sigma_factor * sigma, typically between [0, 1] depending
    on the statistical properties of the image series. The data intensity is centered around the mean.

    Args:
        image (ndarray): image to be rescaled. E.g continuum intensity from SDO/HMI
        mean (float): offset for rescaling the image as a data surface to make it centered around the mean
        sigma (float): standard deviation to normalize with, it can be the one of the image itself,
        or of another one in the series
        sigma_factor (float): Multiplier to sigma

    Returns:
        surface (ndarray): data surface (2D array)
    """

    # Filter the image to isolate granulation
    fdata = filter_image(image)
    # Rescale image to a data surface
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(np.float32)

    return surface


def prep_data2(image, sigma_factor=1, pixel_radius=0):
    """
    Mean-normalized the input data where sigma is calculated from the input, filtered data and not from
    an externally provided sigma.

    Implemented as follows:
        - First, filter image to focus on the granulation
        - Sigma-clip outliers (e.g. sunspots)
        - Get mean and standard deviation from the masked array, not affected by bad invalid values (sunspot, ...)
        - Generate the data surface from the image with the masked mean and sigma

    Args:
        image (ndarray): input image e.g continuum intensity from SDO/HMI
        sigma_factor (float): Multiplier to the standard deviation
        pixel_radius (float): radius of the fourier filter converted in spatial domain units (pixels) instead of k-space

    Returns:
        surface (ndarray): data surface
        mean: mean of the data used to offset
        sigma: standard deviation of the filtered image used to normalize before multiplying by the sigma_factor
    """

    # Filter the image to isolate granulation. Ignored if pixel_radius = 0
    if pixel_radius == 0:
        fdata = image
    else:
        fdata = filter_image(image, pixel_radius=pixel_radius)
    masked_fdata = calculate_invalid_mask(fdata)
    mean = masked_fdata.mean()
    sigma = masked_fdata.std()
    # Rescale image to a data surface
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(np.float32)

    return surface, mean, sigma


def put_balls_on_surface(surface, x, y, rs, dp):
    """Initialize the vertical position of the balls for given x,y coordinates"""

    if x.ndim != 1 or y.ndim != 1:
        sys.exit("Input coordinates have incorrect dimensions. "
                 "x and y must be 1D numpy arrays")

    z = surface[y.astype(np.int32), x.astype(np.int32)]
    z += rs * (1 - dp / 2)
    return z


def integrate_motion(bt, surface, return_copies=False):
    """Integrate balls position for one integration step"""

    # Get the valid balls & unpack vector components for better readability
    #print("Get the valid balls")
    xt, yt, zt = bt.pos[:, bt.new_valid_balls_mask]
    vxt, vyt, vzt = bt.vel[:, bt.new_valid_balls_mask]
    # Update the balls grids with current positions
    # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
    # Clipping balls rows and cols. Stay away from borders by 1 px
    # because of the interpolation scheme
    #print("Clipping balls rows and cols.")
    bcols = np.clip(bt.bcols + xt, 1, bt.nx - 2)
    brows = np.clip(bt.brows + yt, 1, bt.ny - 2)
    # "ds" stands for "data surface"
    # These interpolations gives identical results in Matlab
    ds = cinterp.bilin_interp2f(surface, bcols, brows)
    fxt, fyt, fzt = compute_force(bt.rs, bt.am, bt.k_force, brows, bcols, xt, yt, zt, ds)

    # Integrate velocity
    vxt += fxt
    vyt += fyt
    vzt += fzt
    # Integrate position including effect of a damped velocity
    # Damping is added arbitrarily for the stability of the code.
    xt += vxt * bt.tdx * (1 - bt.e_tdx_)
    yt += vyt * bt.tdy * (1 - bt.e_tdy_)
    zt += vzt * bt.zdamping * (1 - bt.e_tdz_)

    bt.pos[0, bt.new_valid_balls_mask] = xt
    bt.pos[1, bt.new_valid_balls_mask] = yt
    bt.pos[2, bt.new_valid_balls_mask] = zt
    # Update the velocity with the damping used above
    bt.vel[0, bt.new_valid_balls_mask] = vxt * bt.e_tdx_
    bt.vel[1, bt.new_valid_balls_mask] = vyt * bt.e_tdy_
    bt.vel[2, bt.new_valid_balls_mask] = vzt * bt.e_tdz_

    if return_copies:
        force = np.array([fxt, fyt, fzt])
        return bt.pos.copy(), bt.vel.copy(), force


def compute_force(rs, am, k_force, brows, bcols, xt, yt, zt, ds):
    """Compute the Newtonian forces that pushes the balls toward local minima"""

    delta_x = xt - bcols
    delta_y = yt - brows
    delta_z = ds - zt

    r = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    # Singularity at r = 0. Need to get rid of them and force beyond the radius must be set to zero
    # Need to fill the masked values before the summation. Otherwise, subtracting bt.am on a masked value still gives a
    # masked value, instead of -bt.am = -1.0.
    rmask = np.logical_or(r == 0, r > rs)
    rm = np.ma.masked_array(r, mask=rmask)
    # When the sum is performed on an entirely masked row (1 ball grid), we must not end up with the default filled value
    # instead, we must get fn = 0 for that row. The fill_value =0 takes care of that.
    fn = k_force * (rm - rs)/rm

    fxtm = -np.ma.sum(fn * delta_x, 0)
    fytm = -np.ma.sum(fn * delta_y, 0)
    fxt = np.ma.filled(fxtm, fill_value=0)
    fyt = np.ma.filled(fytm, fill_value=0)
    # # Buoyancy must stay oriented upward. Used to be signed, but that caused more lost balls without other advantage
    fztm = -np.ma.sum(fn * np.abs(delta_z), 0)
    # On a few previous versions this was mistakenly resulting in all force components = 0 when filling the value after
    # the subtraction by bt.am.
    fzt = np.ma.filled(fztm, fill_value=0) - am

    return fxt, fyt, fzt


def ravel_index(x, dims):
    i = 0
    for dim, j in zip(dims, x):
        i *= dim
        i += j
    return int(i)


def get_off_edges_mask(rs, nx, ny, x, y):
    """
    Create a mask that flags the positions of balls that have crossed the edges of the image

    Args:
        rs (int): balls radius
        nx (int): horizontal image size in pixels
        ny (int): vertical image size in pixels
        x (ndarray): array of balls x-coordinates
        y (ndarray): array of balls y-coordinates

    Returns:
        off_edges_mask (ndarray) : mask of flagged positions. 1-valued where the balls are off the edges.

    """
    off_edge_left = x - rs < 0
    off_edge_right = x + rs > nx - 1
    off_edge_bottom = y - rs < 0
    off_edge_top = y + rs > ny - 1
    off_edges_mask = np.logical_or.reduce(np.array((off_edge_left, off_edge_right, off_edge_bottom, off_edge_top)))

    return off_edges_mask


def make_velocity_from_tracks(ballpos, dims, trange, fwhm, kernel='gaussian'):
    """
    Calculate the velocity field

    Differentiate the position to get the velocity in Lagrange ref. frame and
    convert to Euler ref. frame.

    Args:
        ballpos (ndarray): array of ball positions. Dimensions are [xyz, ball number, time]
        dims (tuple): (ny, nx) dimensions of the images used for the tracking.
        trange (sequence): sequence of [1st index, last index[ on time axis over which the flows are averaged
        fwhm (int): full width at half maximum for the spatial gaussian smoothing.
        kernel (str): kernel for smoothing the velocity: either 'gaussian' or 'boxcar'

    Returns:
        vx (ndarray): 2D x-component of the flow field
        vy (ndarray): 2D y-component of the flow field
        wplane (ndarray): weight plane
    """

    # Slices for differentiating the ball positions, in ascending start & end frame number
    tslices = (slice(trange[0], trange[1]-1), slice(trange[0]+1, trange[1]))
    ny, nx = dims

    # Differentiate positions. Must take care of the flagged values? Yes. -1 - (-1) = 0, not NaN.
    # 1) Get the coordinate of the velocity vector
    bposx = ballpos[0, :, :].copy()
    bposy = ballpos[1, :, :].copy()

    nan_mask = bposx == -1
    bposx[nan_mask] = np.nan
    bposy[nan_mask] = np.nan

    vx_lagrange = bposx[:, tslices[1]] - bposx[:, tslices[0]]
    vy_lagrange = bposy[:, tslices[1]] - bposy[:, tslices[0]]

    # px where bposx == -1 will give -1. Same for py
    px_lagrange = np.round((bposx[:, tslices[0]] + bposx[:, tslices[1]])/2)
    py_lagrange = np.round((bposy[:, tslices[0]] + bposy[:, tslices[1]])/2)
    # Exclude the -1 and NaN flagged positions using a mask.
    valid_mask = np.isfinite(vx_lagrange)
    # Taking the mask of the 2D arrays convert them to 1D arrays
    px_lagrange = px_lagrange[valid_mask]
    py_lagrange = py_lagrange[valid_mask]
    vx_lagrange = vx_lagrange[valid_mask]
    vy_lagrange = vy_lagrange[valid_mask]
    # Convert 2D coordinates of position into 1D indices. These are the 1D position of each v*_lagrange data point
    p1d = (px_lagrange + py_lagrange*nx).astype(np.uint32)
    # Weight plane. Add 1 for each position
    wplane = np.zeros([ny*nx])
    np.add.at(wplane, p1d, 1)
    # Build the Euler velocity map
    # vxplane = np.zeros([ny, nx])
    # vyplane = np.zeros([ny, nx])
    vx_euler = np.zeros([nx*ny])
    vy_euler = np.zeros([nx*ny])
    # Implement Matlab version below
    # % Matlab version
    # for jj=1:numel(vpos1D)
    #     vxplane(vpos1D(jj)) = vxplane(vpos1D(jj)) + vxii(jj);
    #     vyplane(vpos1D(jj)) = vyplane(vpos1D(jj)) + vyii(jj);
    # end

    for j in range(p1d.size):
        vx_euler[p1d[j]] += vx_lagrange[j]
        vy_euler[p1d[j]] += vy_lagrange[j]

    # as:
    # np.add.at(vx_euler, p1d, vx_lagrange)
    # np.add.at(vy_euler, p1d, vy_lagrange)
    # Reshape to 2D
    vx_euler = vx_euler.reshape([ny, nx])
    vy_euler = vy_euler.reshape([ny, nx])
    wplane   = wplane.reshape([ny, nx])

    # Spatial average (convolve with gaussian)
    sigma = fwhm/2.35
    if kernel == 'gaussian':
        vx_euler = gaussian_filter(vx_euler, sigma=sigma, order=0)
        vy_euler = gaussian_filter(vy_euler, sigma=sigma, order=0)
        wplane   = gaussian_filter(wplane, sigma=sigma, order=0)
    elif kernel == 'boxcar':
        box = np.ones([fwhm, fwhm]) / fwhm**2
        vx_euler = convolve2d(vx_euler, box, mode='same')
        vy_euler = convolve2d(vy_euler, box, mode='same')
        wplane = convolve2d(wplane, box, mode='same')

    vx_euler /= wplane
    vy_euler /= wplane

    return vx_euler, vy_euler, wplane


def bilin_interp_d(image, x, y):
    """ Bilinear interpolation. About 7x to 10x slower than the Cython implementation. """

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    q00 = image[y0, x0]
    q01 = image[y1, x0]
    q10 = image[y0, x1]
    q11 = image[y1, x1]

    dx0 = x - x0
    dy0 = y - y0
    dx1 = x1 - x
    dy1 = y1 - y

    w11 = dx1 * dy1
    w10 = dx1 * dy0
    w01 = dx0 * dy1
    w00 = dx0 * dy0

    return w11*q00 + w10*q01 + w01*q10 + w00*q11


def mesh_ball(rs, npts=20):
    """Creates a spherical mesh"""
    # parametrize mesh
    t = np.linspace(0, 1, npts)
    th, ph = np.meshgrid(t*pi, t*2*pi)
    # coordinates of sphere mesh
    x = rs * cos(th)
    y = rs * sin(th) * cos(ph)
    z = rs * sin(th) * sin(ph)
    return x, y, z

##############################################################################################################
#################    Calibration     #########################################################################
##############################################################################################################
class Calibrator:

    def __init__(self, images, drift_rates, trange, rs, ballspacing, dp, sigma_factor, filter_radius, intsteps,
                 outputdir, drift_dir=None, output_prep_data=False, normalization = True, read_write_drift_images=False,
                 filter_function=None, subdirs=None, basename='drift', save_ballpos_list=True, nthreads=1, verbose=False):

        """

        :param images: if None, will use the one already on disk
        :param drift_rates: rate at which the images are shifted (px/frame)
        :param trange: list of 2 indices in the list of images for the start and end frame used to calibrate.
        :param rs: ball radius
        :param ballspacing: minimum spacing between the balls at any new frame, also exact spacing at initialization.
        :param dp: characteristic depth.
        :param sigma_factor: image intensity scaling factor
        :param filter_radius: fourier k-filtering radius in pixels (will be converted in k-frequencies).
        :param intsteps: number of integration steps between the images. as many interpolated frames gets created
        :param drift_dir: parent directory hosting all the drift data subdirectories (one for each drift rate)
        :param outputdir: directory for output calibration data
        :param output_prep_data: [optional] output directory for the prepped filtered data
        :param normalization: [optional] Enable/Disable image intensity mean-normalization (default is True)
        :param filter_function: [optional] custom function for filtering before writing the drift images.
        Reminder: Balltracking will still use its own filtering function (see param 'filter_radius')
        :param subdirs: [optional] user can provide drift data subdirectories. default is drift_ under drift_dir.
        :param basename: [optional] basename of the fits files in each subdirectory. They get appended by a 2-digit number.
        :param save_ballpos_list = enable/disable writing the list of ballpos arrays for all drift rates.
        :param nthreads: [optional] number of threads to use for parallelization. Default to 1.
        """

        self.images = images
        self.drift_rates = drift_rates
        self.trange = trange
        self.nframes = trange[1] - trange[0]
        self.rs = rs
        self.dp = dp
        self.intsteps = intsteps
        self.sigma_factor = sigma_factor
        self.read_write_drift_images = read_write_drift_images
        self.drift_dir = drift_dir
        self.outputdir = outputdir
        self.output_prep_data = output_prep_data
        self.normalization = normalization
        self.filter_function = filter_function
        self.filter_radius = filter_radius
        self.ballspacing = ballspacing
        self.basename = basename
        self.save_ballpos_list = save_ballpos_list
        self.nthreads = nthreads
        self.verbose = verbose

        if self.drift_dir is None:
            self.drift_dir = outputdir

        os.makedirs(self.outputdir, exist_ok=True)
        os.makedirs(self.drift_dir, exist_ok=True)
        if subdirs is None:
            self.subdirs = [os.path.join(self.drift_dir, 'drift_{:02d}'.format(i)) for i in range(len(drift_rates))]
        else:
            self.subdirs = subdirs


    def drift_series(self, rate_idx):

        if self.read_write_drift_images:
            # Files supposed to be created or to be read if already exist.
            filepaths = [Path(os.path.join(self.subdirs[rate_idx], '{:s}_{:04d}.fits'.format(self.basename, i))) for i in range(*self.trange)]

            if self.images is None and check_file_series(filepaths):
                # does not save much time compared to the total time of balltracking, should I bother?
                print("Reading existing drift images at rate: [{:.2f}, {:.2f}] px/frame"
                      .format(self.drift_rates[rate_idx][0], self.drift_rates[rate_idx][1]))

                # Get a sample for the size
                sample = fitstools.fitsread(str(filepaths[0]), cube=false, astropy=True) #fitsio.read(str(filepaths[0]))
                drift_images = np.zeros([sample.shape[0], sample.shape[1], self.nframes])
                for i, f in enumerate(filepaths):
                    drift_images[:, :, i] = fitstools.fitsread(str(f), cube=false, astropy=True) #fitsio.read(str(f))
            elif self.images is not None:
                # Use the self.images to create the drift images out of them
                # os.makedirs(self.subdirs[rate_idx], exist_ok=True)
                print("Creating drift images at rate: [{:.2f}, {:.2f}] px/frame".format(self.drift_rates[rate_idx][0], self.drift_rates[rate_idx][1]))
                drift_images = create_drift_series(self.images, self.drift_rates[rate_idx], filepaths=filepaths, filter_function=self.filter_function)
            else:
                print("Drift data do not exist. Sources images not provided. Must provide them as input")
                sys.exit(1)

        else:
            if self.images is None:
                print("Drift data do not exist. Sources images not provided. Must provide them as input")
                sys.exit(1)

            if self.verbose:
                print("Creating drift images at rate: [{:.2f}, {:.2f}] px/frame".format(self.drift_rates[rate_idx][0],
                                                                                        self.drift_rates[rate_idx][1]))
            drift_images = create_drift_series(self.images, self.drift_rates[rate_idx], filter_function=self.filter_function)

        return drift_images


    def drift_all_rates(self):
        rate_idx_list = range(len(self.drift_rates))
        for idx in rate_idx_list:
             self.drift_series(idx)


    def balltrack_rate(self, rate_idx):
        """
        Balltrack the drifted images at a given drift rate index.

        :param rate_idx: index in the list of drift rates
        :return: if no_tracking is False (default), export ball position from top-side and bottom-side tracking
        """

        print(f'balltrack_rate() at rate_idx = {rate_idx}')

        drift_images = self.drift_series(rate_idx)

        # Ball parameters
        bt_params_top = OrderedDict({'rs': self.rs,
                                     'ballspacing': self.ballspacing,
                                     'intsteps': self.intsteps,
                                     'dp': self.dp,
                                     'sigma_factor': self.sigma_factor,
                                     'fourier_radius': self.filter_radius})

        bt_params_bottom = OrderedDict({'rs': self.rs,
                                        'ballspacing': self.ballspacing,
                                        'intsteps': self.intsteps,
                                        'dp': self.dp,
                                        'sigma_factor': self.sigma_factor,
                                        'fourier_radius': self.filter_radius})

        bt_params = {'top': bt_params_top, 'bottom': bt_params_bottom,
                     'nframes': self.nframes,
                     'outputdir': self.subdirs[rate_idx],
                     'output_prep_data': False,
                     'verbose': False}

        ballpos_top, ballpos_bottom = balltrack_all(bt_params, data=drift_images, write_ballpos=False, ncores=1)
        return ballpos_top, ballpos_bottom


    def balltrack_all_rates(self):
        """
        Balltrack the different series of drifting images. Each series drift at a different drift velocity or "drift rate".
        results saved in 2 different files. One for top-side tracking, one for bottom-side tracking

        :return: list of ballpos arrays for top-side and bottom side tracking at all drift rates.
        """

        print(f'balltrack_all_rates() on {len(self.drift_rates)} different drift rates:')
        print(self.drift_rates)
        rate_idx_list = range(len(self.drift_rates))

        if self.nthreads < 2:
            ballpos_top_list, ballpos_bottom_list = zip(*map(self.balltrack_rate, rate_idx_list))
        else:
            with Pool(processes=self.nthreads) as pool:
                print(f'starting multiprocessing pool with {self.nthreads} workers')
                ballpos_top_list, ballpos_bottom_list = zip(*pool.map(self.balltrack_rate, rate_idx_list))

        if self.save_ballpos_list:
            np.savez(os.path.join(self.outputdir, 'ballpos_list.npz'),
                     ballpos_top_list=ballpos_top_list,
                     ballpos_bottom_list=ballpos_bottom_list)
            print(f'saved ballpos_top_list and ballpos_bottom_list in {self.drift_dir}/ballpos_list.npz')

            # if return_ballpos:
        return ballpos_top_list, ballpos_bottom_list


def drift_series(images, drift_rate, subdir, use_existing=True):

    # number of files to process:
    nfiles = images.shape[2]
    # # Create outputdir
    # if drift_rate[0] < 0:
    #     drate0_str = 'ratea_m{:.2f}'.format(abs(drift_rate[0]))
    # else:
    #     drate0_str = 'ratea_p{:.2f}'.format(drift_rate[0])
    #
    # if drift_rate[1] < 0:
    #     drate1_str = 'ratea_m{:.2f}'.format(abs(drift_rate[1]))
    # else:
    #     drate1_str = 'rateb_p{:.2f}'.format(abs(drift_rate[1]))

    os.makedirs(subdir, exist_ok=True)
    # Create filenames "drift_[drift rate on x][drift rate on y]_[file number i].fits"
    filepaths = [Path(os.path.join(subdir, 'drift_{:03d}.fits'.format(i))) for i in range(nfiles)]

    drift_images = np.zeros(images.shape)

    if use_existing and check_file_series(filepaths):
        # does not save much time compared to the total time of balltracking,
        # but it significantly reduces disk usage compared to creating the images all over again.
        print("Reading existing drift images at rate: [{:.2f}, {:.2f}] px/frame".format(drift_rate[0], drift_rate[1]))
        for i in range(nfiles):
            drift_images[:, :, i] = fitstools.fitsread(str(filepaths[i]), cube=False, astropy=True) #fitsio.read(str(filepaths[i]))
    else:
        print("Creating drift images at rate: [{:.2f}, {:.2f}] px/frame".format(drift_rate[0], drift_rate[1]))
        for i in range(nfiles):
            dx = -drift_rate[0] * float(i)
            dy = -drift_rate[1]*i
            drift_images[:,:,i] = filters.translate_by_phase_shift(images[:,:,i], dx, dy)
            fitstools.writefits(drift_images[:,:,i], filepaths[i])

    return drift_images


def create_drift_series(images, drift_rate, filepaths=None, filter_function=None):
    """
    Drift the image series by translating a moving reference by an input 2D velocity vector.
    The drift operates by shifting the phase of the Fourier transform that also circularly shifts the escaping pixels
    back to the other edge.

    :param images: data cube to drift
    :param drift_rate: list of [vx,vy] signed velocity values.
    :param filepaths: paths where the output files are written.
    :param filter_function: optional filter to apply to the image
    :return: drifted images
    """

    if drift_rate[0] == 0 and drift_rate[1] == 0:
        drift_images = images
    else:
        drift_images = np.zeros(images.shape)

    for i in range(images.shape[2]):
        if (drift_rate[0] != 0) or (drift_rate[1]!=0):
            dx = drift_rate[0] * float(i)
            dy = drift_rate[1] * i
            drift_images[:, :, i] = filters.translate_by_phase_shift(images[:, :, i], dx, dy)

        if filter_function is not None:
            drift_images[:, :, i] = filter_function(drift_images[:,:,i])

        if filepaths is not None:
            fitstools.writefits(drift_images[:, :, i], filepaths[i])

    return drift_images


def fit_calibration(ballpos_list, shift_rates, trange, fwhm, dims, fov_slices, kernel):

    """
    This fits linear calibration value by calculating the mean velocity for each drift rate.
    Edge effects exist and must be excluded by slicing the data in an area unaffected by edge effects.
    :param ballpos_list:
    :param shift_rates:
    :param trange:
    :param fwhm:
    :param dims:
    :param fov_slices: 2d slices for selecting an area free of edge effects
    :param kernel: 2d smoothing kernel of the velocity field. Either 'gaussian' or 'boxcar'.
    :return:
    """

    if not isinstance(fov_slices, list):
        # print('fov_slices is not a list. Converting into one-element list.')
        fov_slices = [fov_slices, ]

    vxs, vys, wplanes = zip(*[make_velocity_from_tracks(ballpos, dims, trange, fwhm, kernel=kernel)
                              for ballpos in ballpos_list])
    # Select an ROI that contains valid data. At least one should exclude edges as wide as the ball radius.
    # This one also excludes the sunspot in the middle. Beware of bias due to differential rotation!

    vxmeans = 0

    for slices in fov_slices:
        vxmeans += np.array([vx[slices].mean() for vx in vxs])
    vxmeans /= len(fov_slices)

    p, r, _, _, _ = np.polyfit(vxmeans, shift_rates, 1, full=True)
    rmse = np.sqrt(r[0]/vxmeans.size)

    return p, rmse, vxmeans, vxs, vys


def check_file_series(filepaths):
    """
    Check if all files in a list exist
    :param filepaths: list of file paths. Can be of type string or Path objects. Converted to the latter in case of string.
    :return:
    """

    if not any(isinstance(x,Path) for x in filepaths):
        paths = [Path(x) for x in filepaths]
    else:
        paths = filepaths

    if any(x.is_file() for x in paths):
        return True
    else:
        return False

##############################################################################################################
##############################################################################################################

def make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, trange, fwhm, kernel, outputdir=None):

    vx_top, vy_top, wplane_top = make_velocity_from_tracks(ballpos_top, dims, trange, fwhm, kernel)
    vx_bottom, vy_bottom, wplane_bottom = make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm, kernel)

    vx_top *= cal_top
    vy_top *= cal_top
    vx_bottom *= cal_bottom
    vy_bottom *= cal_bottom

    vx = 0.5 * (vx_top + vx_bottom)
    vy = 0.5 * (vy_top + vy_bottom)

    if outputdir is not None:
        np.savez_compressed(
            os.path.join(outputdir, 'vxy_{:s}_fwhm_{:d}_avg_{:d}.npz'.format(kernel, fwhm, trange[1])),
            vx=vx, vy=vy)

    return vx, vy


def make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, tranges, fwhm, nsteps, maxstep, outputdir, kernel='gaussian'):


    vxl = []
    vyl = []
    lanesl = []
    for i, trange in enumerate(tranges):
        tavg = trange[1] - trange[0]
        # Velocity field
        vx, vy = make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, trange, fwhm, kernel=kernel)
        # lanes
        lanes = make_lanes(vx, vy, nsteps, maxstep)
        # Write fits file
        fitstools.writefits(vx, os.path.join(outputdir, 'vx_fwhm%d_tavg%03d_%03d.fits'%(fwhm, tavg, i)))
        fitstools.writefits(vy, os.path.join(outputdir, 'vy_fwhm%d_tavg%03d_%03d.fits'%(fwhm, tavg, i)))
        fitstools.writefits(lanes, os.path.join(outputdir, 'lanes_fwhm%d_tavg%03d_nsteps%d_%03d.fits' %(fwhm, tavg, nsteps, i)))

        vxl.append(vx)
        vyl.append(vy)
        lanesl.append(lanes)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(lanes, origin='lower', cmap='gray_r')
        # plt.xlabel('x [px]')
        # plt.ylabel('y [px]')
        # plt.title('Supergranular lanes at fwhm = %d px ; tavg = %d ; map # %03d'%(fwhm, tavg, i))
        # plt.tight_layout()
        # plt.savefig(os.path.join(outputdir, 'lanes_fwhm%d_tavg%03d_%03d.png'%(fwhm, tavg, i)))
        # plt.close()

    return vxl, vyl, lanesl


def make_lanes(vx, vy, nsteps, maxstep):

    dims = vx.shape

    # Gamma scale the data
    vmag = np.sqrt(vx**2 + vy**2)
    vmax = vmag.max()
    vn = vmag/vmax
    gamma = 0.5
    g = vn**(gamma-1)
    vxn = vx/vmax
    vyn = vy/vmax
    vxng = vxn*g
    vyng = vyn*g

    vblank = np.zeros([dims[0]+2*maxstep, dims[1]+2*maxstep], dtype=np.float32)
    vx2 = vblank.copy()
    vy2 = vblank.copy()
    vx2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vxng.astype(np.float32)
    vy2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vyng.astype(np.float32)

    vx2 *= -1
    vy2 *= -1

    x0, y0 = np.meshgrid( maxstep + np.arange(dims[1]) , maxstep + np.arange(dims[0]))
    xold = x0.flatten().astype(np.float32)
    yold = y0.flatten().astype(np.float32)

    maxv = np.sqrt(vx2.max() ** 2 + vy2.max() ** 2)

    for n in range(nsteps):

        dx1 = maxstep * cinterp.cbilin_interp1(vx2, xold, yold)/maxv
        dy1 = maxstep * cinterp.cbilin_interp1(vy2, xold, yold)/maxv

        dx2 = maxstep * cinterp.cbilin_interp1(vx2, xold+dx1, yold+dy1) / maxv
        dy2 = maxstep * cinterp.cbilin_interp1(vy2, xold+dx1, yold+dy1) / maxv

        x = xold + (dx1 + dx2)/2
        y = yold + (dy1 + dy2)/2

        xold = x
        yold = y

    xforwards = xold.reshape(dims)
    yforwards = yold.reshape(dims)

    xforwards -= maxstep
    yforwards -= maxstep
    # Clip the positions in case they overshoot due to maxstep
    np.clip(xforwards, 0, dims[1]-1, out=xforwards)
    np.clip(yforwards, 0, dims[0]-1, out=yforwards)

    gxx = np.gradient(xforwards, axis=1)
    gyy = np.gradient(yforwards, axis=0)

    lanes = np.sqrt(gxx**2 + gyy**2)

    return lanes


def make_lanes_visualization(vx, vy, nsteps, maxstep):

    dims = vx.shape

    # Gamma scale the data
    vmag = np.sqrt(vx**2 + vy**2)
    vmax = vmag.max()
    vn = vmag/vmax
    gamma = 0.5
    g = vn**(gamma-1)
    vxn = vx/vmax
    vyn = vy/vmax
    vxng = vxn*g
    vyng = vyn*g

    vblank = np.zeros([dims[0]+2*maxstep, dims[1]+2*maxstep], dtype=np.float32)
    vx2 = vblank.copy()
    vy2 = vblank.copy()
    vx2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vxng.astype(np.float32)
    vy2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vyng.astype(np.float32)

    vx2 *= -1
    vy2 *= -1

    x0, y0 = np.meshgrid( maxstep + np.arange(dims[1]) , maxstep + np.arange(dims[0]))
    xold = x0.flatten().astype(np.float32)
    yold = y0.flatten().astype(np.float32)

    maxv = np.sqrt(vx2.max() ** 2 + vy2.max() ** 2)

    # Create a storage array for the intermediate integration steps
    xtracks = np.zeros([nsteps+1, x0.size], dtype=np.float32)
    ytracks = np.zeros([nsteps+1, y0.size], dtype=np.float32)
    xtracks[0, :] = xold
    ytracks[0, :] = yold
    lanes_series = np.empty([nsteps+1, *dims])
    lanes_series[0, ...] = np.zeros(dims)

    for n in range(nsteps):

        dx1 = maxstep * cinterp.cbilin_interp1(vx2, xtracks[n, :], ytracks[n, :])/maxv
        dy1 = maxstep * cinterp.cbilin_interp1(vy2, xtracks[n, :], ytracks[n, :])/maxv

        dx2 = maxstep * cinterp.cbilin_interp1(vx2, xtracks[n, :]+dx1, ytracks[n, :]+dy1) / maxv
        dy2 = maxstep * cinterp.cbilin_interp1(vy2, xtracks[n, :]+dx1, ytracks[n, :]+dy1) / maxv

        newx = xtracks[n, :] + (dx1 + dx2)/2
        newy = ytracks[n, :] + (dy1 + dy2)/2
        xtracks[n + 1, :] = newx.copy()
        ytracks[n + 1, :] = newy.copy()

        newx -= maxstep
        newy -= maxstep

        newx = newx.reshape(dims)
        newy = newy.reshape(dims)

        # Clip the positions in case they overshoot due to maxstep
        np.clip(newx, 0, dims[1]-1, out=newx)
        np.clip(newy, 0, dims[0]-1, out=newy)

        gxx = np.gradient(newx, axis=1)
        gyy = np.gradient(newy, axis=0)

        lanes_series[n + 1, ...] = np.sqrt(gxx**2 + gyy**2)

    np.clip(xtracks - maxstep, 0, dims[1] - 1, out=xtracks)
    np.clip(ytracks - maxstep, 0, dims[0] - 1, out=ytracks)

    return lanes_series, [xtracks.reshape([nsteps+1, *dims]), ytracks.reshape([nsteps+1, *dims])]


def balltrack_calibration(bt_params, drift_rates, trange, fov_slices, reprocess_bt, outputdir, kernel, fwhm, dims,
                          drift_dir=None, images=None, basename='drift', save_ballpos_list=True, csvfile=None, verbose=False, nthreads=1):

    if 'index' not in bt_params:
        print('bt_params missing "index" key')
        sys.exit(1)

    if verbose:
        print(bt_params)

    xrates = np.array(drift_rates)[:, 0]
    idx0 = np.where(xrates == 0)[0][0]
    ballpos_list_file = os.path.join(outputdir, 'ballpos_list.npz')

    if (reprocess_bt is True) or (reprocess_bt == 'once' and not os.path.isfile(ballpos_list_file)):

        cal = Calibrator(images, drift_rates, trange,
                         bt_params['rs'],
                         bt_params['ballspacing'],
                         bt_params['dp'],
                         bt_params['sigma_factor'],
                         bt_params['fourier_radius'],
                         bt_params['intsteps'],
                         outputdir,
                         drift_dir=drift_dir,
                         output_prep_data=False,
                         basename=basename,
                         save_ballpos_list=save_ballpos_list,
                         nthreads=nthreads,
                         verbose=verbose)

        ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()
        ballposf = os.path.join(outputdir, 'ballpos_{:s}.npz'.format(str(bt_params['index'])))
        np.savez(ballposf, ballpos_top=ballpos_top_list[idx0], ballpos_bottom=ballpos_bottom_list[idx0])

    else:
        npzfile = np.load(ballpos_list_file)
        ballpos_top_list = npzfile['ballpos_top_list']
        ballpos_bottom_list = npzfile['ballpos_bottom_list']

    vx_headers_top = ['vx_top {:1.2f}'.format(vx[0]) for vx in drift_rates]
    vx_headers_bottom = ['vx_bottom {:1.2f}'.format(vx[0]) for vx in drift_rates]
    # Concatenate headers
    vx_headers = vx_headers_top + vx_headers_bottom

    p_top, _, vxmeans_top, vxs_top, vys_top = fit_calibration(ballpos_top_list, xrates, trange, fwhm, dims, fov_slices, kernel)
    p_bot, _, vxmeans_bot, vxs_bot, vys_bot = fit_calibration(ballpos_bottom_list, xrates, trange, fwhm, dims, fov_slices, kernel)

    # Concatenate above results in one single list and create a dictionnary with the concatenated keys
    dict_vxmeans = OrderedDict(zip(vx_headers, vxmeans_top.tolist() + vxmeans_bot.tolist()))

    dict_results = bt_params.copy()
    dict_results['kernel'] = kernel
    dict_results['fwhm'] = fwhm
    dict_results['p_top_0'] = p_top[0]
    dict_results['p_top_1'] = p_top[1]
    dict_results['p_bot_0'] = p_bot[0]
    dict_results['p_bot_1'] = p_bot[1]
    dict_results.update(dict_vxmeans)

    if csvfile is None:
        csvfile = os.path.join(outputdir, 'param_sweep_{:s}.csv'.format(str(bt_params['index'])))
    with open(csvfile, 'w') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow(list(dict_results.keys()))
        csvwriter.writerow(list(dict_results.values()))

    npzf = os.path.join(outputdir, 'mean_velocity_{:s}.npz'.format(str(bt_params['index'])))
    # Index where xrates = 0. Used to save specific velocity map

    np.savez_compressed(npzf,
                        vx_top=vxs_top[idx0],
                        vy_top=vys_top[idx0],
                        vx_bot=vxs_bot[idx0],
                        vy_bot=vys_bot[idx0])

    print(f'saved dictionnary at {npzf}')

    return dict_results


def meshgrid_params_to_list(args):
    mesh = np.meshgrid(*args, indexing='ij')
    list_ravel = []
    for elem in mesh:
        list_ravel.append(np.ravel(elem))
    args_list = [list(a) for a in zip(*list_ravel)]
    return args_list


def get_bt_params_list(bt_params, param_names, param_lists):

    param_mesh_list = meshgrid_params_to_list(param_lists)
    bt_params_list = []
    bt_params2 = bt_params.copy()
    for i, p_list in enumerate(param_mesh_list):
        for n, name in enumerate(param_names):
            bt_params2[name] = p_list[n]
        bt_params2['index'] = i
        bt_params_list.append(bt_params2)
        bt_params2 = bt_params.copy()
    return bt_params_list
