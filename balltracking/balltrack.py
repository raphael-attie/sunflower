import sys
import os
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
from numpy import pi, cos, sin
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from scipy.signal import convolve2d
from cython_modules import interp as cinterp
import filters
import fitstools
from astropy.io import fits

DTYPE = np.float32


class BT:

    def __init__(self, rs, dp, am, ballspacing, intsteps, sigma_factor, fourier_radius, trange,
                 side='top', direction='forward', datafiles=None, data=None, outputdir_prep=None, verbose=False,
                 image_reader=None):

        """ This is the class hosting all the parameters and intermediate results of balltracking.

        Attributes:

            rs (int): balls radius in pixels
            dp (float): Characteristic percentage depth. 0 < dp < 1
            am (float): acceleration factor. Used to be 0.6 in Potts implementation. 0.3 best for HMI
            ballspacing (int): nb of pixels between balls center at the initialization stage
            intsteps (int): nb of intermediate frames using linear interpolation.
            sigma_factor (float): multiplier to the standard deviation
            fourier_radius (float): radius for high-pass image fourier filter,
                                    to be given in period units (pixels) instead of k-space frequency.
            trange (tuble): Frame indices for the (start, end) frame
            side (str): determines which side of the data surface to track: 'top' or 'bottom'.
            direction (str): determines whether we track 'forward' or 'backward' in time.
            datafiles (list): list of FITS file paths
            data (ndarray): instead of providing a list of files with datafiles, one can directly provide a 3D array
            outputdir_prep (str): output directory to write out the intermediary surface data. For sanity check.
            verbose (bool): toggle verbosity
            image_reader (ufunc): user-supplied function to read the image files

        """

        self.datafiles = datafiles
        self.data = data
        if image_reader is None:
            self.image_reader = fitstools.fitsread
        else:
            self.image_reader = image_reader

        if direction != 'forward' and direction != 'backward':
            raise ValueError

        self.rs = rs
        self.am = am
        self.dp = dp
        self.ballspacing = int(ballspacing)
        self.intsteps = int(intsteps)

        self.outputdir_prep= outputdir_prep
        self.verbose = verbose

        self.side = side
        self.direction = direction

        self.trange = trange
        self.nframes = trange[-1] - trange[0]


        # Get a sample. 1st of the series in forward direction. last of the series in backward direction.

        if self.data is None:
            self.sample = self.image_reader(self.datafiles, tslice=trange[0], cube=False).astype(DTYPE)
        else:
            self.sample = self.data[trange[0], :, :]


        # Set frame dimensions
        self.nx = int(self.sample.shape[1])
        self.ny = int(self.sample.shape[0])
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
        self.zstart = np.zeros(self.xstart.shape, dtype=DTYPE)
        self.nballs = self.xstart.size
        # Initialize the array of the lifetime (age) of the balls
        self.balls_age = np.ones([self.nballs], dtype=np.uint32)

        # Coarse grid
        # Matlab: finegrid=zeros(ceil(BT.nr/BT.finegridspacing),ceil(BT.nc/BT.finegridspacing));
        self.coarse_grid = np.zeros(
            [np.ceil(self.ny / self.ballspacing).astype(int), np.ceil(self.nx / self.ballspacing).astype(int)],
            dtype=np.uint32)
        # Dimensions of the coarse grid
        self.nyc_, self.nxc_ = self.coarse_grid.shape

        # Force scaling factor
        self.k_force = self.am / (self.dp ** 2 * pi * self.rs ** 2)
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
        self.balls_age_t = np.zeros([self.nballs, self.nframes], dtype=np.uint32)
        # Storage arrays of the above, for all time steps
        self.ballpos = np.zeros([3, self.nballs, self.nframes], dtype=DTYPE)
        self.ballvel = np.zeros([3, self.nballs, self.nframes], dtype=DTYPE)

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
        self.min_ds_ = -5
        # Mask of bad balls
        self.bad_balls_mask = np.zeros(self.nballs, dtype=bool)
        # Mask of valid balls
        self.new_valid_balls_mask = np.ones(self.nballs, dtype=bool)

        self.surface, mean, sigma = prep_data2(self.sample, self.sigma_factor)
        self.mean = mean
        self.sigma = sigma
        # Initialize the height of the ball. Only possible if the data surface is given.
        self.pos[0, :] = self.xstart.flatten()
        self.pos[1, :] = self.ystart.flatten()
        self.zstart = put_balls_on_surface(self.surface, self.xstart.ravel(), self.ystart.ravel(), self.rs, self.dp)
        self.pos[2, :] = self.zstart.copy()
        # Set the coarse grid: populate edges to avoid "leaks" of balls ~ balls falling off.
        # Although can't remember if that was actually ever used in the Matlab implementation
        self.coarse_grid[0,:] = 1
        self.coarse_grid[:, 0] = 1
        self.coarse_grid[-1,:] = 1
        self.coarse_grid[:, -1] = 1

    def coarse_grid_pos(self, x, y):
        """
            Convert the x,y coordinates of the balls dropped on the surface into coordinates in the coarse grid.

        Args:
            x (np.ndarray): x-coordinates of the balls
            y (np.ndarray): y-coordinates of the balls

        Returns:
            x,y coordinates on the coarse grid, and corresponding 1D coordinates
        """
        #
        # Get the position on the coarse grid, clipped to the edges of that grid.
        xcoarse = np.uint32(np.clip(np.floor(x / self.ballspacing), 0, self.nxc_ - 1))
        ycoarse = np.uint32(np.clip(np.floor(y / self.ballspacing), 0, self.nyc_ - 1))
        # Convert to linear (1D) indices. One index per ball
        coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), self.coarse_grid.shape)
        return xcoarse, ycoarse, coarse_idx

    def track(self):
        """Tracking outer loop

        Loop over the images and update the balls positions
        """

        # Outer loop goes over the data frames.
        # If data is a fits cube, we just access a slice of it

        for n in range(self.trange[0], self.trange[1]):
            if self.verbose:
                print("Tracking direction {}/{}, frame {:d}".format(self.direction, self.side, n))

            if self.direction == 'forward':
                if self.data is None:
                    image = self.image_reader(self.datafiles, tslice=n, cube=False).astype(DTYPE)
                else:
                    image = self.data[n, :, :]
            else:
                if self.data is None:
                    image = self.image_reader(self.datafiles, tslice=self.nframes - 1 - n, cube=False).astype(DTYPE)
                else:
                    image = self.data[self.nframes - 1 - n, :, :]


            # TODO: check the choice of prep_data regarding mean normalization with fixed mean or time-dependent one
            # self.surface = prep_data(image, self.mean, self.sigma, sigma_factor=self.sigma_factor)
            self.surface, _, _ = prep_data2(image, sigma_factor=self.sigma_factor, pixel_radius=self.fourier_radius)
            if self.side == 'bottom':
                self.surface = -self.surface

            if self.outputdir_prep is not None:
                filename_surface = Path(self.outputdir_prep, f'prep_data_{self.direction}_{self.side}_{n:05d}.fits')
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
            # At the end of the time loop, flip timeline for backward tracking to restore the forward timeline
            self.ballpos = np.flip(self.ballpos, 2)
            self.ballvel = np.flip(self.ballvel, 2)

    def get_bad_balls(self):
        """
        Mask out all the balls that are outside the tracking requirements.

        Returns
        -------

        """
        # See https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
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

        # Get ball number and balls age sorted by position, and array of valid balls indices
        sorted_balls = np.argsort(coarse_pos)
        balls_age = balls_age[sorted_balls]
        coarse_pos = coarse_pos[sorted_balls]
        valid_balls_idx = valid_balls_idx[sorted_balls]
        # There can be repetitions in the coarse_pos because there can be more than one ball per finegrid cell.
        # The point is to keep only one ball per coarse grid point: the oldest.
        # Sort coarse_pos further using the balls age as weight and extract a unique set where each ball is the oldest
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
        xnew = self.ballspacing * x0_empty.astype(DTYPE)#+ bt.rs
        ynew = self.ballspacing * y0_empty.astype(DTYPE)#+ bt.rs
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


def track_instance(params, side_direction, datafiles=None, data=None, **kwargs):
    """
    Wrapper to run balltracking on a given tuple of (side, direction).
    This routine must be executed with 4 of these pairs for minimizing the random error.
    Either the list of files or numpy array of images must be provided.

    Args:
        params (dict): ball parameters for the BT class
        side_direction (tuple): (BT.side) and (BT.direction) listed as ('top', 'forward') or ('bottom', 'backward')
        datafiles (list): path to data cube file or list of FITS files. Ignored if `data` is provided.
        data (np.ndarray): 3D array with time on the 1st dimension: (time, y-axis, x-axis).

    Returns:
        BT.ballpos : Array storing the positions of the balls at each frame
    """

    # Get which side we're tracking: either top or bottom
    params_side = params[side_direction[0]]

    bt_instance = BT(**params_side,
                     side=side_direction[0],
                     direction=side_direction[1],
                     datafiles=datafiles,
                     data=data,
                     **kwargs)

    bt_instance.track()

    return bt_instance.ballpos


def balltrack_all(bt_params_top, bt_params_bottom, outputdir,
                  datafiles=None, data=None, write_ballpos=True, ncores=1, **kwargs):
    """
    Run the tracking on the 4 (side, direction) pairs:
    (('top', 'forward'),
     ('top', 'backward'),
     ('bottom', 'forward'),
     ('bottom', 'backward'))

     Can be executed in a pool of up to 4 parallel workers.

    Args:
        bt_params_top (dict): Top-side ball parameters relevant to the BT class
        bt_params_bottom (dict): Bottom-side ball parameters relevant to the BT class
        outputdir (str): output directory to save the 3D arrays of ball positions
        datafiles (str or list): path to data cube file or list of FITS files
        data (np.ndarray): Series of 3D Numpy arrays with time on the 3rd index: (y-axis, x-axis, time-axis)
        write_ballpos (bool): sets whether to write the array of ball positions
        ncores: number of cores to use for running the 4 sides/directions in parallel
            Default is 1 for sequential processing. There can up to 4 workers for these parallel tasks (2-3x speed-up)

    Returns:
        ballpos_top (np.ndarray): 3D array of ball positions for top-side tracking -> [xyz, ball #, time]
        ballpos_bottom (np.ndarray): 3D array of ball positions for bottom-side tracking -> [xyz, ball #, time]

    """

    # Check user data input
    if (datafiles is None or not isinstance(datafiles, str)) and not isinstance(datafiles, list) and data is None:
        sys.exit('missing data or datafiles in balltrack_all')
    # Get a BT instance with the above parameters
    side_direction_list = (('top', 'forward'),
                           ('top', 'backward'),
                           ('bottom', 'forward'),
                           ('bottom', 'backward'))

    bt_params_sides = {
        'top': bt_params_top,
        'bottom': bt_params_bottom
    }

    partial_track = partial(track_instance, bt_params_sides, datafiles=datafiles, data=data, **kwargs)
    # Only use 1 to 4 workers. 1 means no parallelization.
    if ncores == 1:
        ballpos_top_f, ballpos_top_b, ballpos_bot_f, ballpos_bot_b = list(map(partial_track, side_direction_list))
    else:
        with Pool(processes=max(min(ncores, 4), 1)) as pool:
            ballpos_top_f, ballpos_top_b, ballpos_bot_f, ballpos_bot_b = pool.map(partial_track, side_direction_list)

    ballpos_top = np.concatenate((ballpos_top_f, ballpos_top_b), axis=1)
    ballpos_bottom = np.concatenate((ballpos_bot_f, ballpos_bot_b), axis=1)

    if write_ballpos:
        # Create outputdir if it does not exist
        os.makedirs(outputdir, exist_ok=True)
        np.savez_compressed(Path(outputdir, 'ballpos.npz'),
                            ballpos_top=ballpos_top, ballpos_bottom=ballpos_bottom)

    return ballpos_top, ballpos_bottom


def balltrack_main_hmi(bt_params, outputdir, datafiles=None, data=None, ncores=1, **kwargs):

    # Calibration HMI is so far considering the same top-side and bottom-side parameters
    ballpos_top, ballpos_bottom = balltrack_all(bt_params, bt_params, outputdir,
                                                data=data,
                                                datafiles=datafiles,
                                                # ballpos arrays will be written after all rates are processed
                                                # Thus disabling saving them in each run
                                                write_ballpos=True,
                                                ncores=ncores,
                                                **kwargs)
    return ballpos_top, ballpos_bottom


def calculate_invalid_mask(data, threshold=5):
    """
    Create a masked array where invalid values are where the data values are too small to be coming from granulation
    The default threshold is 5 times the standard deviation below the mean

    Args:
        data (np.ndarray): balltracking input image, e.g as loaded from BT.datafiles or BT.data
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
        image (np.ndarray): input image e.g. continuum intensity from SDO/HMI (2D array).
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
        data (np.ndarray): 2D frame (e.g: continuum image or magnetogram)
        offset (float): scalar that offsets the data. Typically the mean of the data or of masked data
        norm_factor (float): scalar that divides the data. Typically a multiple of the standard deviation

    Returns:
        rescaled_data (np.ndarray): rescaled image
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
        image (np.ndarray): image to be rescaled. E.g continuum intensity from SDO/HMI
        mean (float): offset for rescaling the image as a data surface to make it centered around the mean
        sigma (float): standard deviation to normalize with, it can be the one of the image itself,
        or of another one in the series
        sigma_factor (float): Multiplier to sigma

    Returns:
        surface (np.ndarray): data surface (2D array)
    """

    # Filter the image to isolate granulation
    fdata = filter_image(image)
    # Rescale image to a data surface
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(DTYPE)

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
        image (np.ndarray): input image e.g continuum intensity from SDO/HMI
        sigma_factor (float): Multiplier to the standard deviation
        pixel_radius (float): radius of the fourier filter converted in spatial domain units (pixels) instead of k-space

    Returns:
        surface (np.ndarray): data surface
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
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(DTYPE)

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
    # Damping is added arbitrarily for the stability of the code, fine-tuned with a grid-based optimization.
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
    # Singularity at r = 0. Need to get rid of them. The force beyond the radius must be set to zero
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


def get_off_edges_mask(rs, nx, ny, x, y):
    """
    Create a mask that flags the positions of balls that have crossed the edges of the image

    Args:
        rs (int): balls radius
        nx (int): horizontal image size in pixels
        ny (int): vertical image size in pixels
        x (np.ndarray): array of balls x-coordinates
        y (np.ndarray): array of balls y-coordinates

    Returns:
        off_edges_mask (np.ndarray) : mask of flagged positions. 1-valued where the balls are off the edges.

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
        ballpos (np.ndarray): array of ball positions. Dimensions are [xyz, ball number, time]
        dims (tuple): (ny, nx) dimensions of the images used for the tracking.
        trange (sequence): sequence of [1st index, last index[ on time axis over which the flows are averaged
        fwhm (int): full width at half maximum for the spatial gaussian smoothing.
        kernel (str): kernel for smoothing the velocity: either 'gaussian' or 'boxcar'

    Returns:
        vx (np.ndarray): 2D x-component of the flow field
        vy (np.ndarray): 2D y-component of the flow field
        wplane (np.ndarray): weight plane
    """

    ny, nx = dims

    # Differentiate positions. Must take care of the flagged values? Yes. -1 - (-1) = 0, not NaN.
    # 1) Get the coordinate of the velocity vector
    bposx = ballpos[0, :, :].copy()
    bposy = ballpos[1, :, :].copy()

    nan_mask = bposx == -1
    bposx[nan_mask] = np.nan
    bposy[nan_mask] = np.nan

    # Watch out: slicing excludes the last index. Adding +1 to trange[1] makes sure we reach index trange[1]
    vx_lagrange = bposx[:, trange[0]+1:trange[1]] - bposx[:, trange[0]:trange[1]-1]
    vy_lagrange = bposy[:, trange[0]+1:trange[1]] - bposy[:, trange[0]:trange[1]-1]

    # px where bposx == -1 will give -1. Same for py
    px_lagrange = np.round((bposx[:, trange[0]:trange[1]-1] + bposx[:, trange[0]+1:trange[1]])/2)
    py_lagrange = np.round((bposy[:, trange[0]:trange[1]-1] + bposy[:, trange[0]+1:trange[1]])/2)
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
# Calibration
##############################################################################################################


class Calibrator:

    def __init__(self, bt_params, vx_rates, vy_rates, trange, fwhm, images, outputdir_cal,
                 component='x', kernel='gaussian', read_drift_images=False,
                 save_ballpos_list=True, reprocess_existing=True, verbose=False, return_ballpos=False,
                 ncpus=1):

        """
        Calibrate the top and bottom tracking using a series of images drifting at uniform offset velocities
        
        Attributes:

            bt_params (dict): inputs of top tracking: rs, ballspacing, am, dp, sigma_factor, fourier_radius, intsteps
            vx_rates: x-direction rate at which the images are shifted (px/frame)
            vy_rates: y-direction rate at which the images are shifted (px/frame)
            trange: list of 2 indices in the list of images for the start and end frame used to calibrate.
            fwhm: FWHM of the gaussian smoothing on the flow fields
            images: if None, will use the ones already on disk in `drift_dirs`. images or drift_dirs must be specified
            outputdir_cal: directory for output calibration data
            component (str): velocity vector component over which to fit, either 'x', 'y', or 'xy' for both
            kernel: either 'gaussian' or 'boxcar' for smoothing the velocity vector field
            read_drift_images: toggle whether to read images from existing files
            Balltracking will still use its own filtering function (see input 'filter_radius')
            save_ballpos_list = enable/disable writing the list of ballpos arrays for all drift rates.
            ncpus: number of parallel workers.

        """

        # Input check
        if images is None and not read_drift_images:
            print("Drift data do not exist. Source images must be set")
            sys.exit(1)

        self.bt_params = bt_params
        self.vx_rates = vx_rates
        self.vy_rates = vy_rates
        self.drift_rates = np.stack((vx_rates, vy_rates), axis=1)
        self.trange = trange
        self.fwhm = fwhm
        self.nframes = trange[1] - trange[0]
        self.images = images
        self.read_drift_images = read_drift_images
        self.outputdir_cal = outputdir_cal
        # parent directory hosting all the drift data subdirectories (one for each drift rate)
        self.drift_dirs = sorted(list(Path(outputdir_cal).glob('drift*')))
        if len(self.drift_dirs) < len(vx_rates):
            sys.exit('Drift directories not set correctly')
        self.component = component
        self.kernel = kernel
        self.save_ballpos_list = save_ballpos_list
        self.ncpus = ncpus
        self.reprocess_existing = reprocess_existing
        self.verbose = verbose
        self.return_ballpos = return_ballpos
        if "index" in bt_params:
            self.index = bt_params['index']
        else:
            self.index = 0

        # Get frame dimensions
        sample_file = sorted(list(self.drift_dirs[0].glob('*.fits')))[self.trange[0]].as_posix()
        print('sample file: ', sample_file)
        self.sample = fitstools.fitsread(sample_file, cube=False)

        self.dims = self.sample.shape[-2:]

        if self.kernel == 'boxcar' or self.kernel == 'gaussian':
            self.kernels = [self.kernel]
        elif self.kernel == 'both':
            self.kernels = ['boxcar', 'gaussian']
        else:
            sys.exit('smoothing kernel not supported or missing')

        os.makedirs(self.outputdir_cal, exist_ok=True)

    def get_drift_images(self, rate_idx):

        if self.read_drift_images:
            if self.verbose:
                print(self.drift_dirs[rate_idx])
            # Files supposed to be created or to be read if already exist.
            filepaths = sorted(list(Path(self.drift_dirs[rate_idx]).glob('*.fits')))[self.trange[0]:self.trange[1]]
            if not check_file_series(filepaths):
                print("Drift data do not exist. Sources images not provided. Must provide them as input")
                sys.exit(1)
            if self.verbose:
                print(f"Reading from fits existing drift images at rate: {self.drift_rates[rate_idx]} px/frame")

            drift_images = fitstools.fitsread(filepaths)

        else:
            if self.images is None:
                print("Drift data do not exist. Sources images not provided. Must provide them as input")
                sys.exit(1)

            if self.verbose:
                print(f"Getting drift images at rate: {self.drift_rates[rate_idx]} px/frame")
            drift_images = self.images[rate_idx]

        return drift_images

    def balltrack_rate(self, rate_idx):
        """
        Balltrack the drifted images at a given drift rate index.

        Args:
            rate_idx (int): index in the list of drift rates

        Returns:
            ballpos_top (np.ndarray): ball positions from top-side tracking
            ballpos_bottom (np.ndarray): ball positions from bottom-side tracking
        """

        if self.verbose:
            print(f'balltrack_rate() at rate_idx = {rate_idx}')

        drift_images = self.get_drift_images(rate_idx)

        # The calibration uses the same bt_params for top and bottom.
        # To track top and bottom with different parameters, just create different calibrators
        ballpos_top, ballpos_bottom = balltrack_all(self.bt_params, self.bt_params, self.drift_dirs[rate_idx],
                                                    data=drift_images,
                                                    # ballpos arrays will be written after all rates are processed
                                                    # Thus disabling saving them in each run
                                                    write_ballpos=False)

        return ballpos_top, ballpos_bottom

    def balltrack_all_rates(self):
        """
        Balltrack the different series of drifting images. Each series drift at a different drift velocity or "drift rate".
        results saved as npz file with top-side tracking and bottom-side tracking in two different lists.

        Returns:
            ballpos_top_list (np.ndarray): arrays of ball positions for top-side tracking at all drift rates.
            ballpos_bottom_list (np.ndarray): arrays of ball positions for bottom-side tracking at all drift rates.
        """

        ballpos_top_list, ballpos_bottom_list = None, None

        if self.verbose:
            print(f'balltrack_all_rates() on {len(self.drift_rates)} different drift rates:')
            print(self.drift_rates)
        rate_idx_list = range(len(self.drift_rates))

        npzf = Path(self.outputdir_cal, f'ballpos_list_{self.index:05d}.npz')
        if npzf.exists() and not self.reprocess_existing:
            if self.return_ballpos:
                print('loading data at existing index: ', self.index)
                with np.load(npzf) as npz:
                    ballpos_top_list = npz['ballpos_top_list']
                    ballpos_bottom_list = npz['ballpos_bottom_list']
                    return ballpos_top_list, ballpos_bottom_list
            else:
                print('skipping existing data at index: ', self.index)
                return ballpos_top_list, ballpos_bottom_list

        if self.ncpus < 2:
            ballpos_top_list, ballpos_bottom_list = zip(*map(self.balltrack_rate, rate_idx_list))
        else:
            with Pool(processes=self.ncpus) as pool:
                print(f'starting multiprocessing pool with {self.ncpus} workers')
                ballpos_top_list, ballpos_bottom_list = zip(*pool.map(self.balltrack_rate, rate_idx_list))

        if self.save_ballpos_list:
            np.savez(npzf,
                     ballpos_top_list=ballpos_top_list,
                     ballpos_bottom_list=ballpos_bottom_list)
            if self.verbose:
                print(f'saved ballpos_top_list and ballpos_bottom_list in {self.outputdir_cal}/ballpos_list_{self.index:05d}.npz')

        return ballpos_top_list, ballpos_bottom_list


    def fit_mean_velocities(self, velocities, rates):
        vel_means = np.array([vel.mean() for vel in velocities])
        p, r, _, _, _ = np.polyfit(vel_means, rates, 1, full=True)
        rmse = np.sqrt(r[0] / vel_means.size)
        return p, rmse, vel_means

    def fit_calibration(self, ballpos_list, kernel=None):

        """
        Fit a linear profile by calculating the mean velocity for each drift rate.
        Edge effects exist and must be excluded by slicing an unaffected area

        Args:
            ballpos_list (np.ndarray): List of ball positions at each drift rate.
            kernel (str): 2d smoothing kernel of the velocity field. Either 'gaussian', 'boxcar'

        Returns:
            p (list): fit parameters
            rmse (float): root-mean-square error
            vxmeans (float): mean value over the sliced velocity field
            vxs (np.ndarray): non-averaged flow fields x-component (unsliced)
            vys (np.ndarray): non-averaged flow fields y-components (unsliced)
        """

        if kernel is None:
            kernel = self.kernel
        # Convert to Euler averaged flow fields
        vxs, vys, wplanes = (
            zip(*[make_velocity_from_tracks(ballpos, self.dims, self.trange, self.fwhm, kernel=kernel)
                  for ballpos in ballpos_list])
        )

        if self.component == 'x':
            # Beware of bias due to differential rotation.
            p_x, rmse_x, vx_means = self.fit_mean_velocities(vxs, self.vx_rates)
            return p_x, rmse_x, vx_means, vxs, vys

        elif self.component == 'y':
            # Beware of bias due to differential rotation.
            p_y, rmse_y, vy_means = self.fit_mean_velocities(vys, self.vy_rates)
            return p_y, rmse_y, vy_means, vxs, vys

        elif self.component == 'xy':
            p_x, rmse_x, vx_means = self.fit_mean_velocities(vxs, self.vx_rates)
            p_y, rmse_y, vy_means = self.fit_mean_velocities(vys, self.vy_rates)
            return p_x, p_y, rmse_x, rmse_y, vy_means, vxs, vys

        else:
            sys.exit('component not recognized in fit_calibration()')

    def calibration_run_fit(self, ballpos_top_list, ballpos_bottom_list, verbose=False):
        print('Running calibration fit...')
        vx_headers_top = ['vx_top {:1.2f}'.format(vx[0]) for vx in self.drift_rates]
        vx_headers_bottom = ['vx_bottom {:1.2f}'.format(vx[0]) for vx in self.drift_rates]
        # Concatenate headers
        vx_headers = vx_headers_top + vx_headers_bottom

        dicts = []

        for ker in self.kernels:
            if verbose:
                print('calibration top')
            p_top, _, vxmeans_top, vxs_top, vys_top = self.fit_calibration(ballpos_top_list, kernel=ker)
            if verbose:
                print('calibration bottom')
            p_bot, _, vxmeans_bot, vxs_bot, vys_bot = self.fit_calibration(ballpos_bottom_list, kernel=ker)

            npzf = Path(self.outputdir_cal, f'mean_velocity_{ker}_{self.index:05d}.npz')
            # Index where x- and y- rates = 0, for saving the undrifted flow fields (sanity check)
            idx0 = int(len(self.drift_rates) / 2)
            np.savez_compressed(npzf,
                                vx_top=vxs_top[idx0],
                                vy_top=vys_top[idx0],
                                vx_bot=vxs_bot[idx0],
                                vy_bot=vys_bot[idx0],
                                vxmeans_top=vxmeans_top,
                                vxmeans_bot=vxmeans_bot,
                                drift_rates=self.drift_rates,
                                p_top=p_top,
                                p_bot=p_bot,
                                index=self.index)

            # Concatenate above results in one single list and create a dictionnary with the concatenated keys
            dict_vxmeans = OrderedDict(zip(vx_headers, vxmeans_top.tolist() + vxmeans_bot.tolist()))

            dict_results = self.bt_params.copy()
            dict_results['kernel'] = ker
            dict_results['fwhm'] = self.fwhm
            dict_results['p_top_0'] = p_top[0]
            dict_results['p_top_1'] = p_top[1]
            dict_results['p_bot_0'] = p_bot[0]
            dict_results['p_bot_1'] = p_bot[1]
            dict_results.update(dict_vxmeans)
            dicts.append(dict_results)

        df_fit = pd.DataFrame(dicts)
        df_fit.to_csv(Path(self.outputdir_cal, f'param_sweep_{self.index:05d}.csv'))

        return df_fit


def full_calibration(datafiles, bt_params, cal_args, cal_opt_args, image_reader=None, make_drift_images=True,
                     reprocess_bt=True, verbose=False):
    """
    Main calibration function. It considers top-side and bottom-side to be the same, but output their results
    separately in the csv file. After multiple runs in the cluster, as many csv files are create with a unique
    index defined in the list of bt_params
    Do not use this function for a single calibration run where the top-side and bottom-side parameters need to be
    different.

    Args:
        datafiles (list or str): list of FITS files, or path to FITS cube
        bt_params (dict): balltrack parameters: rs, intsteps, ballspacing, am, dp, sigma_factor, fourier_radius
        cal_args (dict): positional arguments passed to the Calibrator class instance.
        cal_opt_args(dict): optional arguments for the Calibrator class instance
        reprocess_bt (bool): whether to reprocess balltracking over the drifted data or load existing results.
        verbose (bool): True or False for enabling / disabling verbosity

    Returns:
        index (int): identifier for the run.
    """

    if make_drift_images:
        print('Creating drift images...')
        # Create the drift images for the calibration.
        if cal_args['outputdir_cal'] is not None:
            for i, (drx, dry) in enumerate(zip(cal_args['vx_rates'], cal_args['vy_rates'])):
                create_drift_series(datafiles, drx, dry, trange=cal_args['trange'],
                                    outputdir=Path(cal_args['outputdir_cal'], f'drift_{i:02d}'),
                                    image_reader=image_reader)


    # Set the index for having a unique identifier of each run of the parameter sweep
    index = 0
    if "index" in bt_params:
        index = bt_params["index"]

    if verbose:
        print(f'Processing index {index}')

    # trange in calibration is set independently of the main balltracking run. The calibration
    # must work on its own copy, and set calibration-specific trange in `bt_params` sent to the BT class instance
    bt_params_cal = bt_params.copy()
    bt_params_cal['trange'] = cal_args['trange']
    # top-side and bottom-side parameters are considered the same.
    cal = Calibrator(bt_params_cal, **cal_args, **cal_opt_args)

    if reprocess_bt:
        if verbose:
            print('Running balltracking at all rates')
        ballpos_top_list, ballpos_bottom_list = cal.balltrack_all_rates()

    else:
        ballpos_list_file = Path(cal_args['outputdir_cal'], f'ballpos_list_{index:05d}.npz')
        if not os.path.isfile(ballpos_list_file):
            sys.exit('Missing ballpos_list_file for calibration')

        with np.load(ballpos_list_file, allow_pickle=True) as npzfile:
            ballpos_top_list = npzfile['ballpos_top_list']
            ballpos_bottom_list = npzfile['ballpos_bottom_list']

    _ = cal.calibration_run_fit(ballpos_top_list, ballpos_bottom_list, verbose=verbose)

    return index


def create_drift_series(datafiles, vx_rate, vy_rate, trange=None, outputdir=None, filter_function=None, image_reader=None, **kwargs):
    """
    Drift the image series by translating a moving reference by an input 2D velocity vector.
    The drift operates by shifting the phase of the Fourier transform that also circularly shifts the escaping pixels
    back to the other edge.

    Args:
        datafiles (list): list of image file paths to drift
        vx_rate (float): signed value for the velocity drift on the x-direction.
        vy_rate (float): signed value for the velocity drift on the y-direction.
        outputdir (Path): output directory where the drifted images are saved.
        filter_function (function): optional filter to apply to the image

    Returns:
        None
    """

    if image_reader is None:
        image_reader = fitstools.fitsread

    if trange is None:
        trange = range(0, len(datafiles))
    else:
        trange = range(*trange)

    for i in trange:
        # Load the image from disk one by one
        image_to_drift = image_reader(datafiles, tslice=i, **kwargs)

        if (vx_rate != 0) or (vy_rate != 0):
            dx = vx_rate * i
            dy = vy_rate * i
            drifted_image = filters.translate_by_phase_shift(image_to_drift, dx, dy)
        else:
            drifted_image = image_to_drift

        if filter_function is not None:
            drifted_image = filter_function(drifted_image)


        if outputdir is not None:
            if not outputdir.exists():
                os.makedirs(outputdir, exist_ok=True)
                print('created outputdir: ', outputdir)
            filepath = str(Path(outputdir, f'drifted_{i:02d}.fits'))
            fitstools.writefits(drifted_image.astype(np.float32), filepath)

    return None


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


def make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, fwhm,
                        trange=None, kernel='gaussian', header=None, outputdir=None, generate_lanes=False, **kwargs):
    """
    Create the dense flow fields, on a regular, cartesian grid, from the ball positions.

    Args:
        ballpos_top(np.ndarray): 3D cube of ball position [xy(z), nb of balls, nb of frames] for top-side tracking
        ballpos_bottom: 3D cube of ball position [xy(z), nb of balls, nb of frames] for bottom-side tracking
        cal_top (float): Calibration multiplier for the top-side tracking
        cal_bottom (float): Calibration multiplier for the bottom-side tracking
        dims (list or tuple): image dimension (width, height)
        fwhm (int): gaussian FWHM smoothing or width of boxcar average
        trange (list): list of (start frame, end frame) over which to integrate the ball positions
        kernel (str): Spatial smoothing kernel, either 'gaussian' or 'boxcar'
        header (dict): header information, usually from the fits file of the middle of the series
        outputdir (Path or str): where to save the output npz files of the vx and vy flow fields
        generate_lanes (bool): whether to generate the map of the supergranular boundaries ("lanes" for short)
        **kwargs: Additional keyword arguments to be passed to make_lanes()

    Returns:
        Dense Vx and Vy flow fields

    """
    if trange is None:
        trange = [0, ballpos_top.shape[-1]-1]

    vx_top, vy_top, wplane_top = make_velocity_from_tracks(ballpos_top, dims, trange, fwhm, kernel)
    vx_bottom, vy_bottom, wplane_bottom = make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm, kernel)

    vx_top *= cal_top
    vy_top *= cal_top
    vx_bottom *= cal_bottom
    vy_bottom *= cal_bottom

    vx = 0.5 * (vx_top + vx_bottom)
    vy = 0.5 * (vy_top + vy_bottom)

    if generate_lanes:
        lanes = make_lanes(vx, vy, **kwargs)
    else:
        lanes = None

    if outputdir is not None:
        if header is not None:
            header['BSCALE'] = 1
            header['BZERO'] = 0
            header['BITPIX'] = -32

        np.savez_compressed(Path(outputdir, f'vxy_{kernel}_fwhm{fwhm}_t{trange[0]}_{trange[1]}.npz'),
                            vx=vx.astype(np.float32), vy=vy.astype(np.float32), header=header, lanes=lanes)

    return vx, vy, lanes, header


def make_euler_velocity_series(tranges, *args, headers=None, **kwargs):

    """
    Create a series of dense flow fields and supergranular maps over a timeline given by 'tranges'. The pairs of time range can overlap for smoother
    visualizations. For example [(0, 50), (25, 75), (50, 100), ...].

    Args:
        tranges(list or tuple): list of 2-element tuple as defined by "trange" in make_euler_velocity()
        *args: positional arguments to be passed to make_euler_velocity()
        headers (list): list of headers (dict) from the fits files
        **kwargs: Additional keyword arguments to be passed to make_lanes()

    Returns:
        Dense Vx and Vy flow fields

    """

    vxl = []
    vyl = []
    lanes_list = []
    header = None
    for i, trange in enumerate(tranges):
        # Generate the flow fields
        if headers is not None:
            header = headers[i]
        vx, vy, lanes, header = make_euler_velocity(*args, trange=trange, header=header, **kwargs)

        vxl.append(vx)
        vyl.append(vy)
        lanes_list.append(lanes)

    # Make a running average of the series of supergranular maps
    generate_lanes = kwargs.get('generate_lanes', False)
    run_avg_lanes = None
    if generate_lanes:
        run_avg_lanes = np.array(lanes_list).mean(axis=0)

    outputdir = kwargs.get('outputdir', None)
    if outputdir is not None:
        run_avg_header = None

        if headers is not None:
            run_avg_header = headers[len(headers) // 2]

        np.savez_compressed(Path(outputdir, f"vxy_{kwargs.get('kernel', 'gaussian')}_fwhm{args[5]}_run_avg.npz"),
                            run_avg_lanes=run_avg_lanes, header=run_avg_header)

    return vxl, vyl, lanes_list, run_avg_lanes


def make_lanes(vx, vy, nsteps=40, maxstep=1):
    """
        Create the scalar map of the supergranular lanes

    Args:
        vx (np.ndarray): x-component of the velocity field
        vy (np.ndarray): y-component of the velocity field
        nsteps (int): Nb of integration steps. The higher the sharper, at the cost of missing small-scale convective cells
        maxstep (int): Accelerator factor in case of oversampled data, will make the code converge faster but with the
        risk losing resolution. Leave it at 1 if computing time is not an issue.

    Returns:
        2D array of the map of the convective cells. High intensity values are the boundaries of the cells

    """
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

    vblank = np.zeros([dims[0]+2*maxstep, dims[1]+2*maxstep], dtype=DTYPE)
    vx2 = vblank.copy()
    vy2 = vblank.copy()
    vx2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vxng.astype(DTYPE)
    vy2[maxstep : dims[0] + maxstep, maxstep :dims[1] + maxstep] = vyng.astype(DTYPE)

    vx2 *= -1
    vy2 *= -1

    x0, y0 = np.meshgrid( maxstep + np.arange(dims[1]) , maxstep + np.arange(dims[0]))
    xold = x0.flatten().astype(DTYPE)
    yold = y0.flatten().astype(DTYPE)

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


def calibrate_flows(datafiles, calibration_file, balltrack_dir, maps_params):
    """
    Create the Euler, dense flow maps from the ball positions.
    Each map is spatially smoothed and time-averaged over the user-set parameters. Another flow map, averaged over the
    whole tracking time, is also created.

    Args:
        datafiles (Path or str): input FITS files
        calibration_file (Path or str): path to csv file containing the fit parameters of the calibration
        balltrack_dir (Path or str): directory where the
        maps_params:

    Returns:

    """
    # Make calibrated euler flows
    df_fit = (pd.read_csv(calibration_file).query(f"kernel=='{maps_params['kernel']}'"))
    cal_top = df_fit['p_top_0'].values[0]
    cal_bottom = df_fit['p_bot_0'].values[0]

    # Load ball posisions arrays computed in an earlier balltrack run
    arr = np.load(Path(balltrack_dir, 'ballpos.npz'))
    ballpos_top, ballpos_bottom = [arr['ballpos_top'], arr['ballpos_bottom']]

    # Number of images used in balltracking
    nimgs = ballpos_top.shape[-1]
    # Create the list of pairs of (start, end) times for defining the timeline of time-averaged flows
    tranges = [[i, i + maps_params['navg']] for i in range(0, nimgs - maps_params['navg'], maps_params['dt'])]
    # Get FITS headers
    headers=None
    avg_header = None
    if maps_params['use_headers']:
        headers = [fits.getheader(datafiles[tr[0] + maps_params['navg']//2], maps_params['hdu_n']) for tr in tranges]
        avg_header = headers[len(headers) // 2]

    # Make average the flow field over the entire integration time
    avg_trange = [0, nimgs-1]
    vx_avg, vy_avg, lanes_avg, avg_header = make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom,
                                                                maps_params['im_dims'], maps_params['fwhm'],
                                                                trange = avg_trange,
                                                                header=avg_header,
                                                                outputdir=balltrack_dir,
                                                                generate_lanes=maps_params['generate_lanes'],
                                                                nsteps=maps_params['nsteps'])
    v_avg_dict = {
        'vx_avg': vx_avg,
        'vy_avg': vy_avg,
        'lanes_avg': lanes_avg,
        'avg_header': avg_header
    }

    # Make average flow fields every time steps of tranges.
    vxs, vys, lanes_list, run_avg_lanes = make_euler_velocity_series(tranges, ballpos_top, ballpos_bottom, cal_top,
                                                                     cal_bottom,
                                                                     maps_params['im_dims'], maps_params['fwhm'],
                                                                     headers=headers,
                                                                     outputdir=balltrack_dir,
                                                                     generate_lanes=maps_params['generate_lanes'],
                                                                     nsteps=maps_params['nsteps'])

    v_series_dict = {
        'vxs': vxs,
        'vys': vys,
        'lanes_list': lanes_list,
        'run_avg_lanes': run_avg_lanes
    }


    return v_series_dict, v_avg_dict

def meshgrid_params_to_list(param_dict):
    """
    Generate a list of a flattened mesh of parameters. The mesh can have more than 2 dimensions.

    Args:
        param_dict (dict): dictionnary of parameters. At least one value must be a list of more than 1 scalar.

    Returns:
        List of flattened mesh-gridded parameters
    """
    param_list = list(param_dict.values())
    mesh = np.meshgrid(*param_list, indexing='ij')
    list_ravel = [np.ravel(m) for m in mesh]
    args_list = [list(a) for a in zip(*list_ravel)]
    return args_list


def get_bt_params_list(param_dict):
    """
        Generate a list of input balltrack paremeters for doing a parameter sweep from a grid
    Args:
        param_dict (dict): dictionnary of parameters. At least one value must be a list of more than 1 scalar.

    Returns:
        Mesh-gridded list of balltracking input dictionnaries.
    """
    param_mesh_list = meshgrid_params_to_list(param_dict)
    bt_params_list = []
    for i, p_list in enumerate(param_mesh_list):
        bt_params = OrderedDict()
        for n, key in enumerate(param_dict.keys()):
            bt_params[key] = p_list[n]
        bt_params['index'] = i
        bt_params_list.append(bt_params)
    return bt_params_list

