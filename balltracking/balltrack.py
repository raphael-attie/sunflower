"""
This module hosts all the necessary functions to run balltracking.
A main program should execute "balltrack_all()".
"""

import sys, os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from numpy import pi, cos, sin
from scipy.ndimage.filters import gaussian_filter
import cython_modules.interp as cinterp
import filters
import fitstools
from multiprocessing import Pool
from functools import partial

DTYPE = np.float32
class BT:

    def __init__(self, nt, rs, dp, sigma_factor=1, mode='top', direction='forward', datafiles=None, data=None):

        """
        This is the class hosting all the parameters and intermediate results of balltracking.

        :param nt: number of frames to process
        :param rs: balls radius
        :param dp: characteristic depth
        :param sigma_factor: multiplier to the standard deviation
        :param mode: string that determines which side of the data surface to track. Either 'top' or 'bottom'.
        :param direction: string that determines whether we track 'forward' or 'backward' in time.
        :param datafiles: path to data fits cube or to series of fits files.
        :param data: numpy data cube whose dimensions are (y-axis, x-axis, time)
        """

        self.datafiles = datafiles
        self.data = data
        self.direction = direction
        self.nt = nt

        # Get a sample. 1st of the series in forward direction. last of the series in backward direction.
        # TODO: It could be wiser to get some average value between 1st, middle, and last of the series?
        if self.data is None:
            if self.direction == 'forward':
                self.sample = fitstools.fitsread(self.datafiles, tslice=0).astype(np.float32)
            elif self.direction == 'backward':
                self.sample = fitstools.fitsread(self.datafiles, tslice=self.nt - 1).astype(np.float32)
        else:
            if self.direction == 'forward':
                self.sample = self.data[:,:,0]
            elif self.direction == 'backward':
                self.sample = self.data[:,:,-1]

        self.nx = int(self.sample.shape[1])
        self.ny = int(self.sample.shape[0])

        self.intsteps = 3
        self.rs = rs
        self.dp = dp
        self.ballspacing = int(2 * rs)
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
        # Initialize horizontal positions
        self.xstart, self.ystart = initialize_mesh(self)
        self.zstart = np.zeros(self.xstart.shape, dtype=np.float32)
        self.nballs = self.xstart.size
        # Initialize the array of the lifetime (age) of the balls
        self.balls_age = np.ones([self.nballs], dtype=np.uint32)

        # Coarse grid
        # Matlab: finegrid=zeros(ceil(BT.nr/BT.finegridspacing),ceil(BT.nc/BT.finegridspacing));
        self.coarse_grid = np.zeros([np.ceil(self.ny/self.ballspacing).astype(int), np.ceil(self.nx/self.ballspacing).astype(int)], dtype=np.uint32)
        # Dimensions of the coarse grid
        self.nyc, self.nxc = self.coarse_grid.shape

        # Acceleration factor (used to be 0.6 in Potts implementation)
        self.am = 1.0
        # Force scaling factor
        self.k_force = self.am / (self.dp**2 * pi * self.rs**2)
        # Damping
        self.td = 1.0
        self.zdamping = 0.3
        self.e_td = np.exp(-1/self.td)
        self.e_tdz = np.exp(-1/self.zdamping)

        # Rescaling factor for the standard deviation
        self.sigma_factor = sigma_factor
        self.mean = 0
        self.sigma = 0

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
        self.ds    = np.zeros([self.bcols.shape[0]], dtype=DTYPE)
        # Initialize deepest height at a which ball can fall down. Typically it will be set to a multiple of -surface.std().
        self.min_ds = -5
        # Mask of bad balls
        self.bad_balls_mask = np.zeros(self.nballs, dtype=bool)
        # Mask of valid balls
        self.new_valid_balls_mask = np.ones(self.nballs, dtype=bool)

        # Mode and direction
        self.mode = mode
        self.direction = direction


    def initialize(self):

        ### Calculate offset (mean) and standard deviation from  a valid surface ####
        # First, filter image to focus on the granulation
        # Sigma-clip outlyers (e.g. sunspots)
        # Get mean and standard deviation from the masked array, not affected by bad invalid values (sunspot, ...)
        # Generate the data surface from the image with the masked mean and sigma
        surface, mean, sigma = prep_data2(self.sample, self.sigma_factor)
        self.mean = mean
        self.sigma = sigma
        # Initialize the height of the ball. Only possible if the data surface is given.
        self.pos[0, :] = self.xstart.flatten()
        self.pos[1, :] = self.ystart.flatten()
        self.zstart = put_balls_on_surface(surface, self.xstart.ravel(), self.ystart.ravel(), self.rs, self.dp)
        self.pos[2, :] = self.zstart.copy()
        # Setup the coarse grid: populate edges to avoid "leaks" of balls ~ balls falling off.
        # Although can't remember if that was actually ever used in the Matlab implementation
        self.coarse_grid[0,:]   = 1
        self.coarse_grid[:, 0]  = 1
        self.coarse_grid[-1,:]  = 1
        self.coarse_grid[:, -1] = 1
        return


    def track(self):

        # Outer loop goes over the data frames.
        # If data is a fits cube, we just access a slice of it

        # Initialize

        self.initialize()

        for n in range(self.nt):

            if self.direction == 'forward':
                if self.data is None:
                    image = fitstools.fitsread(self.datafiles, tslice=n).astype(np.float32)
                else:
                    image = self.data[:, :, n]
            elif self.direction == 'backward':
                if self.data is None:
                    image = fitstools.fitsread(self.datafiles, tslice=self.nt - 1 - n).astype(np.float32)
                else:
                    image = self.data[:, :, self.nt - 1 - n]

            surface = prep_data(image, self.mean, self.sigma, sigma_factor=self.sigma_factor)

            if self.mode == 'bottom':
                surface = -surface

            # The current position "pos" and velocity "vel" are attributes of bt.
            # They are integrated in place.
            for _ in range(self.intsteps):
                integrate_motion(self, surface)

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
            try:
                _, _ = self.replace_bad_balls(surface)
            except sys.SystemExit:
                print('replace_bad_balls failed.')

            self.balls_age_t[:, n] = self.balls_age.copy()

        if self.direction == 'backward':
            # Flip timeline for backward tracking
            self.ballpos = np.flip(self.ballpos, 2)
            self.ballvel = np.flip(self.ballvel, 2)


    def get_bad_balls(self):
        # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
        # and https://stackoverflow.com/questions/36863404/accumulate-constant-value-in-numpy-array
        # xpos0, ypos0, zpos0 = bt.pos

        # Bad balls are flagged with -1 in the pos array. They will be excluded from the comparisons below
        bad_balls1_mask = get_outliers(self)

        # Ignore these bad balls in the arrays and enforce continuity principle
        valid_balls = np.logical_not(bad_balls1_mask)
        valid_balls_idx = np.nonzero(valid_balls)[0]
        # valid_balls = np.logical_not(bad_balls1_mask)
        # Map back to original balls indices
        # valid_balls_idx = np.nonzero(valid_balls)[0]

        xpos, ypos, zpos = self.pos[:, valid_balls]
        balls_age = self.balls_age[valid_balls]

        # Get the 1D position on the coarse grid, clipped to the edges of that grid.
        _, _, coarse_pos = coarse_grid_pos(self, xpos, ypos)

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


    def replace_bad_balls(self, surface):

        nbadballs = self.bad_balls_mask.sum()

        # Get the mask of the valid balls that we are going to keep
        valid_balls_mask = np.logical_not(self.bad_balls_mask)
        # Work with with views on coordinate and velocity arrays of valid balls for clarity (instead of working with pos[:, :, valid_balls_mask] directly)
        xpos, ypos, zpos = self.pos[:, valid_balls_mask]

        # Get the 1D position on the coarse grid, clipped to the edges of that grid.
        _, _, coarse_pos_idx = coarse_grid_pos(self, xpos, ypos)

        # Set these positions on the coarse grid as filled. Remember that to avoid putting new balls on the edge, the coarse_grid is pre-filled with ones at its edges
        # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
        coarse_grid = self.coarse_grid.copy()
        coarse_grid.ravel()[coarse_pos_idx] = 1
        y0, x0 = np.where(coarse_grid == 0)
        nemptycells = x0.size

        if nemptycells <= nbadballs:

            # Get the indices of bad balls in order to assign them with new positions
            bad_balls_idx = np.nonzero(self.bad_balls_mask)[0][0:nemptycells]
            # Relocate the previously flagged balls in all the empty cells
            # IMPROVEMENT with respect to Matlab version:
            # y0, x0 above are from np.where => integer values! All values in xnew, ynew are integers.
            # => There is no need to interpolate in put_ball_on_surface!
            xnew = self.ballspacing * x0.astype(np.float32)#+ bt.rs
            ynew = self.ballspacing * y0.astype(np.float32)#+ bt.rs
            znew = put_balls_on_surface(surface, xnew, ynew, self.rs, self.dp)

            self.pos[0, bad_balls_idx] = xnew
            self.pos[1, bad_balls_idx] = ynew
            self.pos[2, bad_balls_idx] = znew
            # Reset the velocity and age at these new positions
            self.vel[:, bad_balls_idx] = 0
            self.balls_age[bad_balls_idx] = 0

            # Rest of bad balls
            bad_balls_remaining_idx = np.nonzero(self.bad_balls_mask)[0][nemptycells:nbadballs]

            # Update the list valid balls in bt.new_valid_balls_mask
            new_valid_balls_mask = np.ones([self.nballs], dtype = bool)
            new_valid_balls_mask[bad_balls_remaining_idx] = False
            self.new_valid_balls_mask = new_valid_balls_mask

        else:
            # This case means the number of bad balls available for relocation is smaller than the number of empty cells where they can be relocated.
            # This means the continuity principle is not satisfied and needs investigation.
            raise SystemExit('The number of empy cells is greater than the number of bad balls.')

        return xnew, ynew



def track_instance(mode_direction, nframes, rs, dp, sigma_factor, datafile=None, data=None):

    """
    Run balltracking on a given tuple (mode,direction). This routine must be executed with 4 of these pairs for
    a complete flow tracking.

    :param mode_direction: tracking mode and tracking direction, e.g. ('top', 'forward').
    :param nframes: number of frames to track in the image series
    :param rs: ball radius
    :param dp: charateristic depth
    :param sigma_factor: multiplier to the standard deviation
    :param datafile: path to data cube file or series of files.
    :param data: for calibration only. numpy data arrays of drifting data surface.
    :return:
    """
    bt_instance = BT(nframes, rs, dp, sigma_factor=sigma_factor, mode=mode_direction[0], direction=mode_direction[1], datafiles=datafile, data=data)
    bt_instance.track()

    return bt_instance


def balltrack_all(nt, rs, dp, sigma_factor, datafile, outputdir, ncores=1):

    """
    Run the 4 tracking on the 4 pairs of (mode, direction). Can be executed in parallel over 4 workers.

    :param nt: number of frames to track in the image series
    :param rs: ball radius
    :param dp: charateristic depth
    :param sigma_factor: multiplier to the standard deviation
    :param datafile: path to data cube file or series of files.
    :param outputdir: output directory to write the intermediate ball tracks as .npz files
    :param ncores: number of cores to use for running the 4 modes/directions in parallel.
    Default is 1 for sequential processing. There can up to 4 workers for these parallel tasks.
    :return:
    """
    # Get a BT instance with the above parameters
    mode_direction_list = (('top','forward'),
                           ('top', 'backward'),
                           ('bottom', 'forward'),
                           ('bottom', 'backward'))

    partial_track = partial(track_instance, nframes=nt, rs=rs, dp=dp, sigma_factor=sigma_factor, datafile=datafile)

    # there can only 1 to 4 workers. 1 means no parallelization.
    ncores = max(min(ncores, 4), 1)
    if ncores == 1:
        bt_tf, bt_tb, bt_bf, bt_bb = list(map(partial_track, mode_direction_list))
    else:
        with Pool(processes=ncores) as pool:
            bt_tf, bt_tb, bt_bf, bt_bb = pool.map(partial_track, mode_direction_list)


    ballpos_top = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)
    ballpos_bottom = np.concatenate((bt_bf.ballpos, bt_bb.ballpos), axis=1)


    np.save(os.path.join(outputdir,'ballpos_top.npy'), ballpos_top)
    np.save(os.path.join(outputdir, 'ballpos_bottom.npy'), ballpos_bottom)


    return ballpos_top, ballpos_bottom


def calculate_invalid_mask(data):
    """
    Create a mask where invalid values are where the data values are too small to be coming from granulation signal
    The threshold is 5 times the standard deviation below the mean

    :param data: input 2D array (image or filtered image)
    :return: numpy mask to be used with numpy.masked_array
    """
    mean = data.mean()
    sigma = data.std()
    mind = mean - 5 * sigma

    return ma.masked_less(data, mind)


def filter_image(image):
    """
    Filter the image to enhance granulation signal

    :param image: input image e.g continuum intensity from SDO/HMI (2D array)
    :return: fdata: filtered data (2D array)
    """

    ffilter_hpf = filters.han2d_bandpass(image.shape[0], 0, 5)
    fdata = filters.ffilter_image(image, ffilter_hpf)

    return fdata


def rescale_frame(data, offset, norm_factor):
    """
    Rescales the images for Balltracking. It subtract the mean of the data and divide by a scaling factor.
    For balltracking, this scaling factor shall be the standard deviation of the whole data series.
    See http://mathworld.wolfram.com/HanningFunction.html

    :param data: 2D frame (e.g: continuum image or magnetogram)
    :param offset: scalar that offsets the data. Typically the mean of the data or of masked data
    :param norm_factor: scalar that divides the data. Typically a multiple of the standard deviation.
    :return: rescaled_data: rescaled image.
    """
    rescaled_data = data - offset
    rescaled_data = rescaled_data / norm_factor

    return rescaled_data


def prep_data(image, mean, sigma, sigma_factor=1):
    """
    The image is filtered to enhance the granulation pattern, and rescaled into a data surface
    where the resulting standard deviation is equal to the sigma_factor, typically 1 or 2 depending
    on the statistical properties of the image series. The data intensity is centered around the mean.

    :param image: input image e.g continuum intensity from SDO/HMI (2D array)
    :param mean: offset for rescaling the image as a data surface (i.e. 3D mesh).
    :param sigma: standard deviation of the 1st filtered data of the series
    :param sigma_factor: Multiplier to the standard deviation (scalar)
    :return: data surface (2D array)
    """

    # Filter the image to isolate granulation
    fdata = filter_image(image)
    # Rescale image to a data surface
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(np.float32)

    return surface


def prep_data2(image, sigma_factor=1):
    """
    Similar to prep_data (see prep_data). The difference is that sigma is calculated from the input data and not from
    a user input. This is implemented as follows:
        - First, filter image to focus on the granulation
        - Sigma-clip outlyers (e.g. sunspots)
        - Get mean and standard deviation from the masked array, not affected by bad invalid values (sunspot, ...)
        - Generate the data surface from the image with the masked mean and sigma

    :param image: input image e.g continuum intensity from SDO/HMI (2D array)
    :param sigma_factor: Multiplier to the standard deviation (scalar)
    :return: data surface (2D array)
    """
    # Filter the image to isolate granulation
    fdata = filter_image(image)
    masked_fdata = calculate_invalid_mask(fdata)
    mean = masked_fdata.mean()
    sigma = masked_fdata.std()
    # Rescale image to a data surface
    surface = rescale_frame(fdata, mean, sigma_factor * sigma).astype(np.float32)

    return surface, mean, sigma


def coarse_grid_pos(bt, x, y):

    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.uint32(np.clip(np.floor(x / bt.ballspacing), 0, bt.nxc-1))
    ycoarse = np.uint32(np.clip(np.floor(y / bt.ballspacing), 0, bt.nyc-1))
    # Convert to linear (1D) indices. One index per ball
    coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), bt.coarse_grid.shape)
    return xcoarse, ycoarse, coarse_idx


def fill_coarse_grid(bt, x, y):
    """
    Fill coarse_grid as a chess board: the positions in the original grid are mapped to the coarse grid points
    And each of the mapped grid points are incremented by 1.

    :param x: x-coordinate in original grid
    :param y: y-coordinate in original grid
    :return: coarse grid filled like a chess board.
    """
    # Favor clarity and stay in 2D coordinates instead of linear indices.
    xcoarse, ycoarse, _ = coarse_grid_pos(bt, x, y)
    chess_board = bt.coarse_grid.copy()
    np.add.at(chess_board, (ycoarse, xcoarse), 1)
    return chess_board


def initialize_mesh(bt):
    # Initial horizontal (x,y) positions
    x_start_points = np.arange(bt.ballspacing, bt.nx - bt.ballspacing + 1, bt.ballspacing, dtype=np.float32)
    y_start_points = np.arange(bt.ballspacing, bt.ny - bt.ballspacing + 1, bt.ballspacing, dtype=np.float32)
    xstart, ystart = np.meshgrid(x_start_points, y_start_points)
    return xstart, ystart


def put_balls_on_surface(surface, x, y, rs, dp):
    if x.ndim !=1 or y.ndim !=1:
        sys.exit("Input coordinates have incorrect dimensions. x and y must be 1D numpy arrays")

    #z = np.zeros([x.shape[0]], dtype=np.float32)
    #z = bilin_interp_f(surface, x, y) +  rs*(1-dp/2)

    # THERE IS NO NEED TO INTERPOLATE!!! Here x and y will always be integer coordinates mapping perfectly to pixel
    # coordinates.
    #z = cinterp.cbilin_interp1(surface, x, y)


    z = surface[y.astype(np.int32), x.astype(np.int32)]
    z += rs * (1 - dp / 2)
    return z


def integrate_motion(bt, surface, return_copies=False):

    # Get the valid balls & unpack vector components for better readability
    #print("Get the valid balls")
    xt, yt, zt = bt.pos[:, bt.new_valid_balls_mask]
    vxt, vyt, vzt = bt.vel[:, bt.new_valid_balls_mask]


    # Update the balls grids with current positions
    # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
    # Clipping balls rows and cols. Stay away from borders by 1 px because of the interpolation scheme
    #print("Clipping balls rows and cols.")
    bcols = np.clip(bt.bcols + xt, 1, bt.nx - 2)
    brows = np.clip(bt.brows + yt, 1, bt.ny - 2)


    # "ds" stands for "data surface"
    #ds = surface[np.round(brows).astype(np.int), np.round(bcols).astype(np.int)]
    #print("Interpolating data surface")
    # These interpolations gives identical results in Matlab
    #ds = cinterp.cbilin_interp2(surface, bcols, brows)
    ds = cinterp.bilin_interp2f(surface, bcols, brows)

    #print("Compute force vector")
    fxt, fyt, fzt = compute_force(bt, brows, bcols, xt, yt, zt, ds)

    # Integrate velocity
    vxt += fxt
    vyt += fyt
    vzt += fzt

    # Integrate position including effect of a damped velocity
    # Damping is added arbitrarily for the stability of the code.
    #print("Integrate positions")
    xt += vxt * bt.td * (1 - bt.e_td)
    yt += vyt * bt.td * (1 - bt.e_td)
    zt += vzt * bt.zdamping * (1 - bt.e_tdz)

    bt.pos[0, bt.new_valid_balls_mask] = xt
    bt.pos[1, bt.new_valid_balls_mask] = yt
    bt.pos[2, bt.new_valid_balls_mask] = zt

    #print("Update velocity vector")
    # Update the velocity with the damping used above
    bt.vel[0, bt.new_valid_balls_mask] = vxt * bt.e_td
    bt.vel[1, bt.new_valid_balls_mask] = vyt * bt.e_td
    bt.vel[2, bt.new_valid_balls_mask] = vzt * bt.e_tdz

    if return_copies:
        force = np.array([fxt, fyt, fzt])
        return bt.pos.copy(), bt.vel.copy(), force


def compute_force(bt, brows, bcols, xt, yt, zt, ds):
    # TODO: must get rid of them also in bcols, brows, zt, ...
    # So we need to work out the correct indices. r is 2D but xt, yt are 1D
    # So we define an intermediate variable to calculate the delta, since
    # the minus operation is propagated to either dimensions.
    delta_x = xt - bcols
    delta_y = yt - brows
    delta_z = ds - zt

    r = np.sqrt(delta_x** 2 + delta_y**2 + delta_z**2)
    r_h = np.sqrt(delta_x** 2 + delta_y**2)
    # Singularity at r = 0. Need to get rid of them.
    # & Force that are beyond the radius must be set to zero
    rmask = np.logical_or(r == 0, r > bt.rs)
    rmask_h = np.logical_or(r == 0, r_h > bt.rs)

    rm = np.ma.masked_array(r, mask=rmask)
    rm_h = np.ma.masked_array(r, mask=rmask_h)

    f = bt.k_force * (rm - bt.rs)

    # Calculate each force vector component

    # NEW GREAT STUFF
    # f_h = bt.k_force * (rm_h - bt.rs)
    # fxtm = np.sum(np.abs(f_h) * delta_x / rm_h, 0)
    # fytm = np.sum(np.abs(f_h) * delta_y / rm_h, 0)
    # fztm = -np.sum(f * np.abs(delta_z) / rm, 0) - bt.am

    # OLD BAD STUFF
    fxtm = -np.sum(f * delta_x / rm, 0)
    fytm = -np.sum(f * delta_y / rm, 0)
    # Buoyancy must stay oriented upward. Used to be signed, but that caused more lost balls without other advantage
    fztm = -np.sum(f * np.abs(delta_z) / rm, 0) - bt.am


    # Return the plain numpy array instead of the masked array. The fill value will ensure that in case of
    # a sum performed on an entirely masked array (all elements ignored), we don't end up with the default filled value
    # for the "empty" component.
    fxt = np.ma.filled(fxtm, fill_value=0)
    fyt = np.ma.filled(fytm, fill_value=0)
    fzt = np.ma.filled(fztm, fill_value=0)

    return fxt, fyt, fzt


def ravel_index(x, dims):
    i = 0
    for dim, j in zip(dims, x):
        i *= dim
        i += j
    return int(i)


def get_bad_balls(bt):
    # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
    # and https://stackoverflow.com/questions/36863404/accumulate-constant-value-in-numpy-array
    #xpos0, ypos0, zpos0 = bt.pos

    # Bad balls are flagged with -1 in the pos array. They will be excluded from the comparisons below
    bad_balls1_mask = get_outliers(bt)

    # Ignore these bad balls in the arrays and enforce continuity principle
    valid_balls = np.logical_not(bad_balls1_mask)
    valid_balls_idx = np.nonzero(valid_balls)[0]
    #valid_balls = np.logical_not(bad_balls1_mask)
    # Map back to original balls indices
    #valid_balls_idx = np.nonzero(valid_balls)[0]

    xpos, ypos, zpos = bt.pos[:, valid_balls]
    balls_age = bt.balls_age[valid_balls]

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
    unique_oldest_balls = valid_balls_idx[ sidx[ np.r_[np.flatnonzero(coarse_pos[1:] != coarse_pos[:-1]), -1] ] ]

    # Now the point is to have a mask or list of balls at overpopulated cells.
    # They are simply the ones not listed by unique_oldest_balls
    bad_full_balls = np.ones([bt.nballs], dtype=bool)
    bad_full_balls[unique_oldest_balls] = False
    # The bad balls are not just the ones in overpopulated cells, there's also the ones off edges & sunk balls
    #bad_balls_mask = np.logical_or(bad_balls1, bad_full_balls)
    bt.bad_balls_mask = np.logical_or(bad_balls1_mask, bad_full_balls)

    return


def get_outliers(bt):

    x, y, z = bt.pos
    sunk = z < bt.min_ds
    off_edges_mask = get_off_edges(bt, x, y)
    outlier_mask = np.logical_or(sunk, off_edges_mask)

    return outlier_mask


def get_off_edges(bt, x,y):

    off_edge_left = x - bt.rs < 0
    off_edge_right = x + bt.rs > bt.nx - 1
    off_edge_bottom = y - bt.rs < 0
    off_edge_top = y + bt.rs > bt.ny - 1
    off_edges_mask = np.logical_or.reduce(np.array((off_edge_left, off_edge_right, off_edge_bottom, off_edge_top)))

    return off_edges_mask



def replace_bad_balls(surface, bt):

    nbadballs = bt.bad_balls_mask.sum()

    # Get the mask of the valid balls that we are going to keep
    valid_balls_mask = np.logical_not(bt.bad_balls_mask)
    # Work with with views on coordinate and velocity arrays of valid balls for clarity (instead of working with pos[:, :, valid_balls_mask] directly)
    xpos, ypos, zpos = bt.pos[:, valid_balls_mask]

    # Get the 1D position on the coarse grid, clipped to the edges of that grid.
    _, _, coarse_pos_idx = coarse_grid_pos(bt, xpos, ypos)

    # Set these positions on the coarse grid as filled. Remember that to avoid putting new balls on the edge, the coarse_grid is pre-filled with ones at its edges
    # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
    coarse_grid = bt.coarse_grid.copy()
    coarse_grid.ravel()[coarse_pos_idx] = 1
    y0, x0 = np.where(coarse_grid == 0)
    nemptycells = x0.size

    if nemptycells <= nbadballs:

        # Get the indices of bad balls in order to assign them with new positions
        bad_balls_idx = np.nonzero(bt.bad_balls_mask)[0][0:nemptycells]
        # Relocate the previously flagged balls in all the empty cells
        # IMPROVEMENT with respect to Matlab version:
        # y0, x0 above are from np.where => integer values! All values in xnew, ynew are integers.
        # => There is no need to interpolate in put_ball_on_surface!
        xnew = bt.ballspacing * x0.astype(np.float32)#+ bt.rs
        ynew = bt.ballspacing * y0.astype(np.float32)#+ bt.rs
        znew = put_balls_on_surface(surface, xnew, ynew, bt.rs, bt.dp)

        bt.pos[0, bad_balls_idx] = xnew
        bt.pos[1, bad_balls_idx] = ynew
        bt.pos[2, bad_balls_idx] = znew
        # Reset the velocity and age at these new positions
        bt.vel[:, bad_balls_idx] = 0
        bt.balls_age[bad_balls_idx] = 0

        # Rest of bad balls
        bad_balls_remaining_idx = np.nonzero(bt.bad_balls_mask)[0][nemptycells:nbadballs]

        # Update the list valid balls in bt.new_valid_balls_mask
        new_valid_balls_mask = np.ones([bt.nballs], dtype = bool)
        new_valid_balls_mask[bad_balls_remaining_idx] = False
        bt.new_valid_balls_mask = new_valid_balls_mask

    else:
        # This case means the number of bad balls available for relocation is smaller than the number of empty cells where they can be relocated.
        # This means the continuity principle is not satisfied and needs investigation.
        raise SystemExit('The number of empy cells is greater than the number of bad balls.')

    return xnew, ynew



def replace_bad_balls1(pos, bt):
    # Get the position on the finegrid

    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.round(pos[0, :] / bt.ballspacing).clip(0, bt.nxc).astype(np.uint32)
    ycoarse = np.round(pos[1, :] / bt.ballspacing).clip(0, bt.nyc).astype(np.uint32)

    bt.coarse_grid[ycoarse, xcoarse] = 1

    return xcoarse, ycoarse


def replace_bad_balls2(pos, bt):
    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.round(pos[0, :] / bt.ballspacing).clip(0, bt.nxc).astype(np.uint32)
    ycoarse = np.round(pos[1, :] / bt.ballspacing).clip(0, bt.nyc).astype(np.uint32)

    # Convert to linear (1D) indices
    idx = np.ravel_multi_index((ycoarse, xcoarse), bt.coarse_grid.shape)
    bt.coarse_grid.flat[idx] = 1

    # x = np.bincount(idx, minlength=bt.coarse_grid.size)
    # bt.coarse_grid.flat += x

    return xcoarse, ycoarse


def make_velocity_from_tracks(ballpos, dims, trange, fwhm):
    """
    Calculate the velocity field, i.e, differentiate the position to get the velocity in Lagrange ref. frame and
    convert to Euler ref. frame.

    :param ballpos: array of ball positions. Dimensions are [xyz, ball number, time]
    :param dims: dimensions of the images used for the tracking.
    :param trange: sequence of [1st index, last index[ on time axis over which the flows are averaged
    :param fwhm: full width at half maximum for the spatial gaussian smoothing.
    :return: 2 components of the flow fields and the weight plane.
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
    # Exclude the -1 flagged positions using a mask. Could there be NaNs left here?
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
    vx_euler = gaussian_filter(vx_euler, sigma=sigma, order=0)
    vy_euler = gaussian_filter(vy_euler, sigma=sigma, order=0)
    wplane   = gaussian_filter(wplane, sigma=sigma, order=0)

    vx_euler /= wplane
    vy_euler /= wplane

    return vx_euler, vy_euler, wplane


def initialize_ball_vector(xstart, ystart, zstart):
    # Initialize balls at user-supplied positions
    pos = np.array([xstart, ystart, zstart], dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[:, np.newaxis]

    vel = np.zeros(pos.shape, dtype=np.float32)

    return pos, vel


def bilin_interp_d(image, x, y):
    # Bilinear interpolation. About 7x to 10x slower than the Cython implementation.
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    q00 = image[ y0, x0 ]
    q01 = image[ y1, x0 ]
    q10 = image[ y0, x1 ]
    q11 = image[ y1, x1 ]

    dx0 = x - x0
    dy0 = y - y0
    dx1 = x1 - x
    dy1 = y1 - y

    w11 = dx1 * dy1
    w10 = dx1 * dy0
    w01 = dx0 * dy1
    w00 = dx0 * dy0

    return w11*q00 + w10*q01 + w01*q10 + w00*q11


def bilin_interp_f(image, x, y):
    # Bilinear interpolation. About 7x to 10x slower than the Cython implementation.
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    q00 = image[ y0, x0 ]
    q01 = image[ y1, x0 ]
    q10 = image[ y0, x1 ]
    q11 = image[ y1, x1 ]

    dx0 = np.float32(x - x0)
    dy0 = np.float32(y - y0)
    dx1 = np.float32(x1 - x)
    dy1 = np.float32(y1 - y)

    w11 = dx1 * dy1
    w10 = dx1 * dy0
    w01 = dx0 * dy1
    w00 = dx0 * dy0

    return w11*q00 + w10*q01 + w01*q10 + w00*q11


def gauss2d(size,  sigma):

    xgrid, ygrid = np.meshgrid(np.arange(size), np.arange(size))
    r = np.sqrt((xgrid - (size / 2 - 0.5)) ** 2 + (ygrid - (size / 2 - 0.5)) ** 2)
    gauss = 1 - np.exp(- r**2 / (2*sigma**2))

    return gauss

# Display functions
def show_ballpos(image, ballpos):
    plt.figure(1)
    plt.imshow(image, cmap='gray', origin='lower')
    plt.plot(ballpos[:,0], ballpos[:,1], 'r.')


def mesh_ball(rs):

    # number of mesh points along 1 dimension
    p = 20
    # parametrize mesh
    t = np.linspace(0, 1, p)
    th, ph = np.meshgrid(t*pi, t*2*pi)
    # coordinates of sphere mesh
    x = rs * cos(th)
    y = rs * sin(th) * cos(ph)
    z = rs * sin(th) * sin(ph)
    return x,y,z

### Numpy-only function (no C, no Cython)

def integrate_motion0(pos, vel, bt, surface):
    """
    Integrate the motion of a series of balls. This is one integration step.
    Position and velocity are changed in place.
    Either use a loop or a list comprehension to integrate over more than one time step

    :param pos: input and output x, y, z coordinates (1D array)
    :param vel: input and output velocity. (1D array)
    :param bt: BT instance
    :param surface: data surface (2D array)
    :return:
    """
    # Unpack vector components for better readability
    xt, yt, zt = pos
    vxt, vyt, vzt = vel

    # Update the balls grids with current positions
    # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
    bcols = np.clip(bt.bcols + xt, 0, bt.nx - 1).squeeze()
    brows = np.clip(bt.brows + yt, 0, bt.ny - 1).squeeze()

    # "ds" stands for "data surface"
    ds = bilin_interp_f(surface, bcols, brows)


    fxt, fyt, fzt = compute_force(bt, brows, bcols, xt, yt, zt, ds)

    # Integrate velocity
    vxt += fxt
    vyt += fyt
    vzt += fzt

    # Integrate position including effect of a damped velocity
    # Damping is added arbitrarily for the stability of the code.
    pos[0, :] += vxt * bt.td * (1 - bt.e_td)
    pos[1, :] += vyt * bt.td * (1 - bt.e_td)
    pos[2, :] += vzt * bt.zdamping * (1 - bt.e_tdz)
    # Update the velocity with the damping used above
    vel[0, :] *= bt.e_td
    vel[1, :] *= bt.e_td
    vel[2, :] *= bt.e_tdz

    force = np.array([fxt, fyt, fzt])

    #return xt, yt, zt, vxt, vyt, vzt, fxt, fyt, fzt
    # return xt.copy(), yt.copy(), zt.copy(), vxt.copy(), vyt.copy(), vzt.copy(), fxt.copy(), fyt.copy(), fzt.copy()
    return pos.copy(), vel.copy(), force


def drift_series(images, drift_rate):

    drift_images = np.zeros(images.shape)
    for i in range(images.shape[2]):
        dx = -drift_rate[0] * float(i)
        dy = -drift_rate[1]*i
        drift_images[:,:,i] = filters.translate_by_phase_shift(images[:,:,i], dx, dy)

    return drift_images

##############################################################################################################
#################    Calibration     #########################################################################
##############################################################################################################
def process_calibration_series(rotation_rate, nt, rs, dp, sigma_factor, samples):

    print("Creating drift images at rate: [{:.2f}, {:.2f}]".format(rotation_rate[0], rotation_rate[1]))
    # Make the series of drifting image for 1 rotation rate
    drift_images = drift_series(samples, rotation_rate)
    # Balltrack forward and backward
    # ballpos_top, _, _ = balltrack_all(nt, rs, dp, sigma_factor=sigma_factor, mode='top', data=drift_images)
    # ballpos_bottom, _, _ = balltrack_all(nt, rs, dp, sigma_factor=sigma_factor, mode='bottom', data=drift_images)

    # Get a BT instance with the above parameters
    mode_direction_list = (('top','forward'),
                           ('top', 'backward'),
                           ('bottom', 'forward'),
                           ('bottom', 'backward'))

    partial_track = partial(track_instance, nframes=nt, rs=rs, dp=dp, sigma_factor=sigma_factor, data=drift_images)
    bt_tf, bt_tb, bt_bf, bt_bb = list(map(partial_track, mode_direction_list))

    ballpos_top = np.concatenate((bt_tf.ballpos, bt_tb.ballpos), axis=1)
    ballpos_bot = np.concatenate((bt_bf.ballpos, bt_bb.ballpos), axis=1)


    return ballpos_top, ballpos_bot


def fit_calibration(ballpos_list, shift_rates, trange, fwhm, dims):

    vxs, vys, wplanes = zip(*[make_velocity_from_tracks(ballpos, dims, trange, fwhm) for ballpos in ballpos_list])
    # Select an ROI that contains valid data. At least one should exclude edges as wide as the ball radius.
    # This one also excludes the sunspot in the middle. Beware of bias due to differential rotation!

    vxmeans1 = np.array([vx[10:200, 30:-30].mean() for vx in vxs])
    vxmeans2 = np.array([vx[350:500, 30:-30].mean() for vx in vxs])
    vxmeans = 0.5*(vxmeans1 + vxmeans2)

    p = np.polyfit(shift_rates, vxmeans, 1)
    a = 1 / p[0]
    vxfit = a * (vxmeans - p[1])
    return a, vxfit, vxmeans


def loop_calibration_series(rotation_rates, images, nt, rs, dp, sigma_factor, nthreads=1):

    # Use partial to give process_calibration_series() the constant input "samples"
    process_calibration_partial = partial(process_calibration_series, nt=nt, rs=rs, dp=dp, sigma_factor=sigma_factor, samples=images)
    if nthreads < 2:
        ballpos_top_list, ballpos_bottom_list = zip(*map(process_calibration_partial, rotation_rates))
    else:
        pool = Pool(processes=nthreads)
        ballpos_top_list, ballpos_bottom_list = zip(*pool.map(process_calibration_partial, rotation_rates))

    return ballpos_top_list, ballpos_bottom_list

##############################################################################################################
##############################################################################################################


def make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, trange, fwhm):

    vx_top, vy_top, wplane_top = make_velocity_from_tracks(ballpos_top, dims, trange, fwhm)
    vx_bottom, vy_bottom, wplane_bottom = make_velocity_from_tracks(ballpos_bottom, dims, trange, fwhm)

    vx_top *= cal_top
    vy_top *= cal_top
    vx_bottom *= cal_bottom
    vy_bottom *= cal_bottom

    vx = 0.5 * (vx_top + vx_bottom)
    vy = 0.5 * (vy_top + vy_bottom)

    return vx, vy


def make_euler_velocity_lanes(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, nframes, tavg, tstep, fwhm, nsteps, maxstep, outputdir):

    tstarts = np.arange(0, nframes - tavg, tstep)
    tranges = [[tstart, tstart + tavg] for tstart in tstarts]

    for i in range(len(tranges)):
        # Velocity field
        vx, vy = make_euler_velocity(ballpos_top, ballpos_bottom, cal_top, cal_bottom, dims, tranges[i], fwhm)
        # lanes
        lanes = make_lanes(vx, vy, nsteps, maxstep)
        # Write fits file
        fitstools.writefits(vx, os.path.join(outputdir, 'vx_fwhm%d_tavg%d_%03d.fits'%(fwhm, tavg, i)))
        fitstools.writefits(vy, os.path.join(outputdir, 'vy_fwhm%d_tavg%d_%03d.fits'%(fwhm, tavg, i)))
        fitstools.writefits(lanes, os.path.join(outputdir, 'lanes_fwhm%d_tavg%d_nsteps%d_%03d.fits' %(fwhm, tavg, nsteps, i)))


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







