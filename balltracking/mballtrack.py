import sys
import numpy as np
from numpy import pi, cos, sin
import cython_modules.interp as cinterp
import fitstools
import balltracking.balltrack as blt

DTYPE = np.float32
class MBT:

    def __init__(self, dims, nt=1, rs =2, dp=0.3, td=5, zdamping=1,
                 ballspacing=10, intsteps=15, sigma_factor=1, mag_thresh=20, nlevel=5, polarity=1,
                 track_emergence=False, emergence_box=10, datafiles=None, data=None):

        self.datafiles = datafiles
        self.data = data
        self.nx = int(dims[0])
        self.ny = int(dims[1])
        self.nt = nt
        self.intsteps = intsteps
        self.rs = rs
        self.dp = dp
        self.ballspacing = ballspacing
        self.polarity=polarity
        # Number of balls in a row
        self.nballs_row = int((self.nx - 4 * self.rs) / self.ballspacing + 1)
        # Number of balls in a column
        self.nballs_col = int((self.ny - 4 * self.rs) / self.ballspacing + 1)
        # Total number of balls
        self.nballs = 0
        # Image coordinates
        self.xcoords = np.arange(self.nx)
        self.ycoords = np.arange(self.ny)
        # Image mesh
        self.meshx, self.meshy = np.meshgrid(self.xcoords, self.ycoords)
        # Initialize horizontal positions
        #self.xstart, self.ystart = initialize_mesh(self)
        #self.zstart = np.zeros(self.xstart.shape, dtype=np.float32)
        # Maximum number of balls
        self.nballs_max = 2*self.nx*self.ny
        # Initialize the array of the lifetime (age) of the balls
        self.balls_age = np.ones([self.nballs_max], dtype=np.uint32)
        self.balls_age_t = np.ones([self.nballs_max, self.nt], dtype=np.uint32)

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
        self.td = td
        self.zdamping = zdamping
        self.e_td = np.exp(-1/self.td)
        self.e_tdz = np.exp(-1/self.zdamping)

        # Rescaling factor for the standard deviation
        self.sigma_factor = sigma_factor
        self.mean = 0
        self.sigma = 0

        # Current position, force and velocity components, updated after each frame
        self.pos = np.full([3, self.nballs_max], -1, dtype=DTYPE)
        self.vel = np.zeros([3, self.nballs_max], dtype=DTYPE)
        self.force = np.zeros([3, self.nballs_max], dtype=DTYPE)
        self.age = np.zeros([self.nballs_max], dtype=np.uint32)
        # Storage arrays of the above, for all time steps
        self.ballpos = np.zeros([3, self.nballs_max, self.nt], dtype=DTYPE)

        # Ball grid and mesh. Add +1 at the np.arange stop for including right-hand side boundary
        self.ballgrid = np.arange(-self.rs, self.rs + 1, dtype=DTYPE)
        self.ball_cols, self.ball_rows = np.meshgrid(self.ballgrid, self.ballgrid)
        self.bcols = self.ball_cols.ravel()[:, np.newaxis]
        self.brows = self.ball_rows.ravel()[:, np.newaxis]
        self.ds    = np.zeros([self.bcols.shape[0]], dtype=DTYPE)
        # Initialize deepest height at a which ball can fall down. Typically it will be set to a multiple of -surface.std().
        self.min_ds = -5
        # Mask of bad balls
        self.bad_balls_mask = np.zeros(self.nballs_max, dtype=bool)
        # Mask of valid balls
        self.new_valid_balls_mask = np.ones(self.nballs_max, dtype=bool)

        # Data dependent parameters
        self.noise_level = nlevel
        self.mag_thresh = mag_thresh
        self.track_emergence = track_emergence
        self.emergence_box = emergence_box

        self.xstart = 0
        self.ystart = 0
        self.zstart = 0
        # Initialize data & surface. Will update each time a new one is loaded.
        self.image = 0
        self.surface = 0

        self.unique_valid_balls = 0

    def initialize(self):

        if self.data is None:
            self.image = fitstools.fitsread(self.datafiles, tslice=0).astype(DTYPE)
        else:
            self.image = self.data[:,:,0]

        # Initialize for positive flux (we'll see how it goes to parallelize + // -)
        ypos, xpos = np.where(self.image > self.mag_thresh)
        pos = np.array([xpos, ypos], dtype=DTYPE)

        self.surface = prep_data(self.image)

        set_bad_balls(self, pos, check_polarity=False, check_noise=False, check_sinking=False)
        self.nballs = self.unique_valid_balls.size

        self.xstart = pos[0, self.unique_valid_balls]
        self.ystart = pos[1, self.unique_valid_balls]
        self.zstart = blt.put_balls_on_surface(self.surface, self.xstart, self.ystart, self.rs, self.dp)

        # Insert the init positions contiguously
        self.pos[0:2, 0:self.nballs] = pos[0:2, self.unique_valid_balls]
        self.pos[2, 0:self.nballs] = self.zstart.copy()

        self.new_valid_balls_mask = np.zeros([self.nballs_max], dtype=bool)
        self.new_valid_balls_mask[0:self.nballs] = True
        self.unique_valid_balls = np.arange(self.nballs)

        return

    def initialize2(self):

        if self.data is None:
            self.image = fitstools.fitsread(self.datafiles, tslice=0).astype(DTYPE)
        else:
            self.image = self.data[:,:,0]

        # Initialize for positive flux (we'll see how it goes to parallelize + // -
        posy, posx = np.where(self.image > self.mag_thresh)
        nballs = posx.size
        coarse_grid = self.coarse_grid.copy()

        keep_balls = np.ones([nballs], dtype=bool)

        for n in range(nballs):

            xf = np.uint32(np.ceil(posx[n])/self.ballspacing)
            yf = np.uint32(np.ceil(posy[n])/self.ballspacing)

            coarse_grid[yf, xf] +=1

            if coarse_grid[yf, xf] >1:
                keep_balls[n] = False
                coarse_grid[yf, xf] -= 1

        posx = posx[keep_balls]
        posy = posy[keep_balls]

        return posx, posy



    def track_all_frames(self):

        self.initialize()


        for n in range(self.nt):

            #print('Frame n=%d'%n)

            if self.data is None:
                self.image = fitstools.fitsread(self.datafiles, tslice=0).astype(np.float32)
            else:
                self.image = self.data[:, :, 0]

            self.surface = prep_data(self.image)
            # The current position "pos" and velocity "vel" are attributes of bt.
            # They are integrated in place.
            for i in range(self.intsteps):
                #print('intermediate step i=%d'%i)
                blt.integrate_motion(self, self.surface)

            set_bad_balls(self, self.pos)

            # Flag the bad balls with -1
            self.pos[:, self.bad_balls_mask] = -1
            self.vel[:, self.bad_balls_mask] = np.nan
            self.balls_age[self.new_valid_balls_mask] += 1

            self.balls_age_t[:, n] = self.balls_age.copy()
            self.ballpos[..., n] = self.pos.copy()

    def populate_emergence(self):

        # Get the flux according to which polarity the tracking is set to.
        if self.polarity >= 0:
            flux_posy, flux_posx = np.where(self.image > self.mag_thresh)
        else:
            flux_posy, flux_posx = np.where(self.image < -self.mag_thresh)

        # Consider getting a view by using tuples of indices...
        # For now we are getting a copy
        ball_posx, ball_posy = self.pos[0:2, self.new_valid_balls_mask]

        distance_matrix = np.sqrt((flux_posx[:,np.newaxis] - ball_posx[np.newaxis,:])**2 + (flux_posy[:,np.newaxis] - ball_posy[np.newaxis,:])**2)
        distance_min = distance_matrix.min(axis=1)
        populate_flux_mask = distance_min > self.emergence_box

        # Populate only if there's something
        if populate_flux_mask.sum() > 0 :

            newposx = flux_posx[populate_flux_mask]
            newposy = flux_posx[populate_flux_mask]
            newposz = blt.put_balls_on_surface(self.surface, newposx, newposy, self.rs, self.dp)

            # Insert the new positions contiguously in the self.pos array
            # We need to use the number of balls at initialization (self.nballs) and increment it with the number
            # of new balls to populate the emerging flux.
            self.pos[0, self.nballs:self.nballs+newposx.size] = newposx
            self.pos[1, self.nballs:self.nballs + newposx.size] = newposy
            self.pos[2, self.nballs:self.nballs + newposx.size] = newposz

            self.nballs += newposx.size


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
        sunk_mask = pos2[2,:] > bt.surface[pos2[1, :], pos2[0, :]] - 4*bt.rs
        valid_balls_mask2 = np.logical_and(valid_balls_mask2, sunk_mask)

    # Get indices in the original array. Remember that valid_balls_mask2 has the same size as pos2
    valid_balls_idx = valid_balls_idx[valid_balls_mask2]
    # Get the valid balls from the input pos array.
    # indexing scheme below returns a copy, just like with boolean index arrays.
    xpos, ypos = pos[0:2, valid_balls_idx]
    balls_age = bt.balls_age[valid_balls_idx]

    ## Decimation based on the coarse grid.
    # Get the 1D position on the coarse grid, clipped to the edges of that grid.
    _, _, coarse_pos = blt.coarse_grid_pos(bt, xpos, ypos)

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






