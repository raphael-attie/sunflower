import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos, sin
import cython_modules.interp as cinterp
import filters

DTYPE = np.float32
class BT:


    def __init__(self, dims, nt, rs, dp):
        self.nx = int(dims[0])
        self.ny = int(dims[1])
        self.nt = nt
        self.intsteps = 3
        self.rs = rs
        self.ballspacing = int(2*rs)
        self.dp = dp
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


        # Current position, force and velocity components, updated after each frame
        self.pos_t = np.zeros([3, self.nballs], dtype=DTYPE)
        self.vel_t = np.zeros([3, self.nballs], dtype=DTYPE)
        self.force_t = np.zeros([3, self.nballs], dtype=DTYPE)
        self.age = np.zeros([self.nballs], dtype=np.uint32)
        # Storage arrays of the above
        self.pos = np.zeros([3, self.nt, self.nballs], dtype=DTYPE)
        self.vel = np.zeros([3, self.nt, self.nballs], dtype=DTYPE)

        # Ball grid and mesh. Add +1 at the np.arange stop for including right-hand side boundary
        self.ballgrid = np.arange(-self.rs, self.rs + 1, dtype=DTYPE)
        self.ball_cols, self.ball_rows = np.meshgrid(self.ballgrid, self.ballgrid)
        self.bcols = self.ball_cols.ravel()[:, np.newaxis]
        self.brows = self.ball_rows.ravel()[:, np.newaxis]
        self.ds    = np.zeros([self.bcols.shape[0]], dtype=DTYPE)
        # Initialize deepest height at a which ball can fall down. Typically it will be set to a multiple of -surface.std().
        self.min_ds = 0.0

    def initialize_ballpos(self, surface):
        # Initialize the height of the ball. Only possible if the data surface is given.
        self.pos_t[0, :] = self.xstart.flatten()
        self.pos_t[1, :] = self.ystart.flatten()
        self.zstart = put_balls_on_surface(surface, self.xstart.ravel(), self.ystart.ravel(), self.rs, self.dp)
        self.pos_t[2, :] = self.zstart.copy()
        # Setup the coarse grid: populate edges to avoid "leaks" of balls ~ balls falling off.
        # Although can't remember if that was actually ever used in the Matlab implementation
        # TODO: check Matlab implementation about the use of the edge filling.
        # self.coarse_grid[0,:]   = 1
        # self.coarse_grid[:, 0]  = 1
        # self.coarse_grid[-1,:]  = 1
        # self.coarse_grid[:, -1] = 1
        # Beware of the std() if there's a bad region in the image (e.g sunspot)
        self.min_ds = -5*surface.std()
        return

def coarse_grid_pos(bt, x, y):
    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.uint32(np.clip(np.floor(x / bt.ballspacing), 0, bt.nxc-1))
    ycoarse = np.uint32(np.clip(np.floor(y / bt.ballspacing), 0, bt.nyc-1))
    # Convert to linear (1D) indices. One index per ball
    coarse_idx = np.ravel_multi_index((ycoarse, xcoarse), bt.coarse_grid.shape)
    return xcoarse, ycoarse, coarse_idx

def fill_coarse_grid(bt, x, y):
    """
    Fill coarse_grid as a chess board.

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

    z = np.zeros([x.shape[0]], dtype=np.float32)
    #z = bilin_interp_f(surface, x, y) +  rs*(1-dp/2)
    z = cinterp.cbilin_interp1(surface, x, y)
    z += rs * (1 - dp / 2)
    return z

def compute_force(bt, brows, bcols, xt, yt, zt, ds):

    r = np.sqrt((bcols - xt) ** 2 + (brows - yt) ** 2 + (ds - zt) ** 2)
    # Force that are beyond the radius must be set to zero
    f = bt.k_force * (r - bt.rs)
    f[r > bt.rs] = 0
    # Calculate each force vector component
    fxt = -np.sum(f * (xt - bcols) / r, 0)
    fyt = -np.sum(f * (yt - brows) / r, 0)
    # Buoyancy must stay oriented upward. Used to be signed, but that caused more lost balls without other advantage
    fzt = -np.sum(f * np.abs(zt - ds) / r, 0) - bt.am

    return fxt, fyt, fzt


def integrate_motion(pos, vel, bt, surface):
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

def integrate_motion2(pos_t, vel_t, bt, surface):

    # Unpack vector components for better readability
    xt, yt, zt = pos_t
    vxt, vyt, vzt = vel_t

    # Update the balls grids with current positions
    # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
    bcols = np.clip(bt.bcols + xt, 0, bt.nx - 1)
    brows = np.clip(bt.brows + yt, 0, bt.ny - 1)

    # "ds" stands for "data surface"
    #ds = surface[np.round(brows).astype(np.int), np.round(bcols).astype(np.int)]
    ds = cinterp.cbilin_interp2(surface, bcols, brows)

    fxt, fyt, fzt = compute_force(bt, brows, bcols, xt, yt, zt, ds)

    # Integrate velocity

    vxt += fxt
    vyt += fyt
    vzt += fzt

    # Integrate position including effect of a damped velocity
    # Damping is added arbitrarily for the stability of the code.

    pos_t[0, :] += vxt * bt.td * (1 - bt.e_td)
    pos_t[1, :] += vyt * bt.td * (1 - bt.e_td)
    pos_t[2, :] += vzt * bt.zdamping * (1 - bt.e_tdz)


    # Update the velocity with the damping used above
    vel_t[0, :] *= bt.e_td
    vel_t[1, :] *= bt.e_td
    vel_t[2, :] *= bt.e_tdz

    force = np.array([fxt, fyt, fzt])

    return pos_t.copy(), vel_t.copy(), force

def integrate_balls(bt, surface):

    good_balls_mask = np.logical_and(bt.xt > 0, np.isfinite(bt.xt))
    xt = bt.xt[good_balls_mask]
    yt = bt.yt[good_balls_mask]
    zt = bt.zt[good_balls_mask]

    vxt = bt.vxt[good_balls_mask]
    vyt = bt.vyt[good_balls_mask]
    vzt = bt.vzt[good_balls_mask]

    # The intermediate integrations (over intsteps) should start here.

    # Unless the input are scalars, inputs are numpy arrays, mutable and passed by reference,
    # which makes it unnecessary to return them. However, in case of scalars, it is necessary.
    #xt, yt, zt, vxt, vyt, vzt, fxt, fyt, fzt = integrate_motion(xt, yt, zt, vxt, vyt, vzt, bt, surface)
    integrate_motion(xt, yt, zt, vxt, vyt, vzt, bt, surface)

    bt.xt[good_balls_mask] = xt
    bt.yt[good_balls_mask] = yt
    bt.zt[good_balls_mask] = zt

    bt.vxt[good_balls_mask] = vxt
    bt.vyt[good_balls_mask] = vyt
    bt.vzt[good_balls_mask] = vzt

    return

def get_bad_balls(pos, bt):
    # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays
    # and https://stackoverflow.com/questions/36863404/accumulate-constant-value-in-numpy-array

    # The input come from the integration function, which only treats valid, finite ball positions.
    # There is no need to worry about NaN, or Inf positions.

    # Work with with views on coordinate and velocity arrays for clarity
    xpos0, ypos0, zpos0 = pos

    # Matlab:
    # col_min = int16(floor(x0 - BT.rs));
    # col_max = int16(ceil(x0 + BT.rs));
    # row_min = int16(floor(y0 - BT.rs));
    # row_max = int16(ceil(y0 + BT.rs));

    # badballs = (z0==fallnoted) | col_max>BT.nc | col_min<1 | row_max>BT.nr |...
    # row_min<1 | (z0<(minds-4*BT.rs)) | badfullballs;

    sunk = zpos0 < bt.min_ds
    off_edge_left = xpos0 - bt.rs < 0
    off_edge_right = xpos0 + bt.rs > bt.nx-1
    off_edge_bottom = ypos0 - bt.rs < 0
    off_edge_top = ypos0 + bt.rs > bt.ny-1

    masks = np.array((sunk, off_edge_left, off_edge_right, off_edge_bottom, off_edge_top))
    bad_balls1 = np.logical_or.reduce(masks)

    # Ignore these bad balls in the arrays and enforce continuity principle
    valid_balls = np.logical_not(bad_balls1)
    valid_balls_idx = np.nonzero(valid_balls)[0]

    xpos, ypos, zpos = pos[:, valid_balls]
    balls_age = bt.balls_age[valid_balls]

    # Get the position on the coarse grid, clipped to the edges of that grid.
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
    bad_balls = np.logical_or(bad_balls1, bad_full_balls)

    return bad_balls



#def replace_bad_balls(pos, age, finegrid, newframe, bt):
def replace_bad_balls(pos, bt):
    # Get the position on the finegrid
    # xfinegrid = max(1, min(BT.nc_fg, round(pos(:, 1) / BT.finegridspacing)));
    # yfinegrid = max(1, min(BT.nr_fg, round(pos(:, 2) / BT.finegridspacing)));
    # xyfinegrid1D = sub2ind(size(finegrid), yfinegrid, xfinegrid); % length = nballs

    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.clip(np.round(pos[0, :] / bt.ballspacing), 0, bt.nxc).astype(np.uint32)
    ycoarse = np.clip(np.round(pos[1, :] / bt.ballspacing), 0, bt.nyc).astype(np.uint32)

    # We may not need to convert 2D indices to 1D indices. Let's see how numpy behaves with this.
    # See discussion at https://stackoverflow.com/questions/44802033/efficiently-index-2d-numpy-array-using-two-1d-arrays

    # Set positions on coarse grid as occupied where the balls are (=1)
    bt.coarse_grid[ycoarse, xcoarse] = 1

    return xcoarse, ycoarse

def replace_bad_balls1(pos, bt):
    # Get the position on the finegrid

    # Get the position on the coarse grid, clipped to the edges of that grid.
    xcoarse = np.round(pos[0, :] / bt.ballspacing).clip(0, bt.nxc).astype(np.uint32)
    ycoarse = np.round(pos[1, :] / bt.ballspacing).clip(0, bt.nyc).astype(np.uint32)

    bt.coarse_grid[ycoarse, xcoarse] = 1

    return xcoarse, ycoarse

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



def initialize_ball_vector(xstart, ystart, zstart):
    # Initialize balls at user-supplied positions
    pos = np.array([xstart, ystart, zstart], dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[:, np.newaxis]

    vel = np.zeros(pos.shape, dtype=np.float32)

    return pos, vel


def rescale_frame(data, norm_factor):
    """
    Rescales the images for Balltracking. It subtract the mean of the data and divide by a scaling factor.
    For balltracking, this scaling factor shall be the standard deviation of the whole data series.
    See http://mathworld.wolfram.com/HanningFunction.html

    :param data: 2D frame (e.g: continuum image or magnetogram)
    :param norm_factor: scalar that normalizes the data. Typically the standard deviation.
    :return: rescaled_data: rescaled image.
    """
    rescaled_data = data - np.mean(data)
    rescaled_data = rescaled_data / norm_factor
    return rescaled_data

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

