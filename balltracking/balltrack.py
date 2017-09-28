import numpy as np
import matplotlib.pyplot as plt
from numpy import pi, cos, sin
import filters


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
        self.xstart, self.ystart = self.initialize_mesh()
        self.zstart = np.zeros(self.xstart.shape)
        self.nballs = self.xstart.size

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
        self.xt = np.zeros([self.nballs])
        self.yt = np.zeros([self.nballs])
        self.zt = np.zeros([self.nballs])
        self.fxt = np.zeros([self.nballs])
        self.fyt = np.zeros([self.nballs])
        self.fzt = np.zeros([self.nballs])
        self.vxt = np.zeros([self.nballs])
        self.vyt = np.zeros([self.nballs])
        self.vzt = np.zeros([self.nballs])
        self.age = np.zeros([self.nballs], dtype=np.uint32)
        # Storage arrays of the above
        self.x = np.zeros([self.nt, self.nballs])
        self.y = np.zeros([self.nt, self.nballs])
        self.z = np.zeros([self.nt, self.nballs])
        self.fx = np.zeros([self.nt, self.nballs])
        self.fy = np.zeros([self.nt, self.nballs])
        self.fz = np.zeros([self.nt, self.nballs])
        self.vx = np.zeros([self.nt, self.nballs])
        self.vy = np.zeros([self.nt, self.nballs])
        self.vz = np.zeros([self.nt, self.nballs])

        # Ball grid and mesh. Add +1 at the np.arange stop for including right-hand side boundary
        self.ballgrid = np.arange(-self.rs, self.rs + 1)
        self.ball_cols, self.ball_rows = np.meshgrid(self.ballgrid, self.ballgrid)
        self.bcols = self.ball_cols.ravel()[:, np.newaxis]
        self.brows = self.ball_rows.ravel()[:, np.newaxis]

    def initialize_mesh(self):
        # Initial horizontal (x,y) positions
        x_start_points = np.linspace(2 * self.rs, self.nx - 2 * self.rs, self.nballs_row)
        y_start_points = np.linspace(2 * self.rs, self.ny - 2 * self.rs, self.nballs_col)
        xstart, ystart = np.meshgrid(x_start_points, y_start_points)
        return xstart, ystart

    def initialize_ballpos(self, surface):
        # Initialize the height of the ball. Only possible if the data surface is given.
        self.xt = self.xstart.flatten()
        self.yt = self.ystart.flatten()
        self.zstart = put_balls_on_surface(surface, self.xstart.ravel(), self.ystart.ravel(), self.rs, self.dp)
        self.zt = self.zstart.copy()
        return

def put_balls_on_surface(surface, x, y, rs, dp):
    z = bilin_interp(surface, x, y) +  rs*(1-dp/2)
    return z

def integrate_motion(pos, vel, bt, surface):

    # Unpack vector components for better readability
    xt, yt, zt = pos
    vxt, vyt, vzt = vel

    # Update the balls grids with current positions
    # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
    bcols = np.clip(bt.bcols + xt, 0, bt.nx - 1)
    brows = np.clip(bt.brows + yt, 0, bt.ny - 1)

    # "ds" stands for "data surface"
    ds = bilin_interp(surface, bcols, brows)
    r = np.sqrt((bcols - xt) ** 2 + (brows - yt) ** 2 + (ds - zt) ** 2)
    # Force that are beyond the radius must be set to zero
    f = bt.k_force * (r - bt.rs)
    f[r > bt.rs] = 0
    # Calculate each force vector component
    fxt = -np.sum(f * (xt - bcols) / r, 0)
    fyt = -np.sum(f * (yt - brows) / r, 0)
    # Buoyancy must stay oriented upward. Used to be signed, but that caused more lost balls without other advantage
    fzt = -np.sum(f * np.abs(zt - ds) / r, 0) - bt.am

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
    # vxt *= bt.e_td
    # vyt *= bt.e_td
    # vzt *= bt.e_tdz

    force = np.array([fxt, fyt, fzt])

    #return xt, yt, zt, vxt, vyt, vzt, fxt, fyt, fzt
    # return xt.copy(), yt.copy(), zt.copy(), vxt.copy(), vyt.copy(), vzt.copy(), fxt.copy(), fyt.copy(), fzt.copy()
    return pos.copy(), vel.copy(), force

# def integrate_motion(xt, yt, zt, vxt, vyt, vzt, bt, surface, intsteps, store_integration=False):
#
#     if store_integration:
#         x, y, z, vx, vy, vz, fx, fy, fz = [], [], [], [], [], [], [], [], []
#
#     for i in range(intsteps):
#         # Update the balls grids with current positions
#         # bcols and brows have dimensions = [prod(ballgrid.shape), nballs]
#         bcols = np.clip(bt.bcols + xt, 0, bt.nx - 1)
#         brows = np.clip(bt.brows + yt, 0, bt.ny - 1)
#
#         # "ds" stands for "data surface"
#         ds = bilin_interp(surface, bcols, brows)
#         r = np.sqrt((bcols - xt) ** 2 + (brows - yt) ** 2 + (ds - zt) ** 2)
#         # Force that are beyond the radius must be set to zero
#         f = bt.k_force * (r - bt.rs)
#         f[r > bt.rs] = 0
#         # Calculate each force vector component
#         fxt = -np.sum(f * (xt - bcols) / r, 0)
#         fyt = -np.sum(f * (yt - brows) / r, 0)
#         # Buoyancy must stay oriented upward. Used to be signed, but that caused more lost balls without other advantage
#         fzt = -np.sum(f * np.abs(zt - ds) / r, 0) - bt.am
#
#         # Integrate velocity
#         vxt += fxt
#         vyt += fyt
#         vzt += fzt
#
#         # Integrate position including effect of a damped velocity
#         # Damping is added arbitrarily for the stability of the code.
#         xt += vxt * bt.td * (1 - bt.e_td)
#         yt += vyt * bt.td * (1 - bt.e_td)
#         zt += vzt * bt.zdamping * (1 - bt.e_tdz)
#         # Update the velocity with the damping used above
#         vxt *= bt.e_td
#         vyt *= bt.e_td
#         vzt *= bt.e_tdz
#
#         if store_integration:
#             x.append(xt)
#             y.append(yt)
#             z.append(zt)
#             vx.append(vxt)
#             vy.append(vyt)
#             vz.append(vzt)
#             fx.append(fxt)
#             fy.append(fyt)
#             fz.append(fzt)
#
#     if store_integration:
#         return x, y, z, vx, vy, vz, fx, fy, fz
#     else:
#         return #xt, yt, zt, vxt, vyt, vzt, fxt, fyt, fzt

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

def initialize_ball_vector(xstart, ystart, zstart):

    # xt = np.array([xstart], dtype=float)
    # yt = np.array([ystart], dtype=float)
    # zt = np.array([zstart], dtype=float)



    # vxt = np.array([0], dtype=float)
    # vyt = np.array([0], dtype=float)
    # vzt = np.array([0], dtype=float)

    pos = np.array([xstart, ystart, zstart], dtype=float)
    if pos.ndim == 1:
        pos = pos[:, np.newaxis]

    vel = np.zeros(pos.shape, dtype=float)

    # return xt, yt, zt, vxt, vyt, vzt, fxt, fyt, fzt
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

def bilin_interp(im, x, y):

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    q00 = im[ y0, x0 ]
    q01 = im[ y1, x0 ]
    q10 = im[ y0, x1 ]
    q11 = im[ y1, x1 ]

    dx0 = x - x0
    dy0 = y - y0
    dx1 = x1 - x
    dy1 = y1 - y

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

