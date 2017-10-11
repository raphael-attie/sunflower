from importlib import reload
import numpy as np
import balltracking.balltrack as blt
import matplotlib.pyplot as plt
from timeit import timeit

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

### Test and benchmark of the integration on a generic gaussian data surface

# Create a generic surface
size = 50
sigma = 2
surface = blt.gauss2d(size, sigma).astype(np.float32) * 3
# Set ball parameters, the number of frames nt is ignored in this benchmar
rs = float(2.0)
dp = float(0.2)
nt = 50
# Instatiate the BT class with the gaussian generic surface dimensions
bt = blt.BT(surface.shape, nt, rs, dp)

# Initialize 1 ball
xstart = np.array([20], dtype=np.float32)
ystart = np.array([24], dtype=np.float32)
zstart = blt.put_balls_on_surface(surface, xstart, ystart, rs, dp)

# Try with python/numpy interpolation
pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)
pos, vel, force = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion(pos, vel, bt, surface) for i in range(nt)])]
# Try with cython
pos2, vel2 = blt.initialize_ball_vector(xstart, ystart, zstart)
pos2, vel2, force2 = [np.array(v).squeeze() for v in zip(*[blt.integrate_motion2(pos2, vel2, bt, surface) for i in range(nt)])]


# Display and compare results
f1 = plt.figure(figsize=(10, 10))
plt.imshow(surface, origin='lower', cmap='gray')
plt.plot(xstart, ystart, 'r+', markersize=10)

plt.plot(pos[:,0], pos[:,1], 'go', markerfacecolor='none')
plt.plot(pos2[:,0], pos2[:,1], 'b+', markerfacecolor='none')


# Profile

xstart = np.full([16129], 20, dtype=np.float32)
ystart = np.full([16129], 30, dtype=np.float32)
zstart = blt.put_balls_on_surface(surface, xstart, ystart, rs, dp)


pos, vel = blt.initialize_ball_vector(xstart, ystart, zstart)
mywrap = wrapper(blt.integrate_motion, pos, vel, bt, surface)
print(timeit(mywrap, number = 100))

pos2, vel2 = blt.initialize_ball_vector(xstart, ystart, zstart)
mywrap2 = wrapper(blt.integrate_motion2, pos2, vel2, bt, surface)
print(timeit(mywrap2, number = 100))

