import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPEf = np.float32
DTYPEd = np.double
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float32_t DTYPE_tf
ctypedef np.double_t DTYPE_td
ctypedef np.uint_t DTYPE_tui

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
# def bilin_interp1(np.ndarray[DTYPE_t, ndim=2, mode="c"] im, np.ndarray[DTYPE_t, ndim=1, mode="c"] x, np.ndarray[DTYPE_t, ndim=1, mode="c"] y):
#
#     assert im.dtype == DTYPE and x.dtype == DTYPE and y.dtype == DTYPE
#
#     # The "cdef" keyword is also used within functions to type variables. It
#     # can only be used at the top indentation level (there are non-trivial
#     # problems with allowing them in other places, though we'd love to see
#     # good and thought out proposals for it).
#     #
#     # For the indices, the "int" type is used. This corresponds to a C int,
#     # other C types (like "unsigned int") could have been used instead.
#     # Purists could use "Py_ssize_t" which is the proper Python type for
#     # array indices.
#
#     cdef unsigned int x0, x1, y0, y1
#     cdef int k
#     cdef DTYPE_t q00, q01, q10, q11, dx0, dy0, dx1, dy1, w11, w10, w01, w00
#
#     cdef int npts = x.shape[0]
#     cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros([npts], dtype=DTYPE)
#
#     for k in range(npts):
#
#         x0 = <unsigned int>x[k]
#         y0 = <unsigned int>y[k]
#         x1 = x0 + 1
#         y1 = y0 + 1
#
#         q00 = im[y0, x0]
#         q01 = im[y1, x0]
#         q10 = im[y0, x1]
#         q11 = im[y1, x1]
#
#         dx0 = x[k] - x0
#         dy0 = y[k] - y0
#         dx1 = x1 - x[k]
#         dy1 = y1 - y[k]
#
#         w11 = dx1 * dy1
#         w10 = dx1 * dy0
#         w01 = dx0 * dy1
#         w00 = dx0 * dy0
#
#         result[k] = w11*q00 + w10*q01 + w01*q10 + w00*q11
#
#     return result

def bilin_interp1f(DTYPE_tf[:,:] im, DTYPE_tf[:] x, DTYPE_tf[:] y):

    #assert im.dtype == DTYPE and x.dtype == DTYPE and y.dtype == DTYPE

    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.

    cdef unsigned int x0, x1, y0, y1
    cdef unsigned int i
    cdef DTYPE_tf q00, q01, q10, q11, dx0, dy0, dx1, dy1, w11, w10, w01, w00

    cdef unsigned int n = <unsigned int>x.shape[0]
    cdef np.ndarray[DTYPE_tf, ndim=1] result = np.zeros([n], dtype=DTYPEf)

    for i in range(n):

        x0 = <unsigned int>x[i]
        y0 = <unsigned int>y[i]
        x1 = x0 + 1
        y1 = y0 + 1

        q00 = im[y0, x0]
        q01 = im[y1, x0]
        q10 = im[y0, x1]
        q11 = im[y1, x1]

        dx0 = x[i] - x0
        dy0 = y[i] - y0
        dx1 = x1 - x[i]
        dy1 = y1 - y[i]

        w11 = dx1 * dy1
        w10 = dx1 * dy0
        w01 = dx0 * dy1
        w00 = dx0 * dy0

        result[i] = w11*q00 + w10*q01 + w01*q10 + w00*q11

    return result

def bilin_interp2f(np.ndarray[DTYPE_tf, ndim=2, mode="c"] image, np.ndarray[DTYPE_tf, ndim=2, mode="c"] x, np.ndarray[DTYPE_tf, ndim=2, mode="c"] y):

    assert image.dtype == DTYPEf and x.dtype == DTYPEf and y.dtype == DTYPEf

    cdef unsigned int k
    cdef unsigned int npts = x.shape[0]
    cdef unsigned int nballs = x.shape[1]


    cdef np.ndarray[DTYPE_tf, ndim=2] result = np.zeros([npts, nballs], dtype=DTYPEf)

    cdef float [:, :] x_view = x
    cdef float [:, :] y_view = y
    for k in range(npts):

        result[k, :] = bilin_interp1f(image, x_view[k, :], y_view[k, :])

    return result

def bilin_interp3f(np.ndarray[DTYPE_tf, ndim=2, mode="c"] im, np.ndarray[DTYPE_tf, ndim=2, mode="c"] x, np.ndarray[DTYPE_tf, ndim=2, mode="c"] y):

    assert im.dtype == DTYPEf and x.dtype == DTYPEf and y.dtype == DTYPEf

    cdef unsigned int npts = x.shape[0]
    cdef unsigned int nballs = x.shape[1]
    cdef unsigned int x0, x1, y0, y1
    cdef unsigned int b, k
    cdef DTYPE_tf q00, q01, q10, q11, dx0, dy0, dx1, dy1, w11, w10, w01, w00
    cdef np.ndarray[DTYPE_tf, ndim=2] result = np.zeros([npts, nballs], dtype=DTYPEf)
    cdef float [:, :] x_view = x
    cdef float [:, :] y_view = y

    for j in range(npts):
        for i in range(nballs):

            x0 = <unsigned int>x[j, i]
            y0 = <unsigned int>y[j, i]
            x1 = x0 + 1
            y1 = y0 + 1

            q00 = im[y0, x0]
            q01 = im[y1, x0]
            q10 = im[y0, x1]
            q11 = im[y1, x1]

            dx0 = x[j, i] - x0
            dy0 = y[j, i] - y0
            dx1 = x1 - x[j, i]
            dy1 = y1 - y[j, i]

            w11 = dx1 * dy1
            w10 = dx1 * dy0
            w01 = dx0 * dy1
            w00 = dx0 * dy0

            result[j, i] = w11*q00 + w10*q01 + w01*q10 + w00*q11

    return result


def bilin_interp1d(DTYPE_td[:,:] im, DTYPE_td[:] x, DTYPE_td[:] y):

    #assert im.dtype == DTYPE and x.dtype == DTYPE and y.dtype == DTYPE

    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.

    cdef unsigned int x0, x1, y0, y1
    cdef int k
    cdef DTYPE_td q00, q01, q10, q11, dx0, dy0, dx1, dy1, w11, w10, w01, w00

    cdef int npts = x.shape[0]
    cdef np.ndarray[DTYPE_td, ndim=1] result = np.zeros([npts], dtype=DTYPEd)

    for k in range(npts):

        x0 = <unsigned int>x[k]
        y0 = <unsigned int>y[k]
        x1 = x0 + 1
        y1 = y0 + 1

        q00 = im[y0, x0]
        q01 = im[y1, x0]
        q10 = im[y0, x1]
        q11 = im[y1, x1]

        dx0 = x[k] - x0
        dy0 = y[k] - y0
        dx1 = x1 - x[k]
        dy1 = y1 - y[k]

        w11 = dx1 * dy1
        w10 = dx1 * dy0
        w01 = dx0 * dy1
        w00 = dx0 * dy0

        result[k] = w11*q00 + w10*q01 + w01*q10 + w00*q11

    return result

def bilin_interp2d(np.ndarray[DTYPE_td, ndim=2, mode="c"] im, np.ndarray[DTYPE_td, ndim=2, mode="c"] x, np.ndarray[DTYPE_td, ndim=2, mode="c"] y):

    assert im.dtype == DTYPEd and x.dtype == DTYPEd and y.dtype == DTYPEd

    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.

    cdef unsigned int b
    cdef unsigned int nballs = x.shape[0]
    cdef unsigned int npts = x.shape[1]


    cdef np.ndarray[DTYPE_td, ndim=2] result = np.zeros([nballs, npts], dtype=DTYPEd)

    cdef double [:, :] x_view = x
    cdef double [:, :] y_view = y
    for b in range(nballs):

        result[b, :] = bilin_interp1d(im, x_view[b, :], y_view[b, :])

    return result

# declare the interface to the C code
cdef extern void cbilin_interp(float *image, float *xout, float *yout, float *values, int nx, int npts)

@cython.boundscheck(False)
@cython.wraparound(False)
def cbilin_interp1(np.ndarray[DTYPE_tf, ndim=2, mode="c"] image, np.ndarray[DTYPE_tf, ndim=1, mode="c"] x, np.ndarray[DTYPE_tf, ndim=1, mode="c"] y):
    """
    Bilinear interpolation for 1-dimensional arrays of coordinates.
    Mostly used for interpolating the values at the ball center, e.g when dropping the ball at initialization

    :param image: data surface (2D array)
    :param x: x-coordinate to interpolate (1D array)
    :param y: y-coordinate to interpolate (1D array)
    :param z: interpolated value (1D array)
    :return:
    """
    cdef int nx, npts

    nx, npts = image.shape[1], x.shape[0]

    cdef np.ndarray[DTYPE_tf, ndim=1, mode="c"] z = np.zeros([npts], dtype=DTYPEf)

    cbilin_interp(&image[0,0], &x[0], &y[0], &z[0], nx, npts)

    return z

cdef extern void cbilin_interp2d(float *image, float *xout, float *yout, float *values, int nx, int npts, int nballs)

@cython.boundscheck(False)
@cython.wraparound(False)
def cbilin_interp2(np.ndarray[DTYPE_tf, ndim=2, mode="c"] image, np.ndarray[DTYPE_tf, ndim=2, mode="c"] x, np.ndarray[DTYPE_tf, ndim=2, mode="c"] y):
    """
    Bilinear interpolation for 2-dimensional arrays of coordinates.
    These arrays occur when interpolating at not only the coordinate of the balls center,
    but also at all the surface points where the grid points of the sphere project onto.

    :param image: data surface (2D array)
    :param x: x-coordinate to interpolate (2D array)
    :param y: y-coordinate to interpolate (2D array)
    :return: z, interpolated values (2D array)
    """
    cdef int nx, npts, nballs

    nx, npts, nballs = image.shape[1], x.shape[0], x.shape[1]

    cdef np.ndarray[DTYPE_tf, ndim=2, mode="c"] z = np.zeros([npts, nballs], dtype=DTYPEf)

    cbilin_interp2d(&image[0,0], &x[0,0], &y[0,0], &z[0,0], nx, npts, nballs)

    return z

