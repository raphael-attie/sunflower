import numpy as np

# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).

cimport numpy as np

# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.double
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.double_t DTYPE_t

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
def bilin_interp1(np.ndarray[DTYPE_t, ndim=2, mode="c"] im, np.ndarray[DTYPE_t, ndim=1, mode="c"] x, np.ndarray[DTYPE_t, ndim=1] y):

    assert im.dtype == DTYPE and x.dtype == DTYPE and y.dtype == DTYPE

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
    cdef DTYPE_t q00, q01, q10, q11, dx0, dy0, dx1, dy1, w11, w10, w01, w00

    cdef int npts = x.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.zeros([npts], dtype=DTYPE)

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

# declare the interface to the C code
cdef extern void cbilin_interp(double *image, double *xout, double *yout, double *values, int nx, int npts)

@cython.boundscheck(False)
@cython.wraparound(False)
def bilin_interp2(np.ndarray[double, ndim=2, mode="c"] image, np.ndarray[double, ndim=1, mode="c"] x, np.ndarray[double, ndim=1, mode="c"] y, np.ndarray[double, ndim=1, mode="c"] z):

    cdef int nx, npts

    nx, npts = image.shape[1], x.shape[0]

    cbilin_interp(&image[0,0], &x[0], &y[0], &z[0], nx, npts)

    return None



