#cimport segmentation.pxd

# import skimage.measure
# import skimage.morphology

import numpy as np
cimport numpy as np

cimport cython

cimport segmentation

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef csegmentation(np.ndarray[np.float64_t, ndim=2] m, double threshold):
    """
    Sort the the array intensity from max to min
    Label each pixel with the same number if their connected (i.e do connected component labelling)
    :param m: magnetogram (numpy array)
    :param threshold: value above which to label the magnetic elements
    :return:
    """
    # Get locations (indices) of SORTED array intensities from max to min (locations are positives values, unsigned)
    cdef unsigned int[:] indices = np.argsort(m.flatten()).astype(np.uint32)[::-1]
    # Initialize the labels for the connected component labelling. First, there are only zeros in the array
    labels_arr = np.zeros((m.shape[0], m.shape[1]), dtype=np.uint32)
    cdef unsigned int[:,:] labels = labels_arr
    cdef unsigned int next_label = 1
    cdef unsigned int index
    cdef unsigned int rows = m.shape[0]
    cdef unsigned int cols = m.shape[1]
    # Build an array that will point to the 8-connected neighbours.
    cdef int[:,:] offsets = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]], dtype=np.int32)
    cdef unsigned int offset_i
    cdef int y, x, y_off, x_off
    cdef double value
    cdef unsigned int label_at_off
    cdef unsigned int index_i
    for i in range(indices.shape[0]):
        # Iterate over each pixel, starting from the MAXIMUM value and grow the region downhill
        index = indices[i]
        y = index / cols
        x = index % cols
        value = m[y, x]
        if value < threshold:
            break
        for offset_i in range(offsets.shape[0]):
            y_off = offsets[offset_i, 0]
            x_off = offsets[offset_i, 1]
            # Detect 8-connected neighbours with label & make sure we stay inside the array
            if y+y_off >= 0 and y+y_off < rows and x+x_off >= 0 and x+x_off < cols:
                # Get the labels & intensity of the neighbours intensity
                label_at_off = labels[y+y_off, x+x_off]
                value_at_off = m[y+y_off, x+x_off]
                if label_at_off > 0 and label_at_off < labels[y, x] and value <= value_at_off:
                    labels[y, x] = label_at_off
            # If no labeled neighbours with a same or higher value were found, use as new seed
            if labels[y, x] == 0:
                labels[y, x] = next_label
                next_label += 1
    return labels_arr

# The function below is only for letting csegmentation() above known to other Python modules and not just Cython.
# The detect() function further down is thus commented because detect() function can now be imported from this module
# by other external modules
def detect_polarity(m ,threshold):
    return csegmentation(m, threshold)

# def detect(magnetogram, threshold=50):
#     magnetogram = np.ma.masked_invalid(magnetogram.astype(np.float64)).filled(0)
#     m_pos = np.ma.masked_less(magnetogram, 0).filled(0)
#     m_neg = np.ma.masked_less(-magnetogram, 0).filled(0)
#
#     labels_pos = detect_polarity(m_pos, float(threshold))
#     labels_neg = detect_polarity(m_neg, float(threshold))
#     labels_pos = np.ma.masked_less(skimage.measure.label(labels_pos, neighbors=4, background=0), 0).filled(0)
#     labels_neg = np.ma.masked_less(skimage.measure.label(labels_neg, neighbors=4, background=0), 0).filled(0)
#     skimage.morphology.remove_small_objects(labels_pos, 6, in_place=True)
#     skimage.morphology.remove_small_objects(labels_neg, 6, in_place=True)
#
#     return labels_pos, labels_neg
        
