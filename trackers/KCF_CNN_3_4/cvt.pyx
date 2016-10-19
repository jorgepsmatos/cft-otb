import numpy as np
from libc.string cimport memcpy

def nptype2cvtype(npty, nch):
    if npty == np.uint8:
        r = CV_8U
    elif npty == np.int8:
        r = CV_8S
    elif npty == np.uint16:
        r = CV_16U
    elif npty == np.int16:
        r = CV_16S
    elif npty == np.int32:
        r = CV_32S
    elif npty == np.float32:
        r = CV_32F
    elif npty == np.float64:
        r = CV_64F

    return CV_MAKETYPE(r, nch)

def cvtype2nptype(cvty):
    d = CV_MAT_DEPTH(cvty)
    nch = CV_MAT_CN(cvty)
    if d == CV_8U:
        r = np.uint8
    elif d == CV_8S:
        r = np.int8
    elif d == CV_16U:
        r = np.uint16
    elif d == CV_16S:
        r = np.int16
    elif d == CV_32S:
        r = np.int32
    elif d == CV_32F:
        r = np.float32
    elif d == CV_64F:
        r = np.float64
    return [r, nch]

cdef Mat nparray2cvmat(np.ndarray ary):
    if ary.ndim==3 and ary.shape[2]==1:
        ary = ary.reshape(ary.shape[0], ary.shape[1])

    cdef int r = ary.shape[0]
    cdef int c = ary.shape[1]
    if ary.ndim == 2:
        nch = 1
    else:
        nch = ary.shape[2]

    cdef Mat outmat = Mat(r, c, nptype2cvtype(ary.dtype, nch))
    memcpy(outmat.data, ary.data, ary.nbytes)
    return outmat

cdef np.ndarray cvmat2nparray(Mat &inmat):
    [ty, nch] = cvtype2nptype(inmat.type())
    if nch == 1:
        dim = [inmat.rows, inmat.cols]
    else:
        dim = [inmat.rows, inmat.cols, nch]

    cdef np.ndarray ary = np.empty(dim, dtype=ty)
    memcpy(ary.data, inmat.data, ary.nbytes)
    return ary