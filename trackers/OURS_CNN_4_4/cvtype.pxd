cdef extern from "opencv2/core/cvdef.h":
    cdef int CV_8U
    cdef int CV_8S
    cdef int CV_16U
    cdef int CV_16S
    cdef int CV_32S
    cdef int CV_32F
    cdef int CV_64F
    cdef int CV_MAKETYPE(int, int)
    cdef int CV_MAT_DEPTH(int)
    cdef int CV_MAT_CN(int)

cdef extern from "opencv2/core/mat.hpp" namespace "cv":
    cdef cppclass Mat:
        Mat()
        Mat(int, int, int)
        int type()
        void* data
        int cols
        int rows