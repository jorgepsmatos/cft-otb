from cvt cimport *

cdef extern from "ocv.h":
    cdef cppclass OCV:
        Mat m
        OCV()
        void getMat(Mat, int)
        Mat returnMat(int)
        void showImg()

cdef class pyOCV:
    cdef OCV *classptr

    def __cinit__(self):
        self.classptr = new OCV()

    def __dealloc__(self):
        del self.classptr

    def getMat(self, inary, incell_size):
        cdef Mat tmp = nparray2cvmat(inary)
        self.classptr.getMat(tmp, incell_size)

    def returnMat(self, inindex):
        cdef Mat tmp = self.classptr.returnMat(inindex)
        return cvmat2nparray(tmp)

    def showImg(self):
        self.classptr.showImg()
