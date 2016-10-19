from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

libdr = ['/usr/local/lib']
incdr = [numpy.get_include(), "/usr/local/include/", "/home/jorgematos/dlib-18.18"]

ext_modules = [
    Extension("cvt", ["cvt.pyx"],
              language = "c++",
              extra_compile_args=["-std=c++11"],
              include_dirs=incdr,
              library_dirs = libdr,
              libraries=["opencv_core"]),
    Extension("fhog", ["fhog.pyx", "ocv.cpp"],
              language = "c++",
              extra_compile_args=["-std=c++11","-I.. /home/jorgematos/dlib-18.18/dlib/all/source.cpp -I//home/jorgematos/dlib-18.18/","-lpthread","-lX11"],
              include_dirs=incdr,
              library_dirs = libdr,
              libraries=['opencv_core', 'opencv_highgui'])
    ]

setup(
  name = 'app',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
