

# Robust Tracking of Vessels in Oceanographic Airborne Images 

The code in this repository was used to generate the results of a benchmark of general purpose tracking algorithms on the maritime setting using airborne imagery. We used the OTB framework [1].

We also present a new approach [2] which is based on KCF [3] tracker and blob analysis. The evaluations are done either with CNN [4] or HOG [5] features.

# Results

<img src="http://imageshack.com/a/img924/648/scQuoA.png" width="425"/> <img src="http://imageshack.com/a/img921/4269/mLQFyd.png" width="425"/> 

<img src="http://imageshack.com/a/img924/6910/mcFCVi.png" width="425"/> <img src="http://imageshack.com/a/img922/6995/s2erO4.png" width="425"/> 

# Requirements
Requirements for the evaluation of all methods.

- Matlab (2015a used)
- Python 2.7
- Numpy
- OpenCV 3.1
- Caffe 
- Dlib 18.18
- VLFeat (for OTB)
- Matconvnet (for CF2 and MDNet)
- Mexopencv (for MUSTer)

Airborne Maritime Dataset.

# HOW-TO

Our method:

  0. Setup Python 2.7 with Numpy
  0. Install OpenCV 3.1 [(Tutorial)](http://www.pyimagesearch.com/2015/06/22/install-opencv-3-0-and-python-2-7-on-ubuntu/)
  0. Install Caffe with CUDA [(Tutorial)](http://caffe.berkeleyvision.org/installation.html)
  0. Setup CAFFE_ROOT environment variable to the folder that contains the Caffe framework and models.
  0. Download VGG-Net [4] from https://gist.github.com/ksimonyan/3785162f95cd2d5fee77 and put it into the '.../caffe/model/' folder.
  0. Download the Maritime Dataset from 
  0. For OURS_HOG you might need to recompile the HOG extraction code. To do this edit the setup.py file to point to your dlib folder and then run in the terminal: python setup.py
  
Other algorithms:

Usually the other algorithms should run if the required libraries are correctly installed. Either way each tracker has a readme file in its folder.

The CF2 [6] tracker requires that you download the ConvNet model and put it into '/trackers/CF2/model/' 


If you have any questions: jorgep.s.matos@gmail.com

# References

[1] Wu, Yi, Jongwoo Lim, and Ming-Hsuan Yang. "Online object tracking: A benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2013.

[2] J. Matos, A. Bernardino, and R. Ribeiro, “Robust tracking of vessels in oceanographic airborne
images,” in OCEANS’16 MTS/IEEE Monterey. MTS/IEEE.

[3] Henriques, João F., et al. "High-speed tracking with kernelized correlation filters." IEEE Transactions on Pattern Analysis and Machine Intelligence 37.3 (2015): 583-596.

[4] Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

[5] Felzenszwalb, Pedro F., et al. "Object detection with discriminatively trained part-based models." IEEE transactions on pattern analysis and machine intelligence 32.9 (2010): 1627-1645.

[6] Ma, Chao, et al. "Hierarchical convolutional features for visual tracking." Proceedings of the IEEE International Conference on Computer Vision. 2015.


