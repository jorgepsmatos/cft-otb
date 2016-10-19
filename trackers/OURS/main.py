import cv2
import loadVideo
import numpy as np
import time
import tracker
import sys, getopt


import os
os.environ['GLOG_minloglevel'] = '2'


# Structure with general parameters (explanation further down)
class Params:
    def __init__(self):
        self.padding = 1.5                              # extra area surrounding the target
        self.output_sigma_factor = 0.1                  # standard deviation for the desired translation filter output
        self.lmbda = 1e-4                               # regularization weight
        self.learning_rate = 0.02

        self.scale_sigma_factor = 1.0/4                 # standard deviation for the desired scale filter output
        self.number_of_scales = 1                      # number of scale levels
        self.scale_step = 1.02                          # Scale increment factor
        self.scale_model_max_area = 512                 # maximum scale

        self.features = "HoG"
        self.cell_size = 4.0
        self.high_freq_threshold = 2 * 10 ** 66
        self.peak_to_sidelobe_ratio_threshold = 0       # Set to 0 to disable (Detect if the target is lost)
        self.rigid_transformation_estimation = False    # Try to detect camera rotation

        self.visualization = False
        self.debug = False

        self.init_pos = np.array((0, 0))
        self.pos = np.array((0, 0))
        self.target_size = np.array((0, 0))
        self.img_files = None
        self.video_path = None

        self.kernel = Kernel()


# Structure with kernel parameters
class Kernel:
    def __init__(self):
        self.kernel_type = "Linear"
        self.kernel_sigma = 0.5


def main(argv):

    try:
        opts, args = getopt.getopt(argv,"f:v:x:y:w:h:",["features=","visualization=","x=","y=","width=", "height="])
    except getopt.GetoptError:
        print 'Error: Command example: python main.py -f <features> -v <1>'
        sys.exit(2)

    parameters = Params()

    if parameters.features == 'CNN':
        parameters.learning_rate = 0.02			        # tracking model learning rate (denoted "eta" in the paper)
        parameters.kernel.kernel_type = 'Linear'        # Choose kernel type (Ex: 'Linear', 'Gaussian')
        parameters.kernel.kernel_sigma = 0.5            # standard deviation for gaussian kernel
        parameters.cell_size = 4.0                      # The CNN features downsample the image by a factor of 4.
    elif parameters.features == 'Gray':
        parameters.learning_rate = 0.075                # tracking model learning rate (denoted "eta" in the paper)
        parameters.kernel.kernel_type = 'Gaussian'      # Choose kernel type (Ex: 'Linear', 'Gaussian')
        parameters.kernel.kernel_sigma = 0.2            # standard deviation for gaussian kernel
        parameters.cell_size = 1                        # The image intensity features don't downsample
    elif parameters.features == 'HoG':
        parameters.learning_rate = 0.02                 # tracking model learning rate
        parameters.kernel.kernel_type = 'Gaussian'      # Choose kernel type (Ex: 'Linear', 'Gaussian')
        parameters.kernel.kernel_sigma = 0.5            # standard deviation for gaussian kernel
        parameters.cell_size = 4.0                      # The HoG features downsample the image by a factor of 4.
    else:
        print 'Error selecting Image Features'
        exit(-1)

    with open('imagesPath.txt') as f:
        img_files = f.read().splitlines()

    initRect = np.loadtxt('initRect.txt', delimiter=",")

    pos = np.array((initRect[1], initRect[0]))
    targetSz = np.array((initRect[3], initRect[2]))
    Params.init_pos = np.floor(pos) + np.floor(targetSz / 2)                         # initial center position
    Params.target_size = np.floor(targetSz)                                        # size of target

    positions = np.zeros((len(img_files),4))
    positions[0,:] = initRect

    parameters.init_pos = np.floor(pos) + np.floor(targetSz / 2)                               # initial center position
    parameters.pos = parameters.pos
    parameters.target_size = np.floor(targetSz)                                                # size of target
    parameters.img_files = img_files                                                           # list of all image files
    #parameters.video_path = video_path                                                         #  Path to video

    for opt, arg in opts:
        if opt in ("-f", "--features"):
            Params.features = arg
        if opt in ("-v", "--visualization"):
            Params.visualization = arg

    num_frames = len(img_files)
    results = np.zeros((num_frames, 4))

    start = time.clock()                                # Start timer

    # For each frame
    for frame in xrange(num_frames):

        # Read the image
        im = cv2.imread(img_files[frame], 1)

        # Initialize the tracker using the first frame
        if frame == 0:
            tracker1 = tracker.Tracker(im, parameters)
            tracker1.train(im, True)
            results[frame, :] = np.array((pos[0], pos[1], targetSz[0], targetSz[1]))
        else:
            results[frame, :], lost = tracker1.detect(im)     # Detect the target in the next frame
            if lost:
                break
            tracker1.train(im, False)           # Update the model with the new infomation
            positions[frame,:] = np.round((results[frame, 1]-results[frame, 3]/2,     # Draw a rectangle in the estimated location and show the result
                                          results[frame, 0]-results[frame, 2]/2,
                                          results[frame, 3],
                                          results[frame, 2]))

        if parameters.visualization:
            # Draw a rectangle in the estimated location and show the result
            cvrect = np.array((results[frame, 1]-results[frame, 3]/2,
                               results[frame, 0]-results[frame, 2]/2,
                               results[frame, 1]+results[frame, 3]/2,
                               results[frame, 0]+results[frame, 2]/2))
            cv2.rectangle(im, (cvrect[0].astype(int), cvrect[1].astype(int)),
                              (cvrect[2].astype(int), cvrect[3].astype(int)), (255, 0, 0), 2)
            cv2.imshow('Window', im)
            cv2.waitKey(1)

    end = time.clock()
    fps = num_frames/(end - start)
    # print fps, "FPS"

    np.savetxt('results.txt', results, delimiter=',', fmt='%d')
    np.savetxt('output.txt', positions, delimiter=',', fmt='%d')

if __name__ == "__main__":
    main(sys.argv[1:])