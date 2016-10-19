import cv2
import desiredResponse
import kernel
import numpy as np
import fhog
import os
os.environ['GLOG_minloglevel'] = '3'
#import caffe


# TODO : Get scale samples as gray, HoG and CNN features
def get_scale_sample(im, pos, base_target_sz, scale_factors, scale_window, scale_model_sz):

        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        number_of_scales = len(scale_factors)
        n = scale_model_sz[1] * scale_model_sz[0]
        out = np.zeros((n.astype(int), number_of_scales))

        for s in xrange(0, number_of_scales):
            patch_sz = np.floor(base_target_sz * scale_factors[s])

            xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2)
            ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2)

            xs[xs < 0] = 0
            ys[ys < 0] = 0
            xs[xs >= im.shape[1]] = im.shape[1] - 1
            ys[ys >= im.shape[0]] = im.shape[0] - 1

            im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

            im_patch_resized = cv2.resize(im_patch, (scale_model_sz[1].astype(int), scale_model_sz[0].astype(int)))

            out[:, s] = im_patch_resized.flatten(1) * scale_window[s]

        return out


def fhog_features(image, cell_size, cos_window):

        c = fhog.pyOCV()
        c.getMat(np.single(image), cell_size)

        features = np.zeros((np.floor(cos_window.shape[1]).astype(int), np.floor(cos_window.shape[0]).astype(int), 31))

        for i in xrange(0, 31):
            features[:, :, i] = cv2.resize(c.returnMat(i), (np.floor(cos_window.shape[1]).astype(int),
                                                            np.floor(cos_window.shape[0]).astype(int)))*cos_window

        return features


def gray_feat(im, cos_window):

        im = np.divide(im, 255.0)
        im = im - np.mean(im)
        features = cos_window * im

        return features


def image_segmentation(im, pos, patch_sz):
    xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2) - 1
    ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2) - 1

    xs[xs < 0] = 0
    ys[ys < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

    im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

    _, segmented_image = cv2.threshold(im_patch, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Remove the blobs originated from padding when the image is out-of-bounds, i.e. part of the ROI is outside the image frame.
    # segmented_image[ys == 0, :] = 0
    # segmented_image[:, xs == 0] = 0
    # segmented_image[ys == im.shape[0]-1, :] = 0
    # segmented_image[:, xs == im.shape[1]-1] = 0

    krnel = np.ones((3, 3), np.uint8)
    segmented_image = cv2.erode(segmented_image, krnel, iterations=1)

    return segmented_image


class Tracker:

    def __init__(self, im, params):

        self.parameters = params
        self.pos = self.parameters.init_pos
        self.target_sz = self.parameters.target_size

        if self.parameters.features == 'CNN':
            caffe.set_device(0)
            caffe.set_mode_gpu()

            caffe_root = os.environ['CAFFE_ROOT'] +'/'
            import sys
            sys.path.insert(0, caffe_root + 'python')

            self.net = caffe.Net(caffe_root + 'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt',
                                 caffe_root + 'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                                 caffe.TEST)

            # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
            self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
            self.transformer.set_transpose('data', (2, 0, 1))
            self.transformer.set_mean('data',
                                      np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
            self.transformer.set_raw_scale('data', 255)  # model operates on images in [0,255] range instead of [0,1]
            self.transformer.set_channel_swap('data', (0, 1, 2))  # model channels in BGR order instead of RGB

        # if self.target_sz[0] / 2.0 > self.target_sz[1]:
        #     self.target_sz[1] = self.target_sz[0]
        #     self.pos[1] = self.pos[1] - (self.target_sz[0] - self.target_sz[1]) / 2
        # elif self.target_sz[0] < self.target_sz[1]/2.0:
        #     self.target_sz[0] = self.target_sz[1]
        #     self.pos[0] = self.pos[0] - (self.target_sz[1] - self.target_sz[0]) / 2

        # Initial target size
        self.init_target_sz = self.parameters.target_size
        # target sz at scale = 1
        self.base_target_sz = self.parameters.target_size

        # Window size, taking padding into account
        self.window_sz = np.floor(np.array((max(self.base_target_sz),
                                            max(self.base_target_sz))) * (1 + self.parameters.padding))

        sz = self.window_sz
        sz = np.floor(sz / self.parameters.cell_size)
        self.l1_patch_num = np.floor(self.window_sz / self.parameters.cell_size)

        # Desired translation filter output (2d gaussian shaped), bandwidth
        # Proportional to target size
        output_sigma = np.sqrt(np.prod(self.base_target_sz)) * self.parameters.output_sigma_factor / self.parameters.cell_size
        self.yf = np.fft.fft2(desiredResponse.gaussian_response_2d(output_sigma, self.l1_patch_num))

        # Desired output of scale filter (1d gaussian shaped)
        scale_sigma = self.parameters.number_of_scales / np.sqrt(self.parameters.number_of_scales) * self.parameters.scale_sigma_factor
        self.ysf = np.fft.fft(desiredResponse.gaussian_response_1d(scale_sigma, self.parameters.number_of_scales))

        # Cosine window with the size of the translation filter (2D)
        self.cos_window = np.dot(np.hanning(self.yf.shape[0]).reshape(self.yf.shape[0], 1),
                                 np.hanning(self.yf.shape[1]).reshape(1, self.yf.shape[1]))

        # Cosine window with the size of the scale filter (1D)
        if np.mod(self.parameters.number_of_scales, 2) == 0:
            self.scale_window = np.single(np.hanning(self.parameters.number_of_scales + 1))
            self.scale_window = self.scale_window[1:]
        else:
            self.scale_window = np.single(np.hanning(self.parameters.number_of_scales))

        # Scale Factors [...0.98 1 1.02 1.0404 ...] NOTE: it is not a incremental value (see the scaleFactors values)
        ss = np.arange(1, self.parameters.number_of_scales + 1)
        self.scale_factors = self.parameters.scale_step**(np.ceil(self.parameters.number_of_scales / 2.0) - ss)

        # If the target size is over the threshold then downsample
        if np.prod(self.init_target_sz) > self.parameters.scale_model_max_area:
            self.scale_model_factor = np.sqrt(self.parameters.scale_model_max_area/np.prod(self.init_target_sz))
        else:
            self.scale_model_factor = 1

        self.scale_model_sz = np.floor(self.init_target_sz*self.scale_model_factor)

        self.currentScaleFactor = 1

        self.min_scale_factor = self.parameters.scale_step**np.ceil(np.log(np.max(5.0 / sz)) / np.log(self.parameters.scale_step))
        self.max_scale_factor = self.parameters.scale_step**np.floor(np.log(np.min(im.shape[0:-1] / self.base_target_sz)) / np.log(self.parameters.scale_step))

        self.confidence = np.array(())
        self.high_freq_energy = np.array(())
        self.psr = np.array(())

        # Flag that indicates if the track lost the target or not
        self.lost = False

        self.model_alphaf = None
        self.model_xf = None
        self.sf_den = None
        self.sf_num = None

    def detect(self, im):
        # Extract the features to detect.
        xt = self.translation_sample(im, self.pos, self.window_sz, self.parameters.cell_size,
                                     self.cos_window, self.parameters.features, self.currentScaleFactor)
        # 2D Fourier transform. Spatial domain (x,y) to frequency domain.
        xtf = np.fft.fft2(xt, axes=(0, 1))

        # Compute the feature kernel
        if self.parameters.kernel.kernel_type == 'Gaussian':
            kzf = kernel.gaussian_correlation(xtf, self.model_xf, self.parameters.kernel.kernel_sigma, self.parameters.features)
        else:
            kzf = kernel.linear_correlation(xtf, self.model_xf, self.parameters.features)

        # Translation Response map. The estimated location is the argmax (response).
        translation_response = np.real(np.fft.ifft2(self.model_alphaf * kzf, axes=(0, 1)))

        if self.parameters.debug:
            row_shift, col_shift = np.floor(np.array(translation_response.shape)/2).astype(int)
            r = np.roll(translation_response, col_shift, axis=1)
            r = np.roll(r, row_shift, axis=0)
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            response_map = cv2.applyColorMap(r, cv2.COLORMAP_AUTUMN)
            cv2.imshow('image', response_map)
            cv2.waitKey(1)

        # self.confidence = np.append(self.confidence, np.max(translation_response))
        row, col = np.unravel_index(translation_response.argmax(), translation_response.shape)

        # Peak-to-sidelobe ratio. If this value drops below 6.0 assume the track lost the target.
        # response_aux = np.copy(translation_response)
        # response_aux[row-3:row+3, col-3:col+3] = 0
        # self.psr = np.append(self.psr, (np.max(translation_response) - np.mean(response_aux))/(np.std(response_aux)))
        # if (np.max(translation_response) - np.mean(response_aux))/(np.std(response_aux)) < self.parameters.peak_to_sidelobe_ratio_threshold:
        #     self.lost = True

        if row > xtf.shape[0]/2:
            row = row - xtf.shape[0]
        if col > xtf.shape[1]/2:
            col = col - xtf.shape[1]

        # Compute the new estimated position in the image coordinates (0,0) <-> Top left corner
        self.pos = self.pos + self.parameters.cell_size * np.round(np.array((row, col)) * self.currentScaleFactor)

        # TODO: Implement Sun Reflection Detector

        # TRACK CENTERING PROCESS
        # Center the track on the target using blob analysis in the area defined by the bounding box.
        # Segment the image patch using OTSU's method.
        segmented_im = image_segmentation(im, self.pos, self.window_sz)


        # if True: #if (255 in segmented_im[:,0]) or (255 in segmented_im[0,:]) or (255 in segmented_im[:,-1]) or (255 in segmented_im[-1,:]):
        #     Blob_touches_border = True
        # # else:
        #     im = np.copy(segmented_im)
        #     im3 = np.copy(segmented_im)
        #     im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        #
        #     print len(contours)
        #     if len(contours) < 10:
        #
        #         x = np.zeros((len(contours))).astype(int)
        #         y = np.zeros((len(contours))).astype(int)
        #         w = np.zeros((len(contours))).astype(int)
        #         h = np.zeros((len(contours))).astype(int)
        #         centers = np.zeros((len(contours), 2)).astype(int)
        #         centers_momments = np.zeros((len(contours), 2)).astype(int)
        #         areas = np.zeros((len(contours))).astype(int)
        #
        #         for n,cnt in enumerate(contours):
        #             x[n], y[n], w[n], h[n] = cv2.boundingRect(cnt)
        #             areas[n] = cv2.contourArea(cnt)
        #             centers[n,:] = np.array((x[n]+w[n]/2, y[n]+h[n]/2))
        #             M = cv2.moments(cnt)
        #             cx = int(M['m10']/M['m00'])
        #             cy = int(M['m01']/M['m00'])
        #             centers_momments[n,:] = np.array((cx, cy))
        #
        #         # TODO: check the blob area
        #         x = x[areas>25]
        #         y = y[areas>25]
        #         w = w[areas>25]
        #         h = h[areas>25]
        #         centers = centers[areas>25]
        #
        #         if len(centers)>0:
        #
        #             p = np.array((col,row)) + self.window_sz / 2.0 # np.array(segmented_im.shape) / 2
        #
        #             disp = centers - p
        #             blob_center_arg = np.argmin((disp*disp).sum(1))
        #
        #             cv2.circle(im3, (centers[blob_center_arg,0], centers[blob_center_arg,1]), 4,0)
        #             cv2.rectangle(im3,(x[blob_center_arg],y[blob_center_arg]),(x[blob_center_arg]+w[blob_center_arg],y[blob_center_arg]+h[blob_center_arg]),255,2)
        #
        #             cv2.imshow("Tst3", im3)
        #             cv2.waitKey(1)
        #             #print cv2.contourArea(contours[blob_center_arg])
        #
        #             # TODO: Check last two conditions. Might be wrong
        #             # if x[blob_center_arg]<=1 or y[blob_center_arg]<=1 or x[blob_center_arg]+w[blob_center_arg] >= np.size(im, 1)-2 or y[blob_center_arg]+h[blob_center_arg]>= np.size(im, 0)-2:
        #             #     Blob_touching_border = True
        #
        #             new_pos = np.array((centers[blob_center_arg,1], centers[blob_center_arg,0])) + self.pos - self.window_sz/2.0
        #             self.pos = new_pos
        #             if 1.5*h[blob_center_arg] > 25:
        #                 self.target_sz[0] = 1.5*h[blob_center_arg]
        #             else:
        #                 self.target_sz[0] = 25
        #
        #             if 1.5*w[blob_center_arg] > 25:
        #                 self.target_sz[1] = 1.5*w[blob_center_arg]
        #             else:
        #                 self.target_sz[1] = 25
        #
        #             # TODO: check if the blob bounding box touches any border



        ## OLD BLOB DETECTION (paper version)

        # Blob touches border
        if (255 in segmented_im[:,0]) or (255 in segmented_im[0,:]) or (255 in segmented_im[:,-1]) or (255 in segmented_im[-1,:]):
            if self.parameters.visualization:
                cv2.imshow("Keypoints", segmented_im)
                cv2.waitKey(1)
        else: # No blob touching the border

            # Compute the 2D fourier transform to calculate the high frequency energy. This way it is possible to detect if
            # target is in a region of sun reflection. If so the track centering process cannot be applied.
            # segmented_im_f = np.fft.fft2(segmented_im)
            # row_shift, col_shift = np.floor(np.array(segmented_im_f.shape)/2).astype(int)
            # resp = np.roll(np.abs(segmented_im_f), col_shift, axis=1)
            # resp = np.roll(resp, row_shift, axis=0)
            #
            # response_high_freq = resp[np.floor(resp.shape[0]/2-5).astype(int):np.ceil(resp.shape[0]/2+5).astype(int),
            #                           np.floor(resp.shape[1]/2-5).astype(int):np.ceil(resp.shape[1]/2+5).astype(int)]
            #
            # self.high_freq_energy = np.append(self.high_freq_energy,
            #                                   np.sum(response_high_freq))
            #
            # if self.parameters.debug:
            #     print self.high_freq_energy[-1]

            # Only recenter the track if the high frequency energy is below the threshold
            #if self.high_freq_energy[-1] < self.parameters.high_freq_threshold:

            # Setup SimpleBlobDetector parameters.

            blob_params = cv2.SimpleBlobDetector_Params()
            #blob_params.blobColor = 255
            blob_params.minArea = 0.0
            blob_params.minDistBetweenBlobs = 0.0
            blob_params.filterByColor = False
            blob_params.filterByConvexity = False
            blob_params.filterByInertia = False
            blob_params.filterByArea = False
            blob_params.minRepeatability = 1

            # Find the blobs of the segmented image
            blob_detector = cv2.SimpleBlobDetector_create(blob_params)
            keypoints = blob_detector.detect(segmented_im)


            im = np.copy(segmented_im)
            im3 = np.copy(segmented_im)
            im2, contours, hierarchy = cv2.findContours(im,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

            x = np.zeros((len(contours))).astype(int)
            y = np.zeros((len(contours))).astype(int)
            w = np.zeros((len(contours))).astype(int)
            h = np.zeros((len(contours))).astype(int)
            centers = np.zeros((len(contours), 2)).astype(int)
            areas = np.zeros((len(contours))).astype(int)

            for n,cnt in enumerate(contours):
                x[n], y[n], w[n], h[n] = cv2.boundingRect(cnt)
                areas[n] = cv2.contourArea(cnt)
                M = cv2.moments(cnt)
                if M['m00'] == 0:
                    cx = 0
                    cy = 0
                else:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                centers[n,:] = np.array((cx, cy))


            # print len(contours), len(keypoints)

            # Only recenter the target if the number of blobs is 1
            # The above statement implies that the recentering process invalidates the algorithm if there are 3+
            # vessels very close (inside the BB area of each other)
            if len(keypoints) == 1 or len(keypoints) == 2:

                # Find with keypoints
                #euclidean_distance = np.array(())
                #for blob in keypoints:
                #    euclidean_distance = np.append(euclidean_distance,
                #                                   np.linalg.norm(-np.array(blob.pt)[::-1] + self.window_sz / 2.0 + np.array((col,row)) ))   # ERROR (closest to center and not to new estimation)
                # From the available blobs choose the one closest to the tracking algorithm estimation.
                #new_pos = np.array(keypoints[euclidean_distance.argmin()].pt)[::-1] + self.pos - self.window_sz/2.0
                #new_size = 1.5*np.round(np.array((keypoints[euclidean_distance.argmin()].size, keypoints[euclidean_distance.argmin()].size)))

                # if keypoints[euclidean_distance.argmin()].size > 10.0:
                #     self.pos = new_pos
                #     if new_size[0]/self.target_sz[0] < 2 and new_size[0]/self.target_sz[0] > 0.5:
                #         self.target_sz[0] = new_size[0]
                #     if new_size[1]/self.target_sz[1] < 2 and new_size[1]/self.target_sz[1] > 0.5:
                #         self.target_sz[1] = new_size[1]

                # Find with contours
                p = self.window_sz / 2.0 # np.array(segmented_im.shape) / 2

                disp = centers - p
                blob_center_arg = np.argmin((disp*disp).sum(1))

                new_pos = np.array((centers[blob_center_arg,1], centers[blob_center_arg,0])) + self.pos - self.window_sz/2.0
                new_size = 1.5*np.array((h[blob_center_arg], w[blob_center_arg]))
                if areas[blob_center_arg] > 80:
                    self.pos = new_pos
                    if new_size[0]/self.target_sz[0] < 1.2 and new_size[0]/self.target_sz[0] > 0.8:
                        self.target_sz[0] = new_size[0]
                    elif new_size[0]/self.target_sz[0] >= 1.2:
                        self.target_sz[0] = 1.2*self.target_sz[0]
                    elif new_size[0]/self.target_sz[0] <= 0.8:
                        self.target_sz[0] = 0.8*self.target_sz[0]

                    if new_size[1]/self.target_sz[1] < 1.2 and new_size[1]/self.target_sz[1] > 0.8:
                        self.target_sz[1] = new_size[1]
                    elif new_size[1]/self.target_sz[1] >= 1.2:
                        self.target_sz[1] = 1.2*self.target_sz[1]
                    elif new_size[1]/self.target_sz[1] <= 0.8:
                        self.target_sz[1] = 0.8*self.target_sz[1]

                if self.parameters.visualization:
                    cv2.circle(im3, (centers[blob_center_arg,0], centers[blob_center_arg,1]), 4,0)
                    cv2.imshow("Keypoints", im3)
                    cv2.waitKey(1)



            # if self.parameters.visualization:
            #
            #     im_with_keypoints = cv2.drawKeypoints(segmented_im, keypoints, np.array([]), (0, 0, 255),
            #                                           cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #     cv2.imshow("Keypoints", im_with_keypoints)
            #     cv2.waitKey(1)

        # # Extract the scale samples
        # xs = get_scale_sample(im, self.pos, self.base_target_sz, self.currentScaleFactor * self.scale_factors, self.scale_window, self.scale_model_sz)
        #
        # # Compute the fourier transform from scale space to frequency
        # xsf = np.fft.fft(xs, self.parameters.number_of_scales, axis=1)
        #
        # # Scale Response map. The estimated scale is the argmax (scale_response).
        # scale_response = np.real(np.fft.ifft(np.sum(self.sf_num*xsf, axis=0) / (self.sf_den + self.parameters.lmbda)))
        # recovered_scale = scale_response.argmax()
        #
        # # Compute the new scale factor
        # self.currentScaleFactor = self.currentScaleFactor * self.scale_factors[recovered_scale]
        # if self.currentScaleFactor < self.min_scale_factor:
        #     self.currentScaleFactor = self.min_scale_factor
        # elif self.currentScaleFactor > self.max_scale_factor:
        #     self.currentScaleFactor = self.max_scale_factor

        # Compute the new target size
        #self.target_sz = np.floor(self.base_target_sz * self.currentScaleFactor)

        # Return the position (center of mass in image coordinates) and the lost flag.
        return np.array([self.pos[0], self.pos[1], self.target_sz[0], self.target_sz[1]]), self.lost

    def train(self, im, start=True):
        # Extract the features to train.
        xl = self.translation_sample(im, self.pos, self.window_sz,
                                     self.parameters.cell_size, self.cos_window,
                                     self.parameters.features, self.currentScaleFactor)
        # 2D Fourier transform. Spatial domain (x,y) to frequency domain.
        xlf = np.fft.fft2(xl, axes=(0, 1))

        # Compute the features kernel
        if self.parameters.kernel.kernel_type == 'Gaussian':
            kf = kernel.gaussian_correlation(xlf, xlf, self.parameters.kernel.kernel_sigma, self.parameters.features)
        else:
            kf = kernel.linear_correlation(xlf, xlf, self.parameters.features)

        # Compute the optimal translation filter
        alphaf = self.yf / (kf + self.parameters.lmbda)
        # alpha= np.real(np.fft.ifft2(alphaf, axes=(0,1)))
        # row_shift, col_shift = np.floor(np.array(alphaf.shape)/2).astype(int)
        # alpha = np.roll(alpha, col_shift,axis=1)
        # alpha = np.roll(alpha,row_shift, axis=0)
        # cv2.imshow("alpha", alpha)
        # cv2.waitKey(0)
        # Extract the scale samples
        xs = get_scale_sample(im, self.pos, self.base_target_sz, self.currentScaleFactor * self.scale_factors, self.scale_window, self.scale_model_sz)

        # Compute the fourier transform from scale space to frequency
        xsf = np.fft.fft(xs, self.parameters.number_of_scales, axis=1)

        # Compute the optimal scale filter
        new_sf_num = self.ysf * np.conjugate(xsf)
        new_sf_den = np.sum(xsf * np.conjugate(xsf), axis=0)

        # If first frame create the model
        if start:
            self.model_alphaf = alphaf
            self.model_xf = xlf
            self.sf_den = new_sf_den
            self.sf_num = new_sf_num
        # Update the model in the consecutive frames.
        else:
            self.sf_den = (1 - self.parameters.learning_rate) * self.sf_den + self.parameters.learning_rate * new_sf_den
            self.sf_num = (1 - self.parameters.learning_rate) * self.sf_num + self.parameters.learning_rate * new_sf_num
            self.model_alphaf = (1 - self.parameters.learning_rate) * self.model_alphaf + self.parameters.learning_rate * alphaf
            self.model_xf = (1 - self.parameters.learning_rate) * self.model_xf + self.parameters.learning_rate * xlf

    def translation_sample(self, im, pos, model_sz, cell_size, cos_window, features, current_scale_factor):
        patch_sz = np.floor(model_sz * current_scale_factor)

        if features != 'CNN':
            im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

        if patch_sz[0] < 1:
            patch_sz[0] = 2
        if patch_sz[1] < 1:
            patch_sz[1] = 2

        xs = np.floor(pos[1]) + np.arange(1, patch_sz[1]+1) - np.floor(patch_sz[1]/2) - 1
        ys = np.floor(pos[0]) + np.arange(1, patch_sz[0]+1) - np.floor(patch_sz[0]/2) - 1

        xs[xs < 0] = 0
        ys[ys < 0] = 0
        xs[xs >= im.shape[1]] = im.shape[1] - 1
        ys[ys >= im.shape[0]] = im.shape[0] - 1

        im_patch = im[np.ix_(ys.astype(int), xs.astype(int))]

        im_patch = cv2.resize(im_patch, (model_sz[1].astype(int), model_sz[0].astype(int)))

        # cv2.imshow("Sample patch", im_patch)
        # cv2.waitKey(0)

        if features == 'HoG':
            out = fhog_features(im_patch, cell_size, cos_window)
        elif features == 'CNN':
            out = self.cnn_features(im_patch, cos_window)
        else:
            out = gray_feat(im_patch, cos_window)

        return out

    def cnn_features(self, im_patch, cos_window):

        im_patch = np.asarray(im_patch[:, :])
        im_patch = im_patch.astype(np.float32)/255
        im_patch = im_patch[:, :, np.array((2, 1, 0))]

        self.net.blobs['data'].reshape(1, 3, 224, 224)
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im_patch)

        self.net.forward()
        feat = self.net.blobs['conv4_4'].data[0, :]

        feat = feat.transpose(1, 2, 0)

        feat = cv2.resize(feat, (np.floor(cos_window.shape[1]).astype(int), np.floor(cos_window.shape[0]).astype(int)))
        for i in xrange(512):
            feat[:, :, i] = feat[:, :, i] * cos_window

        return feat
