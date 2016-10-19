import caffe
import cv2
import numpy as np


im_patch = cv2.imread("/home/jorgematos/Dropbox/Paper/vessel3.jpg", 1)

caffe.set_device(0)
caffe.set_mode_gpu()

caffe_root = '/home/jorgematos/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')

net = caffe.Net(caffe_root + 'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers_deploy.prototxt',
                     caffe_root + 'models/3785162f95cd2d5fee77/VGG_ILSVRC_19_layers.caffemodel',
                     caffe.TEST)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data',
                          np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_raw_scale('data', 255)  # model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (0,1,2))


im_patch = np.asarray(im_patch[:, :])
im_patch = im_patch.astype(np.float32)/255
im_patch = im_patch[:, :, np.array((2, 1, 0))]

net.blobs['data'].reshape(1, 3, 224, 224)
net.blobs['data'].data[...] = transformer.preprocess('data', im_patch)

net.forward()
feat = net.blobs['conv3_4'].data[0, :]

feat = feat.transpose(1, 2, 0)

feat = cv2.resize(feat, (224,224))

cv2.imwrite("cnn_feature.jpg", (feat[:,:,21]/feat[:,:,21].max())*255)

cv2.imshow("Window", feat[:,:,21]/feat[:,:,21].max())
cv2.waitKey(0)

# for i in range(256):
#      cv2.imshow("Window", feat[:,:,i]/feat[:,:,i].max())
#      cv2.waitKey(0)
#      print i