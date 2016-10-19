import fhog
import numpy as np
import cv2


def fhog_features(image, cell_size):

    c = fhog.pyOCV()
    c.getMat(np.single(image), cell_size)

    features = np.zeros((np.floor(image.shape[1]).astype(int), np.floor(image.shape[0]).astype(int), 31))

    for i in xrange(0, 31):
        features[:, :, i] = cv2.resize(c.returnMat(i), (np.floor(image.shape[1]).astype(int),
                                                        np.floor(image.shape[0]).astype(int)))

    return features

image = cv2.imread("/home/jorgematos/Dropbox/Paper/vessel3.jpg", 0)

features = fhog_features(image, 4)

for i in xrange(0,31):
    name = "HoG_channel" + str(i) + ".jpg"
    feat = features[:,:,i] * 255
    cv2.imwrite(name, feat)
    # cv2.imshow("feate", feat)
    # cv2.waitKey(0)
