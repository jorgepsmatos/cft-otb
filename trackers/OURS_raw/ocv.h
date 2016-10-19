#ifndef OCV_H_
#define OCV_H_

#include "opencv2/core.hpp"
#include <dlib/image_transforms.h>

class OCV
{
public:
    cv::Mat m;
    dlib::array<dlib::array2d<float> > planar_hog;
    void getMat(cv::Mat inmat, int incell_size);
    cv::Mat returnMat(int inindex);
    void showImg();
};

#endif /* OCV_H_ */
