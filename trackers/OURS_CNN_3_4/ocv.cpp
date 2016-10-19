#include "ocv.h"
#include <iostream>
#include "opencv2/highgui.hpp"

#include <dlib/opencv.h>
#include <dlib/image_transforms.h>

using namespace std;
using namespace dlib;

void OCV::getMat(cv::Mat inmat, int cell_size)
{
    m = inmat;
    // Convert opencv image to dlib. Its a wrapper which means if opencv_image is deleted so is img
    if (m.channels() == 3) { 
        cv_image<bgr_pixel> img(m);
	extract_fhog_features(img, planar_hog, cell_size, 2, 2);
    }

    if (m.channels() == 1) {
	cv_image<float> img(m);
	extract_fhog_features(img, planar_hog, cell_size, 2, 2);
    }
    // Declare dlib array2d for planar_hog

    // Extrac fhog features
    
    // cout << m.size()  << " " << m.type() << endl;


}

cv::Mat OCV::returnMat(int index)
{
    // Declare a vector of opencv images to store the 31 channels of the resulting fhog
	//for (int i = 0; i < 31; i++){
        //	hog[i] = toMat(planar_hog[i]); 
	//}	
    return toMat(planar_hog[index]);
}

void OCV::showImg()
{
    cv::imshow("CPP image", m);
    cv::waitKey();
}
