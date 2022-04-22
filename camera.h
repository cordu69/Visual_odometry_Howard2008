#pragma once
#include "opencv2/core/core.hpp"
#include <vector>
#include <string>
#include "optimization.h"

using namespace cv;
using std::vector;
using namespace alglib;
double left_intrisics[3][3] = {
    {264.3375, 0, 335.41},
    {0, 264.32, 184.21375},
    {0, 0, 1}
};

double right_intrisics[3][3] = {
    {264.605, 0, 338.8375},
    {0, 264.495, 193.22925},
    {0, 0, 1}
};


// Parameters
double left_distorsion[1][5] = { -0.0405834, 0.0110733, 0.00010938, -0.000474769, -0.00533756 };
double right_distorsion[1][5] = { -0.0411496, 0.0098559, -0.000228316, -8.87648e-05, -0.00476447 };
double translation_vector[1][3] = { -119.537, 0, 0 };
double rotation_vector[1][3] = { 0.00150594 * (3.14159 / 180.0), 0.00350327 * (3.14159 / 180.0), 0.000433968 * (3.14159 / 180.0) };

// Mat declaration given parameters
Mat cam_l_intrisics = Mat(3, 3, CV_64FC1, &left_intrisics);
Mat cam_r_intrisics = Mat(3, 3, CV_64FC1, &right_intrisics);
Mat cam_l_distorsion = Mat(1, 5, CV_64FC1, &left_distorsion);
Mat cam_r_distorsion = Mat(1, 5, CV_64FC1, &right_distorsion);
Mat T = Mat(3, 1, CV_64FC1, &translation_vector);
Mat rotation = Mat(3, 1, CV_64FC1, &rotation_vector);

// Create stereo matching class
Ptr<cv::StereoBM> sbm_left = StereoBM::create();

// Stereo matcher 
Ptr<StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(sbm_left);
// Postprocessing filter
Ptr<cv::ximgproc::DisparityWLSFilter> disparity_filter = cv::ximgproc::createDisparityWLSFilter(sbm_left);
// FAST detector features
Ptr<FastFeatureDetector> fast_detector = cv::FastFeatureDetector::create();
// Feature matching termination criteria
TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);

