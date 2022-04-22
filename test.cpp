#include<iostream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/ximgproc/weighted_median_filter.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <opencv2/video/tracking.hpp>

#include <math.h>
using namespace cv;

int main(){
    nlopt_opt opt;
    minlmstate state;
    opt = nlopt_create(NLOPT_LD_MMA, 2);

    return 0;
}