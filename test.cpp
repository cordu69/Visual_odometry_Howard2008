#include<iostream>
#include <vector>
#include <string>
#include "opencv2/core/core.hpp"
#include <math.h>
using namespace cv;
using namespace std;
int main(){
    Mat a = Mat(3,4,CV_64FC1);
    Mat b = Mat(4,1,CV_64FC1);
    cout<<a.size()<<endl;
    cout<<b.size()<<endl;;
    a * b;
    return 0;
}