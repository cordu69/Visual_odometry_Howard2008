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
#include "camera.h"
#include <opencv2/video/tracking.hpp>
#include <optimization.h>

using namespace std::chrono;
using namespace std;
using namespace cv;


// Compute tuples of 2d points for each relevant feature.
vector<Mat> feature_2d_prev;
vector<Mat> feature_2d_current;

// Compute tuples of 3d points for each relevant feature.
vector<Mat> feature_3d_prev;
vector<Mat> feature_3d_current;
// P matrix project 3d to 2d
Mat P;

Mat unidistort_zed_image(Mat src, string cam) {
    /**
     * @brief Unidistort an image tocompensate for the lens distorsion.
     *
     * @param src the image
     * @param cam string left or right camera
     */

    Mat out;

    if (cam.compare("left") == 0) {
        undistort(src, out, cam_l_intrisics, cam_l_distorsion);
    }
    else {
        undistort(src, out, cam_r_intrisics, cam_r_distorsion);
    }

    return out;
}

vector<Mat> compute_transformations_calibrated(Mat example) {
    /**
     * @brief Find the transformations used to map the stereo images perspectives to the same plane where features are epipolar (aligned on x). This uses the camera params.
     *
     */
     // Transform a rotation vector to matrix 

     // Calculate the full rotation matrix by combining the rotation transform on each exis.
    Mat R, R1, R2, P1, P2, Q;
    // From rotation vector to rotation matrix.
    Rodrigues(rotation, R);

    stereoRectify(cam_l_intrisics, cam_l_distorsion, cam_r_intrisics, cam_r_distorsion, example.size(), R, T, R1, R2, P1, P2, Q);

    return vector<Mat>({ R1, R2, Q, P1 });
}



tuple<Mat, Mat> compute_undistorted_images(VideoCapture cap) {
    /*
    * @brief Compute the undistorted images of the stereo camera given the video cap.
    * 
    * @param video_cap : the video capture object.
    */

    // Read the first frame to have a pair
    // frame from zed camera left and right
    Mat frame;
    cap >> frame;

    Mat left_gray, right_gray;
    Mat left_img = frame(Rect(0, 0, frame.cols / 2, frame.rows));
    Mat right_img = frame(Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

    // make the grayscale converison
    cvtColor(left_img, left_gray, COLOR_BGR2GRAY);
    cvtColor(right_img, right_gray, COLOR_BGR2GRAY);

    Mat left_gray_undistorted = unidistort_zed_image(left_gray, "left");
    Mat right_gray_undistorted = unidistort_zed_image(right_gray, "right");

    return make_tuple(left_gray_undistorted, right_gray_undistorted);
}

tuple<Mat, Mat> compute_rectified_images(Mat left_undistorted, Mat right_undistorted, Mat H1, Mat H2) {
    Mat out_right;
    Mat out_left;

    // Warp the perspective of the images to the common plane using the rotation matrices
    warpPerspective(left_undistorted, out_left, H1, Size(left_undistorted.cols, left_undistorted.rows), INTER_LINEAR);
    warpPerspective(right_undistorted, out_right, H2, Size(left_undistorted.cols, left_undistorted.rows), INTER_LINEAR);

    return make_tuple(out_left, out_right);
}

Mat get_disparity_map(Mat current_rectified_left, Mat current_rectified_right) {
    /**
    @ brief : Compute the disparity map given the rectified frames.
    * 
    * @param current_rectified_left : Left frame of the stereo camera rectified on the X axis.
    * @param current_rectified_right : Right frame of the stereo camera rectified on the X axis.
    */
    // Compute disparity and filter it 
    Mat disparity_left, disparity_right, filtered_disparity, disparity_result;
    sbm_left->compute(current_rectified_left, current_rectified_right, disparity_left);
    //right_matcher->compute(current_rectified_right, current_rectified_left, disparity_right);
    //disparity_filter->filter(disparity_left, current_rectified_left, filtered_disparity, disparity_right);
    //cv::ximgproc::getDisparityVis(filtered_disparity, disparity_result);

    return disparity_left;
}

vector<Mat> compute_features_pointcloud(vector<Point2f> features, Mat disparity, Mat Q) {
    /**
    * @brief : Compute the pointcloud of the found features.
    * 
    * @param : features - the detected features from a chosen frame.
    * @parma : disparity - the disparity map of the frame.
    * @param : Q - the reprojection matrix from 2D to 3D.
    */

    vector<Mat> features_pointcloud;
    for (const auto& point : features) {
        double d = disparity.at<ushort>(round(point.y), round(point.x));
        Mat pixel_vector = Mat(1, 4, CV_64FC1);
        pixel_vector.at<double>(0, 0) = double(point.x);
        pixel_vector.at<double>(0, 1) = double(point.y);
        pixel_vector.at<double>(0, 2) = double(d);
        pixel_vector.at<double>(0, 3) = double(1);
        Mat real_world_vector = pixel_vector * Q;
        features_pointcloud.push_back(real_world_vector);
    }

    return features_pointcloud;
}

Mat compute_adjacency_matrix(vector<Mat> previous_keypoints_pc, vector<Mat> matched_keypoints_pc) {
    /**
    * @brief : Compute the adjiency matrix between each feature pointclouds at current timestamp and future timestamp.
    * 
    * @param : left_previous_keypoints_pc - The pointcloud of all features from the previous frame.
    * @param : left_current_keypoints_pc - The pointcloud of all features from the current frame.
    * 
    * @return : An adjiency matrix where the distance 
    */

    Mat adjacency_matrix = Mat(matched_keypoints_pc.size(), matched_keypoints_pc.size(), CV_8UC1);
    Mat distance_matrix_previous = Mat(matched_keypoints_pc.size(), matched_keypoints_pc.size(), CV_64FC1);
    Mat distance_matrix_current = Mat(matched_keypoints_pc.size(), matched_keypoints_pc.size(), CV_64FC1);
    const double threshold = 0.5;

    // Get the previous matrix distances between each feature.
    int i = 0;
    int j = 0;
    for (const auto& x : previous_keypoints_pc) {
        for (const auto& y : previous_keypoints_pc) {
            double distance = sqrt(pow(x.at<double>(0,1) - y.at<double>(0,1), 2) + pow(x.at<double>(0,2) - y.at<double>(0, 2),2) + pow(x.at<double>(0, 3) - y.at<double>(0, 3), 2));
            distance_matrix_previous.at<double>(i, j) = distance;
            j += 1;
        }
        i += 1;
        j = 0;
    }
    // Get the current matri of distances between each feature.
    i = 0;
    j = 0;
    for (const auto& x : matched_keypoints_pc) {
        for (const auto& y : matched_keypoints_pc) {
            double distance = sqrt(pow(x.at<double>(0, 1) - y.at<double>(0, 1), 2) + pow(x.at<double>(0, 2) - y.at<double>(0, 2), 2) + pow(x.at<double>(0, 3) - y.at<double>(0, 3), 2));
            distance_matrix_current.at<double>(i, j) = distance;
            j += 1;
        }
        i += 1;
        j = 0;
    }

    // Compute the adjceny matrix.
    for (i = 0; i < matched_keypoints_pc.size(); i++) {
        for (j = 0; j < matched_keypoints_pc.size(); j++) {
            if (distance_matrix_previous.at<double>(i, j) - distance_matrix_current.at<double>(i, j) < threshold) {
                adjacency_matrix.at<uchar>(i, j) = 1;
            }
            else {
                adjacency_matrix.at<uchar>(i, j) = 0;
            }
        }
    }

    return adjacency_matrix;
}

vector<int> validate_keypoints(vector<Point2f> keypoints, vector<uchar> status, Mat disparity) {
    /**
    * @brief : Only take the index of the matched points with are validates.
    * 
    */

    vector<int> correct_kp;

    for (int i = 0; i < status.size(); i++) {
        if (status[i] == 1) {
            if (keypoints[i].x > 0 && keypoints[i].y > 0 && keypoints[i].y < disparity.size().height && keypoints[i].x < disparity.size().width) {
                correct_kp.push_back(i); // the index of the correct kp
            }
        }
    }

    return correct_kp;
}

int get_maximum_node(Mat adjacency_matrix) {
    int best_node = -1;
    int max_edges = 0;
    for (int i = 0; i < adjacency_matrix.size().height; i++) {
        int row_sum = 0;
        for (int j = 0; j < adjacency_matrix.size().width; j++) {
            row_sum += adjacency_matrix.at<uchar>(i,j);
        }
        if(row_sum > max_edges) {
            best_node = i;
            max_edges = row_sum;
        }
    }

    return best_node;
}

vector<int> get_nodes_connected_to_subgraph(Mat adjacency_matrix, vector<int> subgraph) {
    vector<int> nodes_connected;

    for (int i = 0; i < adjacency_matrix.size().height; i++) {
        int number_of_connections = 0;
        for (int j = 0; j < subgraph.size(); j++) {
            if (adjacency_matrix.at<uchar>(i, subgraph[j]) == 1 && i != subgraph[j]) {
                number_of_connections += 1;
            }else{
                break;
            }
        }
        if (number_of_connections == subgraph.size()) {
            nodes_connected.push_back(i);
        }
    }

    return nodes_connected;
}

int get_best_node_from_connected(Mat adjacency_matrix, vector<int> potential_nodes) {
    int max = -1;
    int result = -1;
    for (const auto& node : potential_nodes) {
        int connected_nodes = 0;
        for (int j = 0; j < adjacency_matrix.size().width; j++) {
            if (adjacency_matrix.at<uchar>(node, j) == 1) {
                connected_nodes += 1;
            }
        }
        if (connected_nodes > max) {
            max = connected_nodes;
            result = node;
        }
    }

    return result;
}

bool contains_pair(vector<tuple<int,int>> vector, int i, int j){
    bool is_present = false;
    for (const auto &tup : vector){
        if (get<0>(tup) == i && get<1>(tup) == j){
            is_present = true;
        }
    }

    return is_present;
}

vector<tuple<int,int>> get_feature_pairs_index(vector<int> subgraph_nodes){
    vector<tuple<int,int>> feature_pairs;

    for (int i = 0 ; i < subgraph_nodes.size(); i++){
        for (int j = 0 ; j < subgraph_nodes.size(); j++){
            if (subgraph_nodes[i] != subgraph_nodes[j] && !contains_pair(feature_pairs, subgraph_nodes[i], subgraph_nodes[j]) && !contains_pair(feature_pairs, subgraph_nodes[j], subgraph_nodes[i])){
                feature_pairs.push_back(make_tuple(subgraph_nodes[i],subgraph_nodes[j]));
            }
        }
    }
    return feature_pairs;
}


vector<Mat> get_feature_pairs_2d(vector<int> feature_index, vector<Point2f> points_2d){
    vector<Mat> pairs_2d;
    for (const auto &pair : feature_index){
        // initialize the first point of the 2d pair
        int feature_1_index = pair;
        Mat feature_1_mat = Mat(1, 4, CV_32FC1);
        feature_1_mat.at<float>(0,0) = points_2d[feature_1_index].x;
        feature_1_mat.at<float>(0,1) = points_2d[feature_1_index].y;
        feature_1_mat.at<float>(0,2) = 1;

    }
    return pairs_2d;
}

vector<Mat> get_feature_pairs_3d(vector<int> feature_index, vector<Mat> points_3d){
    vector<Mat> pairs_3d;
    for (const auto &pair : feature_index){
        // initialize the first point of the 2d pair
        int feature_1_index = pair;
        Mat feature_1_mat = points_3d[feature_1_index].clone();
        pairs_3d.push_back(feature_1_mat);
    }
    return pairs_3d;
}

class Optimiz_F:public cv::MinProblemSolver::Function{
    int getDims() const { return 2; }
    double calc(const double* x) const {
        Mat transform = Mat(4, 4, CV_64FC1);
        transform.at<double>(0,0) = x[0];
        transform.at<double>(0,1) = x[1];
        transform.at<double>(0,2) = x[2];
        transform.at<double>(0,3) = x[3];
        
        transform.at<double>(0,0) = x[4];
        transform.at<double>(0,1) = x[5];
        transform.at<double>(0,2) = x[6];
        transform.at<double>(0,3) = x[7];
        
        transform.at<double>(0,0) = x[8];
        transform.at<double>(0,1) = x[9];
        transform.at<double>(0,2) = x[10];
        transform.at<double>(0,3) = x[11];
        
        transform.at<double>(0,0) = 0;
        transform.at<double>(0,1) = 0;
        transform.at<double>(0,2) = 0;
        transform.at<double>(0,3) = 1;

        Mat test = Mat(4, 4, CV_64FC1);
        test.setTo(5.0);

        Mat result = test - transform;
        Scalar sum_squared = sum(result) * sum(result);
        return sum_squared[0];
    }
};



int main() {
    // vector<Mat> images_cam_1 = get_images("C:\\Projects\\visual\\rsc\\sequences\\00\\image_2", 1);
    // vector<Mat> images_cam_2 = get_images("C:\\Projects\\visual\\rsc\\sequences\\00\\image_3", 1);
    // Open the ZED camera
    VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    // Set the video resolution to HD720 (2560*720)
    cap.set(CAP_PROP_FRAME_WIDTH, 1344);
    cap.set(CAP_PROP_FRAME_HEIGHT, 376);
    sbm_left->setPreFilterCap(32);
    sbm_left->setPreFilterSize(9);
    sbm_left->setMinDisparity(0);
    sbm_left->setNumDisparities(16 * 6);
    sbm_left->setTextureThreshold(0);
    sbm_left->setUniquenessRatio(0);
    sbm_left->setSpeckleWindowSize(0);
    sbm_left->setSpeckleRange(0);

    // Initialise the previous frame structures
    tuple<Mat, Mat> undistorted_images = compute_undistorted_images(cap);
    // Get the stereo rectification transformations.
    vector<Mat> stereo_transformations = compute_transformations_calibrated(get<0>(undistorted_images));
    // Compute the rectified images given the stereo transformation
    tuple<Mat, Mat> rectified_images = compute_rectified_images(get<0>(undistorted_images), get<1>(undistorted_images), stereo_transformations[0], stereo_transformations[1]);
    // Get the rectified images from the tuple object.
    Mat previous_rectified_left = get<0>(rectified_images);
    Mat previous_rectified_right = get<1>(rectified_images);


    for (;;) {
        
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        // Get the undistorted images.
        tuple<Mat, Mat> undistorted_images = compute_undistorted_images(cap);
        // Calculate the rectified images.
        tuple<Mat, Mat> rectified_images = compute_rectified_images(get<0>(undistorted_images), get<1>(undistorted_images), stereo_transformations[0], stereo_transformations[1]);
        // Get the rectified images from the tuple object
        Mat current_rectified_left = get<0>(rectified_images);
        Mat current_rectified_right = get<1>(rectified_images);

        // Calculate the disparity map given the current and previous rectified images.
        Mat previous_disparity_result = get_disparity_map(previous_rectified_left, previous_rectified_right);
        Mat current_disparity_result = get_disparity_map(current_rectified_left, current_rectified_right);

        // Detect features in the current frame and match them with the features of the previous frame
        vector<KeyPoint> detected_keypoints_left;
        vector<Point2f>  detected_points_left, matched_points_left, relevant_points_left;
        fast_detector->detect(previous_rectified_left, detected_keypoints_left);
        
        // Track the keypoints.
        /*Note that in my current implementation, I am just tracking the point from one frame to the next,
        and then again doing the detection part, but in a better implmentation, 
        one would track these points as long as the number of points do not drop below a particular threshold.
        */
        for (int i = 0; i < detected_keypoints_left.size(); i++) {
            detected_points_left.push_back(detected_keypoints_left[i].pt);
        }

        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

        vector<uchar> status;
        vector<float> err;
        try {
            calcOpticalFlowPyrLK(previous_rectified_left, current_rectified_left, detected_points_left, matched_points_left, status, err, Size(15, 15), 2, criteria);
        }
        catch (Exception e){

        }
        // Validate the matched keypoints. 
        vector<int> validated_points_index = validate_keypoints(matched_points_left, status, current_disparity_result);
        vector<Point2f> validated_prev_points_left, validated_current_points_left;

        // Add the correct indexes
        for (const auto& index : validated_points_index) {
            validated_prev_points_left.push_back(detected_keypoints_left[index].pt);
            validated_current_points_left.push_back(matched_points_left[index]);
        }

        // Create pointcloud of the calculated features from left image
        vector<Mat> left_previous_keypoints_pc = compute_features_pointcloud(validated_prev_points_left, previous_disparity_result, stereo_transformations[2]);
        vector<Mat> left_matched_keypoints_pc = compute_features_pointcloud(validated_current_points_left, current_disparity_result, stereo_transformations[2]);

        Mat adjacency_matrix = compute_adjacency_matrix(left_previous_keypoints_pc, left_matched_keypoints_pc);
        
        // Find potential subgraph with maximum edges in the given graph
        //1. Select the node with the maximum degree, and initialize the clique to contain this node.
        //2. From the existing clique, determine the subset of nodes v which are connected to all the nodes present in the clique.
        //3. From the set v, select a node which is connected to the maximum number of other nodes in v. Repeat from step 2 till no more nodes can be added to the clique.
        
        //- 1
        vector<int> max_subgraph;
        int best_starting_node = get_maximum_node(adjacency_matrix);
        max_subgraph.push_back(best_starting_node);
        while (true) {
            //- 2
            vector<int> connected_nodes = get_nodes_connected_to_subgraph(adjacency_matrix, max_subgraph);
            if (connected_nodes.size() == 0) {
                break;
            }
            //- 3
            int best_potential_node = get_best_node_from_connected(adjacency_matrix, connected_nodes);
            max_subgraph.push_back(best_potential_node);
        }

        // Compute tuples of 2d points for each relevant feature.
        feature_2d_prev = get_feature_pairs_2d(max_subgraph, validated_prev_points_left);
        feature_2d_current = get_feature_pairs_2d(max_subgraph, validated_current_points_left);

        // Compute tuples of 3d points for each relevant feature.
        feature_3d_prev = get_feature_pairs_3d(max_subgraph, left_previous_keypoints_pc);
        feature_3d_current = get_feature_pairs_3d(max_subgraph, left_matched_keypoints_pc);
        // P matrix project 3d to 2d
        P = stereo_transformations[3];
        
        cv::Ptr<cv::MinProblemSolver::Function> ptr_F = cv::makePtr<Optimiz_F>();
        Mat x = (Mat_<double>(2, 1) << 5.0, 0.0);
        Ptr<DownhillSolver> solver = DownhillSolver::create(ptr_F,x);
       
        double fval = solver->minimize(x);
        std::cout << fval << std::endl;

        // Create
  /*      Mat depth_keypoints;
        drawKeypoints(left_gray, detected_keypoints_left, depth_keypoints);*/
        // Concat the results and draw epipolar lines.
        //Mat concat;
        //cv::hconcat(out_left, out_right, concat);
        //line(concat, Point(Vec2i(0, 100)), Point(Vec2i(concat.size().width, 100)), Scalar(255, 0, 0), 1, LINE_8);
        //line(concat, Point(Vec2i(0, 175)), Point(Vec2i(concat.size().width, 175)), Scalar(255, 0, 0), 1, LINE_8);
        //line(concat, Point(Vec2i(0, 225)), Point(Vec2i(concat.size().width, 225)), Scalar(255, 0, 0), 1, LINE_8);
        // imshow("depth", previous_disparity_result);
        // waitKey(1);

        // Copy the previous frame data
        previous_rectified_left = current_rectified_left.clone();
        previous_rectified_right = current_rectified_right.clone();
    }

    return 0;

}