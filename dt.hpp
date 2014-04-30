
#include "opencv2/opencv.hpp"

namespace cv {

void distanceTransform(const cv::Mat &inputMatrix, cv::Mat &outputMatrix,
		cv::Mat &locations, std::vector<float> weights = std::vector<float>());

}


