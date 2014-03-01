
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "dt.h"


using namespace cv;
using namespace std;

int main() {
	Mat t = Mat::ones(Size(4, 4), CV_32FC1) * 10;
	Mat r = Mat::zeros(Size(4, 4), CV_32FC1);
	int sizes[] = { 2, 4, 4 };
	Mat l(3, sizes, CV_32SC1);

//	t.at<float>(0, 0) = 10;
//	t.at<float>(1, 0) = 10;
//	t.at<float>(2, 0) = 10;
//	t.at<float>(0, 1) = 10;
//	t.at<float>(0, 2) = 10;
	t.at<float>(2, 2) = 1;

	cout << t << endl << endl;

	distanceTransform(t, r, l);

	cout << r << endl;

	int *locations = (int *) l.data;

	cout << endl;

	/*
	 * Print locations of the minimum values
	 */
	for (size_t row = 0; row < 4; ++row) {
		for (size_t col = 0; col < 4; ++col) {
			cout << locations[col + 4 * row] << ", ";
		}
		cout << endl;
	}

	cout << endl;

	for (size_t row = 0; row < 4; ++row) {
		for (size_t col = 0; col < 4; ++col) {
			cout << locations[16 + col + 4 * row] << ", ";
		}
		cout << endl;
	}

	return 0;

}
