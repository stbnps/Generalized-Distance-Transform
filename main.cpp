
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "dt.hpp"


using namespace cv;
using namespace std;

int main() {
	Mat t = Mat::ones(Size(4, 4), CV_32FC1) * 10;
	Mat r = Mat::zeros(Size(4, 4), CV_32FC1);
	Mat l;

	// Just to test that weighting is working
	vector<float> weights;
	weights.push_back(2);
	weights.push_back(2);

	t.at<float>(2, 2) = 1;

	cout << "Input:" << endl;
	cout << t << endl << endl;

	distanceTransform(t, r, l, weights);

	cout << "Result:" << endl;
	cout << r << endl;

	int *locations = (int *) l.data;

	cout << endl;

	/*
	 * Print locations of the minimum values
	 */
	cout << "Locations X:" << endl;
	for (size_t row = 0; row < 4; ++row) {
		for (size_t col = 0; col < 4; ++col) {
			cout << locations[col + 4 * row] << ", ";
		}
		cout << endl;
	}

	cout << endl;

	cout << "Locations Y:" << endl;
	for (size_t row = 0; row < 4; ++row) {
		for (size_t col = 0; col < 4; ++col) {
			cout << locations[16 + col + 4 * row] << ", ";
		}
		cout << endl;
	}

	return 0;

}
