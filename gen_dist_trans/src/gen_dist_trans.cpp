//	Copyright (c) 2014, Esteban Pardo Sánchez
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification,
//	are permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this
//	list of conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice,
//	this list of conditions and the following disclaimer in the documentation and/or
//	other materials provided with the distribution.
//
//	3. Neither the name of the copyright holder nor the names of its contributors
//	may be used to endorse or promote products derived from this software without
//	specific prior written permission.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//	ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//	WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//	DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
//	ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//	(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//	LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
//	ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/gen_dist_trans.hpp"
#include <algorithm>

using namespace std;
using namespace cv;

namespace cv
{

template<class T> T square(const T &x) {
	return x * x;
}

namespace { // avoid exporting symbols

/*
 * Calculates the distance transform on a one dimensional array.
 * f is the input signal
 * d is the output signal
 * l is the array containing, for each position, the location of the parabola
 * which affects that position
 * n is the size of the array
 */
void distanceTransform1d(float *f, float *d, int *l, int n) {

	const float inf = 1e20f;
	int *v = new int[n];
	float *z = new float[n + 1];
	int k = 0;
	v[0] = 0;
	z[0] = -inf;
	z[1] = +inf;
	
	for (int q = 1; q <= n - 1; q++) {
		float s = ((f[q] + square(q)) - (f[v[k]] + square(v[k])))
				/ (2 * q - 2 * v[k]);
		while (s <= z[k]) {
			k--;
			s = ((f[q] + square(q)) - (f[v[k]] + square(v[k])))
					/ (2 * q - 2 * v[k]);
		}
		k++;
		v[k] = q;
		z[k] = s;
		z[k + 1] = +inf;
	}

	k = 0;
	for (int q = 0; q <= n - 1; q++) {
		while (z[k + 1] < q)
			k++;
		d[q] = square(q - v[k]) + f[v[k]];
		l[q] = v[k];
	}

	delete[] v;
	delete[] z;
}

/*
 * This would be the functions that perform a generalized distance transform on a 2, and 3
 * dimensional matrix. There's a pattern that enables us to perform it on matrices of
 * arbitrary dimension:
 * For each dimension, for each 1d slice on that dimension  perform a
 * 1d generalized distance transform.
 */
//void distanceTransform2d(const Mat &inputMatrix, Mat &outputMatrix,
//		Mat &locations) {
//	// Distance transform along rows
//	for (int row = 0; row < inputMatrix.size[0]; ++row) {
//		int dataStart = row * inputMatrix.step[0] / 4;
//		distanceTransform1d(inputMatrix, outputMatrix, locations, dataStart, 1);
//	}
//
//	// Now do it along columns, taking as input, the previous result
//	for (int col = 0; col < inputMatrix.size[1]; ++col) {
//		int dataStart = col * inputMatrix.step[1] / 4;
//		distanceTransform1d(outputMatrix, outputMatrix, locations, dataStart,
//				0);
//	}
//
//}
//
//void distanceTransform3d(const Mat &inputMatrix, Mat &outputMatrix,
//		Mat &locations) {
//	// Distance transform along X axis
//	for (int z = 0; z < inputMatrix.size[0]; ++z) {
//		for (int row = 0; row < inputMatrix.size[1]; ++row) {
//			int dataStart = row * inputMatrix.step[1] / 4;
//			dataStart += z * inputMatrix.step[0] / 4;
//			distanceTransform1d(inputMatrix, outputMatrix, locations, dataStart,
//					2);
//		}
//	}
//
//	// Y axis
//	for (int z = 0; z < inputMatrix.size[0]; ++z) {
//		for (int col = 0; col < inputMatrix.size[2]; ++col) {
//			int dataStart = col * inputMatrix.step[2] / 4;
//			dataStart += z * inputMatrix.step[0] / 4;
//			distanceTransform1d(outputMatrix, outputMatrix, locations,
//					dataStart, 1);
//		}
//	}
//
//	// Z axis
//	for (int row = 0; row < inputMatrix.size[1]; ++row) {
//		for (int col = 0; col < inputMatrix.size[2]; ++col) {
//			int dataStart = col * inputMatrix.step[2] / 4;
//			dataStart += row * inputMatrix.step[1] / 4;
//			distanceTransform1d(outputMatrix, outputMatrix, locations,
//					dataStart, 0);
//		}
//	}
//
//}

// Parallel invoker
class DistanceTransformInvoker: public ParallelLoopBody {
public:
    DistanceTransformInvoker(Mat& inputMatrix_, Mat *outputMatrix_,
            Mat *locations_, int **steps_, int dim_) {
        *outputMatrix_ = inputMatrix_.clone();
        this->outputMatrix = outputMatrix_;
        this->locationsMatrix = locations_;
        this->steps = steps_;
        this->dim = dim_;

	}

	void operator()(const Range& range) const {
		int i, i1 = range.start, i2 = range.end;

		for (i = i1; i < i2; i++) { // Process current range of scales
			int dataStart = 0;
			for (int d = 0; d < outputMatrix->dims; ++d) {
				// No need to jump when d == dim since steps[i * outputMatrix->dims + dim] will be 0
				dataStart += steps[i][d] * outputMatrix->step[d] / 4;
			}

			// Now we have calculated where the data starts, perform the distance transform
			float *f = new float[outputMatrix->size[dim]];
			float *d = new float[outputMatrix->size[dim]];
			int *l = new int[outputMatrix->size[dim]];
			float *castedOutputMatrix = (float *) outputMatrix->data;
			int *castedLocationsMatrix = (int *) locationsMatrix->data;
			/*
			 * Strided copy.
			 * Creates the 1d array where the distance transform will be performed.
			 * This array will hold the section of the global matrix were the
			 * distance transform would be performed.
			 */
            for (int j = 0; j < outputMatrix->size[dim]; ++j) {
                f[j] = castedOutputMatrix[dataStart
                        + j * outputMatrix->step[dim] / 4];
			}
			distanceTransform1d(f, d, l, outputMatrix->size[dim]);

			// Strided write
            for (int j = 0; j < outputMatrix->size[dim]; ++j) {
                castedOutputMatrix[dataStart + j * outputMatrix->step[dim] / 4] =
                        d[j];
				castedLocationsMatrix[dataStart
                        + j * outputMatrix->step[dim] / 4
                        + dim * locationsMatrix->step[0] / 4] = l[j];
			}

			delete[] f;
			delete[] d;
			delete[] l;

		}

	}

private:
	Mat *outputMatrix;
	Mat *locationsMatrix;
	int **steps;
	int dim;

};

} // local namespace


/*
 * Calculates the distance transform.
 */
void distanceTransform(InputArray _sampled, OutputArray _dist,
    OutputArray _locations, InputArray _weights) {

    const cv::Mat inputMatrix = _sampled.getMat();
    _sampled.copyTo(_dist);
    cv::Mat outputMatrix = _dist.getMat();
    const cv::Mat weights = _weights.getMat();

	// Input matrix has proper type
	CV_Assert(inputMatrix.type() == CV_32FC1);
    // Dimension scaling is unspecified or has proper size
    CV_Assert(weights.empty()
              || (weights.total() == (size_t)inputMatrix.dims));

    // Create location matrix, for each input pixel the location matrix will
    // have "inputMatrix.dims" parameters.
	vector<int> sizes;
	sizes.push_back(inputMatrix.dims);
	for (int d = 0; d < inputMatrix.dims; ++d) {
		sizes.push_back(inputMatrix.size[d]);
	}
	_locations.create(sizes.size(), &sizes[0], CV_32SC1);
	Mat locations = _locations.getMat();

	for (int dim = outputMatrix.dims - 1; dim >= 0; --dim) {

		// Calculate how many iterations there are for the current dimension
		int iterations = 1;
		for (int d = 0; d < outputMatrix.dims; ++d) {
			if (d == dim) {
				continue;
			}
			iterations *= outputMatrix.size[d];
		}

		// Calculate steps for each iteration, so that iterations can be parallelized
		int **currentStep = new int*[iterations]();
		for (int i = 0; i < iterations; ++i) {
			currentStep[i] = new int[outputMatrix.dims]();
		}

		for (int it = 1; it < iterations; ++it) {
			// Add 1 to the array to know which steps to take now
			// Note that the step of the dimension we are calculating will remain 0
			memcpy(&currentStep[it][0], &currentStep[it - 1][0],
					outputMatrix.dims * sizeof(int));
			if (dim != outputMatrix.dims - 1) {
				currentStep[it][outputMatrix.dims - 1]++;
			} else {
				currentStep[it][outputMatrix.dims - 2]++;
			}

			bool carry = false;
			for (int d = outputMatrix.dims - 1; d >= 0; --d) {
				if (d == dim) {
					continue;
				}
				if (carry) {
					currentStep[it][d]++;
					carry = false;
				}
				// Modulo operation; this way we know if the next dimension is increased
				if (currentStep[it][d] >= outputMatrix.size[d]) {
					currentStep[it][d] = currentStep[it][d]
							- outputMatrix.size[d];
					carry = true;
				}
			}
			// End of addition block
		}

		/*
		 * Doing outputMatrix *= weights[dim]; and then outputMatrix /= weights[dim];
		 * may seem odd, but that way we calculate a mahalanobis distance transform
		 * when the covariance matrix is diagonal.
		 */


		// When the weight is too small use 0.00001 since 0 would screw the results
		double zero = 0.00001;

		if (!weights.empty()) {
			if (weights.at<double>(dim) >= 0.1) {
				outputMatrix *= weights.at<double>(dim);
			} else {
				outputMatrix *= zero;
			}
		}


		// Perform 1d distance transform along the current dimension on the whole matrix
		Range range(0, iterations);
		DistanceTransformInvoker invoker(outputMatrix, &outputMatrix,
				&locations, currentStep, dim);

		cv::parallel_for_(range, invoker);

		if (!weights.empty()) {
			if (weights.at<double>(dim) >= 0.1) {
				outputMatrix /= weights.at<double>(dim);
			} else {
				outputMatrix /= zero;
			}
		}

		for (int i = 0; i < iterations; ++i) {
            delete[](currentStep[i]);
		}
		delete[](currentStep);


	}
}

} // namepsace cv
