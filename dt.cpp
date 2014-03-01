#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/opencv_modules.hpp"

using namespace cv;
using namespace std;

template<class T> T square(const T &x) {
	return x * x;
}

/*
 * Calculates the distance transform along one dimension for the whole input matrix.
 */
void distanceTransform1d(const Mat &inputMatrix, Mat &outputMatrix,
		Mat &locations, int dataStart, int dim) {

	const float inf = 1e15f;
	int n = inputMatrix.size[dim];
	int step = inputMatrix.step[dim] / 4; // inputMatrix.step is in bytes; 4 bytes each float
	float *f = (float *) inputMatrix.data;
	f += dataStart;
	float *d = (float *) outputMatrix.data;
	d += dataStart;
	int *l = (int *) locations.data;
	l += dataStart;
	int *v = new int[n];
	float *z = new float[n + 1];
	int k = 0;
	v[0] = 0;
	z[0] = -inf;
	z[1] = +inf;
	for (int q = 1; q <= n - 1; q++) {
		float s = ((f[q * step] + square(q)) - (f[v[k] * step] + square(v[k])))
				/ (2 * q - 2 * v[k]);
		while (s <= z[k]) {
			k--;
			s = ((f[q * step] + square(q)) - (f[v[k] * step] + square(v[k])))
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
		d[q * step] = square(q - v[k]) + f[v[k] * step];
		l[q * step + dim * locations.step[0] / 4] = v[k]; // Save minimum location
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


class DistanceTransformInvoker: public ParallelLoopBody {
public:
	DistanceTransformInvoker(Mat& inputMatrix, Mat *outputMatrix,
			Mat *locations, int *steps, int dim) {
		*outputMatrix = inputMatrix;
		this->outputMatrix = outputMatrix;
		this->locationsMatrix = locations;
		this->steps = steps;
		this->dim = dim;

	}

	void operator()(const Range& range) const {
		int i, i1 = range.start, i2 = range.end;

		for (i = i1; i < i2; i++) { // Process current range of scales
			int dataStart = 0;
			for (int d = 0; d < outputMatrix->dims; ++d) {
				// No need to jump when d == dim since currentStep[dim] will be 0
				dataStart += steps[i * outputMatrix->dims + d]
						* outputMatrix->step[d] / 4;
			}

			// Now we have calculated where the data starts, perform the distance transform
			distanceTransform1d(*outputMatrix, *outputMatrix, *locationsMatrix,
					dataStart, dim);

		}

	}

private:
	Mat *outputMatrix;
	Mat *locationsMatrix;
	int *steps;
	int dim;

};

/*
 * Calculates the distance transform.
 */
void distanceTransform(const Mat &inputMatrix, Mat &outputMatrix,
		Mat &locations) {

	// Input matrix has proper type
	CV_Assert(inputMatrix.type() == CV_32FC1);

	// This way we don't mess with users input, they may want to use it later
	outputMatrix = inputMatrix.clone();

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
		int currentStep[iterations][outputMatrix.dims]; // Stores in which column, row, z step, etc we are

		memset(currentStep, 0, iterations * outputMatrix.dims * sizeof(int));

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

		// Perform 1d distance transform along the current dimension on the whole matrix
		Range range(0, iterations);
		DistanceTransformInvoker invoker(outputMatrix, &outputMatrix,
				&locations, &currentStep[0][0], dim);
		parallel_for_(range, invoker);

	}
}
