#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void cpu_matrix_inverse_init_output(const int N, const int D, 
	Dtype* output) {
	for (int n = 0; n < N; ++n) {
		for (int d = 0; d < D; ++d)
			output[(n*D + d)*D + d] = Dtype(1);
	}
}

template <typename Dtype>
void cpu_check_head_row(const int N, const int D, const int row, const Dtype* input, 
	int* nonzero_row_id) {
	for (int n = 0; n < N; ++n) {
		nonzero_row_id[n] = row;
		int idx = (n*D + row)*D + row;
		while (abs(input[idx]) < 1e-12) {
			++nonzero_row_id[n];
			idx += D;
			if (nonzero_row_id[n] == D) {
				nonzero_row_id[n] = row;
				break;
			}
		}
	}
}

template <typename Dtype>
void cpu_switch_row(const int N, const int D, const int row, const int* nonzero_row_id,
	Dtype* input, Dtype* output) {
	for (int n = 0; n < N; ++n) {
		if (row != nonzero_row_id[n]) {
			int idx1 = (n*D + row)*D;
			int idx2 = (n*D + nonzero_row_id[n])*D;
			for (int d = 0; d < D; ++d) {
				Dtype tmp = input[idx1];
				input[idx1] = input[idx2];
				input[idx2] = tmp;
				tmp = output[idx1];
				output[idx1++] = output[idx2];
				output[idx2++] = tmp;
			}
		}
	}
}

template <typename Dtype>
void cpu_divide_row(const int N, const int D, const int row,
	Dtype* input, Dtype* output, Dtype* log_det) {
	for (int n = 0; n < N; ++n) {
		int idx = (n*D + row)*D;
		Dtype scale = max(Dtype(1e-12), input[idx + row]);
		log_det[n] += log(scale);
		for (int d = 0; d < D; ++d) {
			input[idx] /= scale;
			output[idx++] /= scale;
		}
	}
}

template <typename Dtype>
void cpu_minus_row(const int N, const int D, const int row,
	Dtype* input, Dtype* output) {
	for (int n = 0; n < N; ++n) {
		for (int r = 0; r < D; ++r) {
			if (r == row)
				continue;
			int idx1 = (n*D + row)*D;
			int idx2 = (n*D + r)*D;
			Dtype scale = input[idx2 + row];
			for (int c = 0; c < D; ++c) {
				input[idx2] -= input[idx1] * scale;
				output[idx2++] -= output[idx1++] * scale;
			}
		}
	}
}

template <typename Dtype>
void caffe_cpu_matrix_inverse(const int N, const int D,
	const Dtype* input, Dtype* output, Dtype* log_det) {
	
	Dtype* mutable_input = new Dtype[N*D*D];
	memcpy(mutable_input, input, sizeof(Dtype)*N*D*D);
	int* nonzero_row_id = new int[N];

	caffe_set<Dtype>(N*D*D, Dtype(0), output);
	caffe_set<Dtype>(N, Dtype(0), log_det);
	cpu_matrix_inverse_init_output<Dtype>(N, D, output);
	for (int row = 0; row < D; ++row) {
		cpu_check_head_row<Dtype>(N, D, row, mutable_input, nonzero_row_id);
		cpu_switch_row<Dtype>(N, D, row, nonzero_row_id, mutable_input, output);
		cpu_divide_row<Dtype>(N, D, row, mutable_input, output, log_det);
		cpu_minus_row<Dtype>(N, D, row, mutable_input, output);
	}

	delete[] mutable_input;
	delete[] nonzero_row_id;
}

template void caffe_cpu_matrix_inverse<float>(const int N, const int D, const float* input, float* output, float* log_det);
template void caffe_cpu_matrix_inverse<double>(const int N, const int D, const double* input, double* output, double* log_det);

}