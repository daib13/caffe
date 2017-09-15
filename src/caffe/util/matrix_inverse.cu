#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void gpu_matrix_inverse_init_output(const int nthreads, const int D, 
	Dtype* output) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		output[index * D + index % D] = Dtype(1);
	}
}

template <typename Dtype>
__global__ void gpu_check_head_row(const int nthreads, const int D, const int row, const Dtype* input, 
	int* nonzero_row_id) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		nonzero_row_id[index] = row;
		int idx = (index*D + row)*D + row;
		while (abs(input[idx]) < 1e-12) {
			++nonzero_row_id[index];
			idx += D;
			if (nonzero_row_id[index] == D) {
				nonzero_row_id[index] = row;
				break;
			}
		}
	}
}

template <typename Dtype>
__global__ void gpu_switch_row(const int nthreads, const int D, const int row, const int* nonzero_row_id,
	Dtype* input, Dtype* output, Dtype* scale) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / D;
		int d = index % D;
		if (row != nonzero_row_id[n] && d >= row) {
			int idx1 = (n*D + row)*D + d;
			int idx2 = (n*D + nonzero_row_id[n])*D + d;
			Dtype tmp = input[idx1];
			input[idx1] = input[idx2];
			input[idx2] = tmp;
			tmp = output[idx1];
			output[idx1] = output[idx2];
			output[idx2] = tmp;
		}
		if (d == row)
			scale[n] = max(Dtype(1e-12), input[(n*D + row)*D + d]);
	}
}

template <typename Dtype>
__global__ void gpu_divide_row(const int nthreads, const int D, const int row, const Dtype* scale,
	Dtype* input, Dtype* output, Dtype* log_det, Dtype* row_scale) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / D;
		int d = index % D;
		int idx = (n*D + row)*D + d;
		input[idx] /= scale[n];
		output[idx] /= scale[n];
		if (d == 0)
			log_det[n] += log(scale[n]);
		row_scale[index] = input[(n*D + d)*D + row];
	}
}

template <typename Dtype>
__global__ void gpu_minus_row(const int nthreads, const int D, const int row, const Dtype* row_scale,
	Dtype* input, Dtype* output) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int row_idx = index / D;
		int n = row_idx / D;
		int r = row_idx % D;
		int c = index % D;
		if (r != row) {
			int idx1 = (n*D + row)*D + c;
			int idx2 = (n*D + r)*D + c;
			input[idx2] -= input[idx1] * row_scale[row_idx];
			output[idx2] -= output[idx1] * row_scale[row_idx];
		}
	}
}

template <typename Dtype>
void caffe_gpu_matrix_inverse(const int N, const int D,
	const Dtype* input, Dtype* output, Dtype* log_det) {
	
	Dtype* mutable_input;
	cudaMalloc(&mutable_input, sizeof(Dtype)*N*D*D);
	cudaMemcpy(mutable_input, input, sizeof(Dtype)*N*D*D, cudaMemcpyDeviceToDevice);
	int* nonzero_row_id;
	cudaMalloc(&nonzero_row_id, sizeof(int)*N);
	Dtype* scale;
	cudaMalloc(&scale, sizeof(Dtype)*N);
	Dtype* row_scale;
	cudaMalloc(&row_scale, sizeof(Dtype)*N*D);

	caffe_gpu_set<Dtype>(N*D*D, Dtype(0), output);
	caffe_gpu_set<Dtype>(N, Dtype(0), log_det);
	gpu_matrix_inverse_init_output<Dtype><<<CAFFE_GET_BLOCKS(N*D), CAFFE_CUDA_NUM_THREADS>>>(N*D, D, output);
	for (int row = 0; row < D; ++row) {
		gpu_check_head_row<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, D, row, 
			mutable_input, nonzero_row_id);
		gpu_switch_row<Dtype><<<CAFFE_GET_BLOCKS(N*D), CAFFE_CUDA_NUM_THREADS>>>(N*D, D, row, nonzero_row_id, 
			mutable_input, output, scale);
		gpu_divide_row<Dtype><<<CAFFE_GET_BLOCKS(N*D), CAFFE_CUDA_NUM_THREADS>>>(N*D, D, row, scale,
			mutable_input, output, log_det, row_scale);
		gpu_minus_row<Dtype><<<CAFFE_GET_BLOCKS(N*D*D), CAFFE_CUDA_NUM_THREADS>>>(N*D*D, D, row, row_scale,
			mutable_input, output);
	}

	cudaFree(mutable_input);
	cudaFree(nonzero_row_id);
	cudaFree(scale);
	cudaFree(row_scale);
}

template void caffe_gpu_matrix_inverse<float>(const int N, const int D, const float* input, float* output, float* log_det);
template void caffe_gpu_matrix_inverse<double>(const int N, const int D, const double* input, double* output, double* log_det);

}