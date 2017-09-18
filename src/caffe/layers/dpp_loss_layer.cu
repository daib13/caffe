#include <vector>

#include "caffe/layers/dpp_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DppForwardXnorm(const int nthreads, const int N, const int K, const int D, const Dtype* x_data, Dtype* x_norm_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		x_norm_data[index] = 0;
		int x_idx = (index % K * N + index / K) * D;
		for (int d = 0; d < D; ++d)
			x_norm_data[index] += pow(x_data[x_idx++], 2);
		x_norm_data[index] = sqrt(x_norm_data[index]);
	}
}

template <typename Dtype>
__global__ void DppForwardCorrelate(const int nthreads, const int N, const int K, const int D,
	const Dtype* x_data, const Dtype* x_norm_data, Dtype* correlate_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		correlate_data[index] = 0;
		int k2 = index % K;
		int k1 = index / K % K;
		int n = index / K / K;
		int x_idx1 = (k1*N + n)*D;
		int x_idx2 = (k2*N + n)*D;
		for (int d = 0; d < D; ++d)
			correlate_data[index] += x_data[x_idx1++] * x_data[x_idx2++];
		correlate_data[index] /= x_norm_data[n*K + k1];
		correlate_data[index] /= x_norm_data[n*K + k2];
	}
}

template <typename Dtype>
void DppLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* x_data = bottom[0]->gpu_data();
	Dtype* x_norm_data = x_norm_.mutable_gpu_data();
	DppForwardXnorm<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, N_, K_, D_, x_data, x_norm_data);

	Dtype* correlate_data = correlate_matrix_.mutable_gpu_data();
	DppForwardCorrelate<Dtype><<<CAFFE_GET_BLOCKS(N_*K_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_*K_, N_, K_, D_,
		x_data, x_norm_data, correlate_data);
	
	Dtype* correlate_diff = correlate_matrix_.mutable_gpu_diff();
	Dtype* log_det_data = log_det_.mutable_gpu_data();
	caffe_gpu_matrix_inverse<Dtype>(N_, K_, correlate_data, correlate_diff, log_det_data);

	Dtype loss = 0;
	log_det_data = log_det_.mutable_cpu_data();
	for (int n = 0; n < N_; ++n)
		loss -= log_det_data[n];
	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
__global__ void DppBackward(const int nthreads, const int N, const int K, const int D,
	const Dtype* correlate_diff, const Dtype* correlate_data, const Dtype* x_data, const Dtype* x_norm_data, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int d = index % D;
		int n = index / D % N;
		int k = index / D / N;
		int x_norm_idx = n * K + k;
		x_diff[index] = 0;
		Dtype xd_norm = x_data[index] / pow(x_norm_data[x_norm_idx], 2);
		for (int k_prime = 0; k_prime < K; ++k_prime) {
			int c_idx = (n*K + k)*K + k_prime;
			int x_norm_prime_idx = n * K + k_prime;
			int x_prime_idx = (k_prime * N + n) * D + d;
			x_diff[index] += (x_data[x_prime_idx] / x_norm_data[x_norm_idx] / x_norm_data[x_norm_prime_idx]
				- xd_norm * correlate_data[c_idx]) * correlate_diff[c_idx];
		}
		x_diff[index] *= 2;
	}
}

template <typename Dtype>
void DppLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;
	const Dtype* correlate_diff = correlate_matrix_.gpu_diff();
	const Dtype* correlate_data = correlate_matrix_.gpu_data();
	const Dtype* x_data = bottom[0]->gpu_data();
	const Dtype* x_norm_data = x_norm_.gpu_data();
	Dtype* x_diff = bottom[0]->mutable_gpu_diff();
	DppBackward<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, N_, K_, D_,
		correlate_diff, correlate_data, x_data, x_norm_data, x_diff);
	caffe_gpu_scale<Dtype>(bottom[0]->count(), -top[0]->cpu_diff()[0] / Dtype(N_), x_diff, x_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DppLossLayer);

}  // namespace caffe
