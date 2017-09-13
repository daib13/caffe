#include <vector>

#include "caffe/layers/log_probability_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LogProbabilityForwardDim(const int nthreads, const Dtype* x_data, const Dtype* x_hat_data,
	Dtype* p_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		p_dim_data[index] = x_data[index] * log(max(Dtype(1e-12), x_hat_data[index]))
			+ (1 - x_data[index]) * log(max(Dtype(1e-12), Dtype(1 - x_hat_data[index])));
	}
}

template <typename Dtype>
__global__ void LogProbabilityForwardP(const int nthreads, const int K, const int N, const int D,
	const Dtype* p_dim_data, Dtype* p_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int idx = ((index%K)*N + index / K)*D;
		p_data[index] = 0;
		for (int d = 0; d < D; ++d)
			p_data[index] += p_dim_data[idx++]; 
	}
}

template <typename Dtype>
void LogProbabilityLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	concat_layer_->Forward(concat_layer_bottom_, concat_layer_top_);
	const Dtype* x_hat_data = bottom[0]->gpu_data();
	const Dtype* x_data = x_duplicate_.gpu_data();
	Dtype* p_dim_data = p_dim_.mutable_gpu_data();
	LogProbabilityForwardDim<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_,
		x_data, x_hat_data, p_dim_data);

	Dtype* p_data = top[0]->mutable_gpu_data();
	LogProbabilityForwardP<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_, N_, D_, 
		p_dim_data, p_data);
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXhat(const int nthreads, const int K, const int N, const int D,
	const Dtype* x_data, const Dtype* x_hat_data,
	const Dtype* p_diff, Dtype* x_hat_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_hat_diff[index] = (x_data[index] - x_hat_data[index]) /
			max(Dtype(1e-12), x_hat_data[index] * (1 - x_hat_data[index])) * p_diff[p_idx];
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardX(const int nthreads, const int K, const int N, const int D,
	const Dtype* x_data, const Dtype* x_hat_data,
	const Dtype* p_diff, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_diff[index] = p_diff[p_idx] * (log(max(Dtype(1e-12), x_hat_data[index]))
			- log(max(Dtype(1e-12), Dtype(1) - x_hat_data[index])));
	}
}

template <typename Dtype>
void LogProbabilityLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* p_diff = top[0]->gpu_diff();
	const Dtype* x_hat_data = bottom[0]->gpu_data();
	const Dtype* x_data = x_duplicate_.gpu_data();
	if (propagate_down[0]) {
		Dtype* x_hat_diff = bottom[0]->mutable_gpu_diff();
		LogProbabilityBackwardXhat<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
			x_data, x_hat_data, p_diff, x_hat_diff);
	}
	if (propagate_down[1]) {
		Dtype* x_diff = x_duplicate_.mutable_gpu_diff();
		LogProbabilityBackwardX<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
			x_data, x_hat_data, p_diff, x_diff);
		Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
		caffe_gpu_set<Dtype>(bottom[1]->count(), Dtype(0), bottom_diff);
		for (int k = 0; k < K_; ++k)
			caffe_gpu_axpby<Dtype>(N_*D_, Dtype(1), x_diff + k*N_*D_, Dtype(1), bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LogProbabilityLayer);

}  // namespace caffe
