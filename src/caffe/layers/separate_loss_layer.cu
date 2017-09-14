#include <vector>

#include "caffe/layers/separate_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SeparateLossForwardLoss(const int nthreads, const int N, const int K, const int D, const Dtype delta,
	const Dtype* x_data, Dtype* dis_data, Dtype* loss_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		dis_data[index] = 0;
		int k1 = index % K;
		int k2 = index / K % K;
		int n = index / K / K;
		int x_idx1 = (k1*N + n)*D;
		int x_idx2 = (k2*N + n)*D;
		for (int d = 0; d < D; ++d)
			dis_data[index] += abs(x_data[x_idx1++] - x_data[x_idx2++]);
		if (k1 == k2)
			loss_data[index] = 0;
		else
			loss_data[index] = max(Dtype(0), delta - dis_data[index]);
	}
}

template <typename Dtype>
void SeparateLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* x_data = bottom[0]->gpu_data();
	Dtype* dis_data = dis_.mutable_gpu_data();
	Dtype* loss_data = loss_.mutable_gpu_data();
	SeparateLossForwardLoss<Dtype><<<CAFFE_GET_BLOCKS(N_*K_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_*K_, N_, K_, D_, delta_,
		x_data, dis_data, loss_data);

	Dtype loss = 0;
	caffe_gpu_asum<Dtype>(N_*K_*K_, loss_data, &loss);
	top[0]->mutable_cpu_data()[0] = loss / N_ / (K_*K_ - K_);
}

template <typename Dtype>
__global__ void SeparateLossBackwardDis(const int nthreads, const Dtype scale, const Dtype* loss_data, Dtype* dis_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		dis_diff[index] = (loss_data[index] == 0) ? 0 : -scale;
	}
}

template <typename Dtype>
__global__ void SeparateLossBackwardX(const int nthreads, const int K, const int N, const int D,
	const Dtype* dis_diff, const Dtype* x_data, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		x_diff[index] = 0;
		int dis_idx = ((index / D % N) * K + index / D / N) * K;
		int x_idx_prime = index / D % N * D + index % D;
		for (int k_prime = 0; k_prime < K; ++k_prime) {
			x_diff[index] += dis_diff[dis_idx++] * (2 * (x_data[index] > x_data[x_idx_prime]) - 1);
			x_idx_prime += N*D;
		}
		x_diff[index] *= 2;
	}
}

template <typename Dtype>
void SeparateLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype scale = top[0]->cpu_diff()[0] / N_ / (K_*K_ - K_);
	const Dtype* x_data = bottom[0]->gpu_data();
	const Dtype* loss_data = loss_.gpu_data();
	Dtype* dis_diff = dis_.mutable_gpu_diff();
	SeparateLossBackwardDis<Dtype><<<CAFFE_GET_BLOCKS(N_*K_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_*K_, scale, loss_data, dis_diff);

	Dtype* x_diff = bottom[0]->mutable_gpu_diff();
	SeparateLossBackwardX<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
		dis_diff, x_data, x_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SeparateLossLayer);

}  // namespace caffe
