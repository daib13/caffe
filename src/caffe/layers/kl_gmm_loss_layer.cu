#include <vector>

#include "caffe/layers/kl_gmm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KLGMMForwardSum(const int nthreads, const int D, const Dtype* input, Dtype* output) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		output[index] = 0;
		int idx = index * D;
		for (int d = 0; d < D; ++d)
			output[index] += input[idx++];
	}
}

template <typename Dtype>
__global__ void KLGMMForwardLogPDim(const int nthreads, 
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data, Dtype* logp_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const Dtype safe_sd = max(Dtype(1e-12), sd_z_data[index]);
		logp_dim_data[index] = (-pow((z_data[index] - mu_z_data[index]) / safe_sd, 2) - LOG_TWO_PI) / Dtype(2) 
			- log(safe_sd);
	}
}

template <typename Dtype>
__global__ void KLGMMForwardLogQDim(const int nthreads, const int K, const int D,
	const Dtype* z_data, const Dtype* mu_c_data, const Dtype* sd_c_data, Dtype* logq_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int d = index % D;
		int k = index / D % K;
		int n = index / D / K;
		const Dtype safe_sd = max(Dtype(1e-12), sd_c_data[k*D + d]);
		logq_dim_data[index] = (-pow((z_data[n*D + d] - mu_c_data[k*D + d]) / safe_sd, 2) - LOG_TWO_PI) / Dtype(2) 
			- log(safe_sd);
	}
}

template <typename Dtype>
__global__ void KLGMMForwardLogQMax(const int nthreads, const int K,
	const Dtype* prior_data, Dtype* logq_data, Dtype* logq_max_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		logq_max_data[index] = -INT_MAX;
		int idx = index * K;
		for (int k = 0; k < K; ++k) {
			logq_data[idx] += log(max(Dtype(1e-12), prior_data[k]));
			logq_max_data[index] = max(logq_max_data[index], logq_data[idx++]);
		}
	}
}

template <typename Dtype>
__global__ void KLGMMForwardResQ(const int nthreads, const int K,
	const Dtype* logq_data, const Dtype* logq_max_data, Dtype* resq_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		resq_data[index] = exp(logq_data[index] - logq_max_data[index / K]);
	}
}

template <typename Dtype>
__global__ void KLGMMForwardLoss(const int nthreads,
	const Dtype* logp_data, const Dtype* logq_max_data, const Dtype* resq_sum_data, Dtype* loss_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		loss_data[index] = logp_data[index] - logq_max_data[index] - log(resq_sum_data[index]);
	}
}

template <typename Dtype>
__global__ void KLGMMForwardPosterior(const int nthreads, const int K, const Dtype* resq_data, const Dtype* resq_sum_data,
	Dtype* posterior) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		posterior[index] = resq_data[index] / resq_sum_data[index / K];
	}
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* z_data = bottom[0]->gpu_data();
	const Dtype* mu_z_data = bottom[1]->gpu_data();
	const Dtype* sd_z_data = bottom[2]->gpu_data();
	Dtype* logp_dim_data = logp_dim_.mutable_gpu_data();
	KLGMMForwardLogPDim<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_,
		z_data, mu_z_data, sd_z_data, logp_dim_data);

	Dtype* logp_data = logp_.mutable_gpu_data();
	KLGMMForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, D_, logp_dim_data, logp_data);
	
	const Dtype* mu_c_data = bottom[4]->gpu_data();
	const Dtype* sd_c_data = bottom[5]->gpu_data();
	Dtype* logq_dim_data = logq_dim_.mutable_gpu_data();
	KLGMMForwardLogQDim<Dtype><<<CAFFE_GET_BLOCKS(N_*K_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_*D_, K_, D_,
		z_data, mu_c_data, sd_c_data, logq_dim_data);

	Dtype* logq_data = logq_.mutable_gpu_data();
	KLGMMForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, D_, logq_dim_data, logq_data);

	const Dtype* prior_data = bottom[3]->gpu_data();
	Dtype* logq_max_data = logq_max_.mutable_gpu_data();
	KLGMMForwardLogQMax<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_,
		prior_data, logq_data, logq_max_data);

	Dtype* resq_data = resq_.mutable_gpu_data();
	KLGMMForwardResQ<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
		logq_data, logq_max_data, resq_data);

	Dtype* resq_sum_data = resq_sum_.mutable_gpu_data();
	KLGMMForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_, resq_data, resq_sum_data);

	Dtype* loss_data = item_loss_.mutable_gpu_data();
	KLGMMForwardLoss<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, 
		logp_data, logq_max_data, resq_sum_data, loss_data);

	Dtype loss = 0;
	loss_data = item_loss_.mutable_cpu_data();
	for (int n = 0; n < N_; ++n)
		loss += loss_data[n];
	top[0]->mutable_cpu_data()[0] = loss / N_;

	Dtype* posterior = logq_.mutable_gpu_diff();
	KLGMMForwardPosterior<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
		resq_data, resq_sum_data, posterior);
	if (top.size() == 2)
		caffe_copy<Dtype>(N_*K_, posterior, top[1]->mutable_cpu_data());
}

template <typename Dtype>
__global__ void KLGMMBackwardZ(const int nthreads, const int K, const int D, const Dtype* logp_diff, const Dtype* logq_diff,
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data,
	const Dtype* mu_c_data, const Dtype* sd_c_data, Dtype* z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / D;
		int d = index % D;
		Dtype safe_sd = max(Dtype(1e-12), sd_z_data[index]);
		z_diff[index] = logp_diff[n] * (mu_z_data[index] - z_data[index]) / pow(safe_sd, 2);
		for (int k = 0; k < K; ++k) {
			int q_idx = n*K + k;
			int c_idx = k*D + d;
			safe_sd = max(Dtype(1e-12), sd_c_data[c_idx]);
			z_diff[index] += logq_diff[q_idx] * (mu_c_data[c_idx] - z_data[index]) / pow(safe_sd, 2);
		}
	}
}

template <typename Dtype>
__global__ void KLGMMBackwardMuZ(const int nthreads, const int D, const Dtype* logp_diff,
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data, Dtype* mu_z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype safe_sd = max(Dtype(1e-12), sd_z_data[index]);
		mu_z_diff[index] = logp_diff[index/D] * (z_data[index] - mu_z_data[index]) / pow(safe_sd, 2);
	}
}

template <typename Dtype>
__global__ void KLGMMBackwardSdZ(const int nthreads, const int D, const Dtype* logp_diff,
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data, Dtype* sd_z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		Dtype safe_sd = max(Dtype(1e-12), sd_z_data[index]);
		sd_z_diff[index] = logp_diff[index/D] * (pow((mu_z_data[index] - z_data[index]) / safe_sd, 2) - 1) / safe_sd;
	}
}

template <typename Dtype>
__global__ void KLGMMBackwardPrior(const int nthreads, const int N, const Dtype* logq_diff, const Dtype* prior_data,
	Dtype* prior_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		prior_diff[index] = 0;
		for (int n = 0; n < N; ++n)
			prior_diff[index] += logq_diff[n*nthreads + index];
		prior_diff[index] /= max(Dtype(1e-12), prior_data[index]);
	}
}

template <typename Dtype>
__global__ void KLGMMBackwardMuC(const int nthreads, const int N, const int K, const int D,
	const Dtype* logq_diff, const Dtype* z_data, const Dtype* mu_c_data, const Dtype* sd_c_data,
	Dtype* mu_c_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int k = index / D;
		int d = index % D;
		mu_c_diff[index] = Dtype(0);
		const Dtype safe_sd = max(Dtype(1e-12), sd_c_data[index]);
		const Dtype safe_sd_square = pow(safe_sd, 2);
		for (int n = 0; n < N; ++n) {
			mu_c_diff[index] += logq_diff[n*K + k] * (z_data[n*D + d] - mu_c_data[index]) / safe_sd_square;
		}
	}
}

template <typename Dtype>
__global__ void KLGMMBackwardSdC(const int nthreads, const int N, const int K, const int D,
	const Dtype* logq_diff, const Dtype* z_data, const Dtype* mu_c_data, const Dtype* sd_c_data,
	Dtype* sd_c_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int k = index / D;
		int d = index % D;
		sd_c_diff[index] = Dtype(0);
		const Dtype safe_sd = max(Dtype(1e-12), sd_c_data[index]);
		for (int n = 0; n < N; ++n) {
			sd_c_diff[index] += logq_diff[n*K + k]
				* (pow((z_data[n*D + d] - mu_c_data[index]) / safe_sd, 2) - 1) / safe_sd;
		}
	}
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype scale = top[0]->cpu_diff()[0] / Dtype(N_);
	Dtype* logp_diff = logp_.mutable_gpu_diff();
	caffe_gpu_set<Dtype>(N_, scale, logp_diff);
	Dtype* logq_diff = logq_.mutable_gpu_diff();
	caffe_gpu_scale<Dtype>(N_*K_, -scale, logq_diff, logq_diff);

	const Dtype* z_data = bottom[0]->gpu_data();
	const Dtype* mu_z_data = bottom[1]->gpu_data();
	const Dtype* sd_z_data = bottom[2]->gpu_data();
	const Dtype* prior_data = bottom[3]->gpu_data();
	const Dtype* mu_c_data = bottom[4]->gpu_data();
	const Dtype* sd_c_data = bottom[5]->gpu_data();
	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_gpu_diff();
		KLGMMBackwardZ<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, K_, D_, logp_diff, logq_diff,
			z_data, mu_z_data, sd_z_data, mu_c_data, sd_c_data, z_diff);
	}
	if (propagate_down[1]) {
		Dtype* mu_z_diff = bottom[1]->mutable_gpu_diff();
		KLGMMBackwardMuZ<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, D_, logp_diff,
			z_data, mu_z_data, sd_z_data, mu_z_diff);
	}
	if (propagate_down[2]) {
		Dtype* sd_z_diff = bottom[2]->mutable_gpu_diff();
		KLGMMBackwardSdZ<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, D_, logp_diff,
			z_data, mu_z_data, sd_z_data, sd_z_diff);
	}
	if (propagate_down[3]) {
		Dtype* prior_diff = bottom[3]->mutable_gpu_diff();
		KLGMMBackwardPrior<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>(K_, N_, logq_diff, prior_data, prior_diff);
	}
	if (propagate_down[4]) {
		Dtype* mu_c_diff = bottom[4]->mutable_gpu_diff();
		KLGMMBackwardMuC<Dtype><<<CAFFE_GET_BLOCKS(K_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*D_, N_, K_, D_,
			logq_diff, z_data, mu_c_data, sd_c_data, mu_c_diff);
	}
	if (propagate_down[5]) {
		Dtype* sd_c_diff = bottom[5]->mutable_gpu_diff();
		KLGMMBackwardSdC<Dtype><<<CAFFE_GET_BLOCKS(K_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*D_, N_, K_, D_,
			logq_diff, z_data, mu_c_data, sd_c_data, sd_c_diff);
		int c_idx = 0;
		for (int k = 0; k < K_; ++k)
		for (int d = 0; d < D_; ++d) {
			
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(KLGMMLossLayer);

}  // namespace caffe
