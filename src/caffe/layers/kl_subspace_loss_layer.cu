#include <vector>

#include "caffe/layers/kl_subspace_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KLSubspaceForwardLogPDim(const int nthreads, const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data,
	Dtype* logp_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const Dtype safe_sd = max(Dtype(1e-6), sd_z_data[index]);
		logp_dim_data[index] = -pow((z_data[index] - mu_z_data[index]) / safe_sd, 2) / Dtype(2)
			- LOG_TWO_PI / Dtype(2) - log(safe_sd);
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardSum(const int nthreads, const int D, const Dtype* dim_data, Dtype* sum_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		sum_data[index] = 0;
		int idx = index * D;
		for (int d = 0; d < D; ++d)
			sum_data[index] += dim_data[idx++];
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardLogQDim(const int nthreads, const int K, const int D,
	const Dtype* z_data, const Dtype* sd_gt, Dtype* logq_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int d = index % D;
		int k = index / D % K;
		int n = index / D / K;
		int sd_idx = k*D + d;
		logq_dim_data[index] = -pow(z_data[n*D + d] / sd_gt[sd_idx], 2) / Dtype(2)
			- LOG_TWO_PI / Dtype(2) - log(sd_gt[sd_idx]);
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardMax(const int nthreads, const int K,
	Dtype* logq_data, const Dtype* prior_data, Dtype* max_logq_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		max_logq_data[index] = -INT_MAX;
		int idx = index * K;
		for (int k = 0; k < K; ++k) {
			logq_data[idx] += log(max(Dtype(1e-6), prior_data[k]));
			max_logq_data[index] = max(logq_data[idx++], max_logq_data[index]);
		}
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardResQ(const int nthreads, const int K,
	const Dtype* logq_data, const Dtype* max_logq_data, Dtype* res_q_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		res_q_data[index] = exp(logq_data[index] - max_logq_data[index / K]);
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardLoss(const int nthreads, const Dtype* logp_data, const Dtype* max_logq_data,
	const Dtype* sum_res_q_data, Dtype* item_loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		item_loss[index] = logp_data[index] - max_logq_data[index] - log(sum_res_q_data[index]);
	}
}

template <typename Dtype>
__global__ void KLSubspaceForwardPosterior(const int nthreads, const int K,
	const Dtype* res_q_data, const Dtype* sum_res_q_data, Dtype* posterior) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		posterior[index] = res_q_data[index] / sum_res_q_data[index / K];
	}
}

template <typename Dtype>
void KLSubspaceLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const Dtype* z_data = bottom[0]->gpu_data();
	const Dtype* mu_z_data = bottom[1]->gpu_data();
	const Dtype* sd_z_data = bottom[2]->gpu_data();
	Dtype* logp_dim_data = logp_dim_.mutable_gpu_data();
	KLSubspaceForwardLogPDim<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, z_data, mu_z_data, sd_z_data,
		logp_dim_data);
	Dtype* logp_data = logp_.mutable_gpu_data();
	KLSubspaceForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, D_, logp_dim_data, logp_data);

	const Dtype noise_sd = bottom[4]->cpu_data()[0];
	Dtype* sd_gt = gt_sd_.mutable_cpu_data();
	caffe_set<Dtype>(K_*D_, noise_sd, sd_gt);
	const int dim_per_cluster = D_ / K_;
	for (int d = 0; d < D_; ++d) {
		sd_gt[d / dim_per_cluster*D_ + d]++;
	}
	sd_gt = gt_sd_.mutable_gpu_data();

	Dtype* logq_dim_data = logq_dim_.mutable_gpu_data();
	KLSubspaceForwardLogQDim<Dtype><<<CAFFE_GET_BLOCKS(N_*K_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_*D_, K_, D_,
		z_data, sd_gt, logq_dim_data);
	Dtype* logq_data = logq_.mutable_gpu_data();
	KLSubspaceForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, D_, logq_dim_data, logq_data);

	const Dtype* prior_data = bottom[3]->gpu_data();
	Dtype* max_logq_data = max_logq_.mutable_gpu_data();
	KLSubspaceForwardMax<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_, logq_data, prior_data, max_logq_data);

	Dtype* res_q_data = res_q_.mutable_gpu_data();
	KLSubspaceForwardResQ<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_, 
		logq_data, max_logq_data, res_q_data);

	Dtype* sum_res_q_data = sum_res_q_.mutable_gpu_data();
	KLSubspaceForwardSum<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_, res_q_data, sum_res_q_data);

	Dtype* item_loss = item_loss_.mutable_gpu_data();
	KLSubspaceForwardLoss<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, logp_data, max_logq_data, sum_res_q_data, item_loss);
	item_loss = item_loss_.mutable_cpu_data();
	Dtype loss = 0;
	for (int n = 0; n < N_; ++n)
		loss += item_loss[n];
	top[0]->mutable_cpu_data()[0] = loss / N_;

	if (top.size() == 2) {
		Dtype* posterior = top[1]->mutable_gpu_data();
		KLSubspaceForwardPosterior<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
			res_q_data, sum_res_q_data, posterior);
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardLogQ(const int nthreads, const int K, const Dtype* res_q_data, const Dtype* sum_res_q_data,
	const Dtype scale, Dtype* logq_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		logq_diff[index] = -res_q_data[index] / sum_res_q_data[index / K] * scale;
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardZ(const int nthreads, const int K, const int D, const Dtype* z_data,
	const Dtype* mu_z_data, const Dtype* sd_z_data, const Dtype* logp_diff, const Dtype* sd_gt, const Dtype* logq_diff,
	Dtype* z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int n = index / D;
		int d = index % D;
		z_diff[index] = 0;
		z_diff[index] += (mu_z_data[index] - z_data[index]) / pow(max(Dtype(1e-6), sd_z_data[index]), 2) * logp_diff[n];
		for (int k = 0; k < K; ++k)
			z_diff[index] -= z_data[index] / pow(sd_gt[k*D + d], 2) * logq_diff[n*K + k];
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardMu(const int nthreads, const int D,
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data, const Dtype* logp_diff, Dtype* mu_z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		mu_z_diff[index] = (z_data[index] - mu_z_data[index]) / pow(max(Dtype(1e-6), sd_z_data[index]), 2) * logp_diff[index / D];
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardSd(const int nthreads, const int D,
	const Dtype* z_data, const Dtype* mu_z_data, const Dtype* sd_z_data, const Dtype* logp_diff, Dtype* sd_z_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const Dtype safe_sd = max(Dtype(1e-6), sd_z_data[index]);
		sd_z_diff[index] = (pow((z_data[index] - mu_z_data[index]) / safe_sd, 2) - 1) / safe_sd * logp_diff[index/D];
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardPrior(const int nthreads, const int N,
	const Dtype* logq_diff, const Dtype* prior_data, Dtype* prior_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		prior_diff[index] = Dtype(0);
		int idx = index;
		for (int n = 0; n < N; ++n) {
			prior_diff[index] += logq_diff[idx];
			idx += nthreads;
		}
		prior_diff[index] /= max(Dtype(1e-6), prior_data[index]);
	}
}

template <typename Dtype>
__global__ void KLSubspaceBackwardSdGt(const int nthreads, const int N, const int K, const int D,
	const Dtype* z_data, const Dtype* sd_gt, const Dtype* logq_diff, Dtype* sd_gt_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		sd_gt_diff[index] = 0;
		int d = index % D;
		int k = index / D;
		for (int n = 0; n < N; ++n) {
			sd_gt_diff[index] += (pow(z_data[n*D + d] / sd_gt[index], 2) - 1) / sd_gt[index] * logq_diff[n*K + k];
		}
	}
}

template <typename Dtype>
void KLSubspaceLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype scale = top[0]->mutable_cpu_diff()[0] / Dtype(N_);

	const Dtype* res_q_data = res_q_.gpu_data();
	const Dtype* sum_res_q_data = sum_res_q_.gpu_data();
	Dtype* logq_diff = logq_.mutable_gpu_diff();
	KLSubspaceBackwardLogQ<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
		res_q_data, sum_res_q_data, scale, logq_diff);

	Dtype* logp_diff = logp_.mutable_gpu_diff();
	caffe_gpu_set<Dtype>(N_, scale, logp_diff);

	const Dtype* z_data = bottom[0]->gpu_data();
	const Dtype* mu_z_data = bottom[1]->gpu_data();
	const Dtype* sd_z_data = bottom[2]->gpu_data();
	const Dtype* sd_gt = gt_sd_.gpu_data();
	const Dtype* prior_data = bottom[3]->gpu_data();
	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_gpu_diff();
		KLSubspaceBackwardZ<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, K_, D_, z_data, mu_z_data, sd_z_data, logp_diff,
			sd_gt, logq_diff, z_diff);
	}
	if (propagate_down[1]) {
		Dtype* mu_z_diff = bottom[1]->mutable_gpu_diff();
		KLSubspaceBackwardMu<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, D_,
			z_data, mu_z_data, sd_z_data, logp_diff, mu_z_diff);
	}
	if (propagate_down[2]) {
		Dtype* sd_z_diff = bottom[2]->mutable_gpu_diff();
		KLSubspaceBackwardSd<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, D_,
			z_data, mu_z_data, sd_z_data, logp_diff, sd_z_diff);
	}
	if (propagate_down[3]) {
		Dtype* prior_diff = bottom[3]->mutable_gpu_diff();
		KLSubspaceBackwardPrior<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>(K_, N_,
			logq_diff, prior_data, prior_diff);
	}
	if (propagate_down[4]) {
		Dtype* sd_gt_diff = gt_sd_.mutable_gpu_diff();
		KLSubspaceBackwardSdGt<Dtype><<<CAFFE_GET_BLOCKS(K_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*D_, N_, K_, D_,
			z_data, sd_gt, logq_diff, sd_gt_diff);

		sd_gt_diff = gt_sd_.mutable_cpu_diff();
		Dtype noise_sd_diff = 0;
		const int count = K_*D_;
		for (int i = 0; i < count; ++i) {
			noise_sd_diff += sd_gt_diff[i];
		}
		bottom[4]->mutable_cpu_diff()[0] = noise_sd_diff;
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(KLSubspaceLossLayer);

}  // namespace caffe
