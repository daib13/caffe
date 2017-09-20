#include <vector>

#include "caffe/layers/log_probability_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void LogProbabilityForwardDimBernoulli(const int nthreads, const Dtype* x_data, const Dtype* x_hat_data,
	Dtype* p_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		p_dim_data[index] = x_data[index] * log(max(Dtype(1e-12), x_hat_data[index]))
			+ (1 - x_data[index]) * log(max(Dtype(1e-12), Dtype(1 - x_hat_data[index])));
	}
}

template <typename Dtype>
__global__ void LogProbabilityForwardDimGaussian(const int nthreads, const Dtype* diff_data,
	const Dtype two_var, const Dtype log_sigma, Dtype* p_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		p_dim_data[index] = -pow(diff_data[index], 2) / two_var - LOG_TWO_PI / Dtype(2) - log_sigma;
	}
}

template <typename Dtype>
__global__ void LogProbabilityForwardDimGaussianWithVar(const int nthreads, const int K, const int N, const int D,
	const Dtype* diff_data, const Dtype* sd_data, Dtype* p_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int sd_idx = index / D / N * D + index % D;
		const Dtype safe_sd = max(Dtype(1e-6), sd_data[sd_idx]);
		p_dim_data[index] = (-pow(diff_data[index] / safe_sd, 2) - LOG_TWO_PI) / Dtype(2) - log(safe_sd);
	}
}

template <typename Dtype>
__global__ void LogProbabilityForwardDimOptGaussian(const int nthreads, const Dtype* diff_data, const Dtype epsilon,
	Dtype* p_dim_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		p_dim_data[index] = -(LOG_TWO_PI + log(pow(diff_data[index], 2) + epsilon) + 1) / Dtype(2);
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
	if (distance_type_ == LogProbabilityParameter_DistanceType_BERNOULLI) {
		LogProbabilityForwardDimBernoulli<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_,
			x_data, x_hat_data, p_dim_data);
	}
	else if (distance_type_ == LogProbabilityParameter_DistanceType_GAUSSIAN) {
		Dtype* diff_data = p_dim_.mutable_gpu_diff();
		caffe_gpu_sub<Dtype>(bottom[0]->count(), x_data, x_hat_data, diff_data);
		if (bottom.size() == 2) {
			LogProbabilityForwardDimGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_,
				diff_data, two_var_, log_sigma_, p_dim_data);
		}
		else if (bottom.size() == 3) {
			const Dtype* sd_data = bottom[2]->gpu_data();
			LogProbabilityForwardDimGaussianWithVar<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				diff_data, sd_data, p_dim_data);
		}
	}
	else if (distance_type_ == LogProbabilityParameter_DistanceType_OPT_GAUSSIAN) {
		Dtype* diff_data = p_dim_.mutable_gpu_diff();
		caffe_gpu_sub<Dtype>(bottom[0]->count(), x_data, x_hat_data, diff_data);
		LogProbabilityForwardDimOptGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_,
			diff_data, epsilon_, p_dim_data);
	}

	Dtype* p_data = top[0]->mutable_gpu_data();
	LogProbabilityForwardP<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_, N_, D_, 
		p_dim_data, p_data);
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXhatBernoulli(const int nthreads, const int K, const int N, const int D,
	const Dtype* x_data, const Dtype* x_hat_data,
	const Dtype* p_diff, Dtype* x_hat_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_hat_diff[index] = (x_data[index] - x_hat_data[index]) /
			max(Dtype(1e-12), x_hat_data[index] * (1 - x_hat_data[index])) * p_diff[p_idx];
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXhatGaussian(const int nthreads, const int K, const int N, const int D,
	const Dtype* diff_data, const Dtype* sd_data, const Dtype two_var, const Dtype* p_diff, Dtype* x_hat_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		if (sd_data == NULL)
			x_hat_diff[index] = 2 * diff_data[index] / two_var * p_diff[p_idx];
		else {
			int sd_idx = (index / D / N) * D + index % D;
			x_hat_diff[index] = diff_data[index] / pow(max(Dtype(1e-6), sd_data[sd_idx]), 2) * p_diff[p_idx];
		}
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXhatOptGaussian(const int nthreads, const int K, const int N, const int D,
	const Dtype* diff_data, const Dtype epsilon, const Dtype* p_diff, Dtype* x_hat_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_hat_diff[index] = diff_data[index] / (pow(diff_data[index], 2) + epsilon) * p_diff[p_idx];
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXBernoulli(const int nthreads, const int K, const int N, const int D,
	const Dtype* x_data, const Dtype* x_hat_data,
	const Dtype* p_diff, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_diff[index] = p_diff[p_idx] * (log(max(Dtype(1e-12), x_hat_data[index]))
			- log(max(Dtype(1e-12), Dtype(1) - x_hat_data[index])));
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXGaussian(const int nthreads, const int K, const int N, const int D,
	const Dtype* diff_data, const Dtype* sd_data, const Dtype two_var, const Dtype* p_diff, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		if (sd_data == NULL)
			x_diff[index] = -2 * diff_data[index] / two_var * p_diff[p_idx];
		else {
			int sd_idx = (index / D / N) * D + index % D;
			x_diff[index] = -diff_data[index] / pow(max(Dtype(1e-6), sd_data[sd_idx]), 2) * p_diff[p_idx];
		}
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardXOptGaussian(const int nthreads, const int K, const int N, const int D,
	const Dtype* diff_data, const Dtype epsilon, const Dtype* p_diff, Dtype* x_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int p_idx = (index / D % N) * K + index / D / N;
		x_diff[index] = -diff_data[index] / (pow(diff_data[index], 2) + epsilon) * p_diff[p_idx];
	}
}

template <typename Dtype>
__global__ void LogProbabilityBackwardVar(const int nthreads, const int K, const int N, const int D,
	const Dtype* sd_data, const Dtype* p_diff, const Dtype* diff_data, Dtype* sd_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		sd_diff[index] = 0;
		const Dtype safe_sd = max(Dtype(1e-6), sd_data[index]);
		int k = index / D;
		int d = index % D;
		for (int n = 0; n < N; ++n) {
			int p_idx = n*K + k;
			int diff_idx = (k*N + n)*D + d;
			sd_diff[index] += p_diff[p_idx] * (pow(diff_data[diff_idx] / safe_sd, 2) - 1);
		}
		sd_diff[index] /= safe_sd;
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
		if (distance_type_ == LogProbabilityParameter_DistanceType_BERNOULLI) {
			LogProbabilityBackwardXhatBernoulli<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				x_data, x_hat_data, p_diff, x_hat_diff);
		}
		else if (distance_type_ == LogProbabilityParameter_DistanceType_GAUSSIAN) {
			const Dtype* diff_data = p_dim_.gpu_diff();
			const Dtype* sd_data = NULL;
			if (bottom.size() == 3)
				sd_data = bottom[2]->gpu_data();
			LogProbabilityBackwardXhatGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				diff_data, sd_data, two_var_, p_diff, x_hat_diff);
		}
		else if (distance_type_ == LogProbabilityParameter_DistanceType_OPT_GAUSSIAN) {
			const Dtype* diff_data = p_dim_.gpu_diff();
			LogProbabilityBackwardXhatOptGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				diff_data, epsilon_, p_diff, x_hat_diff);
		}
	}
	if (propagate_down[1]) {
		Dtype* x_diff = x_duplicate_.mutable_gpu_diff();
		if (distance_type_ == LogProbabilityParameter_DistanceType_BERNOULLI) {
			LogProbabilityBackwardXBernoulli<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				x_data, x_hat_data, p_diff, x_diff);
		}
		else if (distance_type_ == LogProbabilityParameter_DistanceType_GAUSSIAN) {
			const Dtype* diff_data = p_dim_.gpu_diff();
			const Dtype* sd_data = NULL;
			if (bottom.size() == 3)
				sd_data = bottom[2]->gpu_data();
			LogProbabilityBackwardXGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				diff_data, sd_data, two_var_, p_diff, x_diff);
		}
		else if (distance_type_ == LogProbabilityParameter_DistanceType_OPT_GAUSSIAN) {
			const Dtype* diff_data = p_dim_.gpu_diff();
			LogProbabilityBackwardXOptGaussian<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
				diff_data, epsilon_, p_diff, x_diff);
		}
		Dtype* bottom_diff = bottom[1]->mutable_gpu_diff();
		caffe_gpu_set<Dtype>(bottom[1]->count(), Dtype(0), bottom_diff);
		for (int k = 0; k < K_; ++k)
			caffe_gpu_axpby<Dtype>(N_*D_, Dtype(1), x_diff + k*N_*D_, Dtype(1), bottom_diff);
	}
	if (bottom.size() == 3 && propagate_down[2]) {
		const Dtype* diff_data = p_dim_.gpu_diff();
		const Dtype* sd_data = bottom[2]->gpu_data();
		Dtype* sd_diff = bottom[2]->mutable_gpu_diff();
		LogProbabilityBackwardVar<Dtype><<<CAFFE_GET_BLOCKS(K_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*D_, K_, N_, D_,
			sd_data, p_diff, diff_data, sd_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(LogProbabilityLayer);

}  // namespace caffe
