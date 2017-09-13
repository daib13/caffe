#include <vector>

#include "caffe/layers/vae_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void VAELossForwardDist(const int nthreads, const int D,
	const Dtype* diff, Dtype* dist) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		dist[index] = 0;
		int diff_idx = index * D;
		for (int d = 0; d < D; ++d)
			dist[index] += pow(diff[diff_idx++], 2);
	}
}

// loss = log(2*pi) + log(variance) + (x-x_hat)**2/variance
// to protect the log and divide operation: variance -> variance + epsilon
template <typename Dtype>
__global__ void VAELossForwardWithVar(const int nthreads, const Dtype* diff, const Dtype* variance,
	const int denominator, Dtype epsilon, Dtype* loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int var_idx = index / denominator;
		loss[index] = log(variance[var_idx] + epsilon) + pow(diff[index], 2) / (variance[var_idx] + epsilon) + LOG_TWO_PI;
	}
}

// 1) pixel wise case: loss = log(2*pi) + log(diff**2) + 1
//    to protect log operation: diff**2 -> diff**2 + epsilon
// 2) item wise case: loss = log(2*pi) + log(dist) + 1/D
//    to protect log operation: dist -> dist + epsilon
template <typename Dtype>
__global__ void VAELossForwardWithoutVar(const int nthreads, const int D,
	const Dtype* diff, const Dtype* dist_data,
	const Dtype epsilon, Dtype* loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		if (dist_data == NULL)
			loss[index] = log(pow(diff[index], 2) + epsilon) + LOG_TWO_PI + 1;
		else
			loss[index] = log(dist_data[index / D] + epsilon) + LOG_TWO_PI + Dtype(1) / Dtype(D);
	}
}

template <typename Dtype>
void VAELossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	// diff = x - x_hat
	int count = bottom[0]->count();
	caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), diff_.mutable_gpu_data());
	if (alpha_ > 0) {
		Dtype l2_dis;
		caffe_gpu_dot(diff_.count(), diff_.gpu_data(), diff_.gpu_data(), &l2_dis);
		l2_dis /= diff_.count();
		mean_l2_ = this->blobs_[0]->cpu_data()[0];
		mean_l2_ = 0.99 * mean_l2_ + 0.01 * l2_dis;
		epsilon_ = alpha_ * mean_l2_;
		this->blobs_[0]->mutable_cpu_data()[0] = mean_l2_;
	}

	// calculate dist in the item wise case
	// dist = \sum_d pow(diff[d], 2)
	const Dtype* diff_data = diff_.gpu_data();
	if (variance_type_ == VAELossParameter_VarianceType_ITEM) {
		Dtype* dist_data = dist_.mutable_gpu_data();
		VAELossForwardDist<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, D_,
			diff_data, dist_data);
	}

	// 1. if input variance
	// loss = log(2*pi) + log(variance) + (x-x_hat)**2/variance
	// to protect the log and divide operation: variance -> variance + epsilon
	Dtype* loss_data = diff_.mutable_gpu_diff();
	if (bottom.size() == 3) {
		const Dtype* variance_data = bottom[2]->gpu_data();
		int denominator = (variance_type_ == VAELossParameter_VarianceType_ITEM) ? D_ : 1;
		VAELossForwardWithVar<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
			diff_data, variance_data, denominator, epsilon_, loss_data);
	}
	// 2. if no input variance
	// 1) pixel wise case: loss = log(2*pi) + log(diff**2) + 1
	//    to protect log operation: diff**2 -> diff**2 + epsilon
	// 2) item wise case: loss = log(2*pi) + log(dist) + 1/D
	//    to protect log operation: dist -> dist + epsilon
	else {
		const Dtype* dist_data = NULL;
		if (variance_type_ == VAELossParameter_VarianceType_ITEM)
			dist_data = dist_.gpu_data();
		VAELossForwardWithoutVar<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, D_, 
			diff_data, dist_data, epsilon_, loss_data);
	}

	Dtype loss = 0;
	loss_data = diff_.mutable_cpu_diff();
	for (int i = 0; i < count; ++i)
		loss += loss_data[i];
	top[0]->mutable_cpu_data()[0] = loss / N_ / 2;
}

// dL/dx = (x-x_hat) / variance (variance -> variance + epsilon)
template <typename Dtype>
__global__ void VAELossBackwardXWithVar(const int nthreads,
	const Dtype* variance, const Dtype* diff, const int denominator, const Dtype epsilon, const Dtype scale,
	Dtype* diff1, Dtype* diff2) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		diff1[index] = diff[index] / (variance[index / denominator] + epsilon) * scale;
		diff2[index] = -diff1[index];
	}
}

// pixel wise case
//    dL/dvariance = 1/variance - (x-x_hat)**2/variance**2 (variance -> variance + epsilon)
template <typename Dtype>
__global__ void VAELossBackwardVarPixelwise(const int nthreads, const Dtype scale,
	const Dtype* variance, const Dtype* diff, Dtype epsilon, Dtype* variance_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		variance_diff[index] = Dtype(1) / (variance[index] + epsilon);
		variance_diff[index] -= pow(diff[index] / (variance[index] + epsilon), 2);
		variance_diff[index] *= scale;
	}
}

// 2) item wise case
//    dL/dvariance = D/variance - dist/variance**2 (variance -> variance + epsilon)
template <typename Dtype>
__global__ void VAELossBackwardVarItemwise(const int nthreads, const int D, const Dtype scale,
	const Dtype* dist, const Dtype* variance, const Dtype epsilon, Dtype* variance_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		variance_diff[index] = Dtype(D) / (variance[index] + epsilon) - dist[index] / pow(variance[index] + epsilon, 2);
		variance_diff[index] *= scale;
	}
}

// pixel wise case (diff**2 -> diff**2 + epsilon)
//    dL/dx = (x-x_hat)/(x-x_hat)**2
template <typename Dtype>
__global__ void VAELossBackwardXWithoutVarPixelwise(const int nthreads,
	const Dtype* diff, const Dtype epsilon, const Dtype scale,
	Dtype* diff1, Dtype* diff2) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		diff1[index] = diff[index] / (pow(diff[index], 2) + epsilon) * scale;
		diff2[index] = -diff1[index];
	}
}

// 2) item wise case (dist -> dist + epsilon)
//    dL/dx = D*(x-x_hat)/dist
template <typename Dtype>
__global__ void VAELossBackwardXWithoutVarItemwise(const int nthreads, const int D,
	const Dtype* diff, const Dtype* dist, const Dtype epsilon,
	const Dtype scale, Dtype* diff1, Dtype* diff2) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		diff1[index] = D * diff[index] / (dist[index / D] + epsilon) * scale;
		diff2[index] = -diff1[index];
	}
}

template <typename Dtype>
void VAELossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
	const Dtype* diff_data = diff_.gpu_data();
	const int count = bottom[0]->count();

	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
	Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();
	// 1. if input variance (variance -> variance + epsilon)
	if (bottom.size() == 3) {
		// backward to x and x_hat
		// dL/dx = (x-x_hat) / variance
		const Dtype* variance_data = bottom[2]->gpu_data();
		int denominator = (variance_type_ == VAELossParameter_VarianceType_ITEM) ? D_ : 1;
		VAELossBackwardXWithVar<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
			variance_data, diff_data, denominator, epsilon_, loss_weight / N_,
			bottom0_diff, bottom1_diff);

		// backward to variance
		// 1) pixel wise case
		//    dL/dvariance = 1/variance - (x-x_hat)**2/variance**2
		Dtype* variance_diff = bottom[2]->mutable_gpu_diff();
		if (variance_type_ == VAELossParameter_VarianceType_PIXEL) {
			VAELossBackwardVarPixelwise<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, loss_weight / N_ / 2,
				variance_data, diff_data, epsilon_, variance_diff);
		}
		// 2) item wise case
		//    dL/dvariance = D/variance - dist/variance**2 
		else {
			const Dtype* dist_data = dist_.gpu_data();
			VAELossBackwardVarItemwise<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, D_, loss_weight / N_ /2,
				dist_data, variance_data, epsilon_, variance_diff);
		}
	}
	// 2. if no input variance (only need to backward to x and x_hat)
	else {
		// 1) pixel wise case (diff**2 -> diff**2 + epsilon)
		//    dL/dx = (x-x_hat)/(x-x_hat)**2
		if (variance_type_ == VAELossParameter_VarianceType_PIXEL) {
			VAELossBackwardXWithoutVarPixelwise<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
				diff_data, epsilon_, loss_weight / N_, bottom0_diff, bottom1_diff);
		}
		// 2) item wise case (dist -> dist + epsilon)
		//    dL/dx = D*(x-x_hat)/dist
		else {
			const Dtype* dist_data = dist_.gpu_data();
			VAELossBackwardXWithoutVarItemwise<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
				D_, diff_data, dist_data, epsilon_, loss_weight / N_, bottom0_diff, bottom1_diff);
		}
	}

}

INSTANTIATE_LAYER_GPU_FUNCS(VAELossLayer);

}  // namespace caffe
