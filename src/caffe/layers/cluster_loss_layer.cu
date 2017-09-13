#include <vector>

#include "caffe/layers/cluster_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ClusterLossForwardLogJoint(const int nthreads, const int K,
	const Dtype* log_posterior_data, const Dtype* prior_data, Dtype* log_joint_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		log_joint_data[index] = log_posterior_data[index] + log(max(Dtype(1e-12), prior_data[index % K]));
	}
}

template <typename Dtype>
__global__ void ClusterLossForwardLogJointMax(const int nthreads, const int K,
	const Dtype* log_joint_data, Dtype* log_joint_max_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		log_joint_max_data[index] = -INT_MAX;
		int idx = index*K;
		for (int k = 0; k < K; ++k)
			log_joint_max_data[index] = max(log_joint_max_data[index], log_joint_data[idx++]);
	}
}

template <typename Dtype>
__global__ void ClusterLossForwardJointRes(const int nthreads, const int K,
	const Dtype* log_joint_data, const Dtype* log_joint_max_data, Dtype* joint_res_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		joint_res_data[index] = exp(log_joint_data[index] - log_joint_max_data[index / K]);
	}
}

template <typename Dtype>
__global__ void ClusterLossForwardItemLoss(const int nthreads, const int K,
	const Dtype* log_joint_max_data, const Dtype* joint_res_data, Dtype* item_loss_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		item_loss_data[index] = 0;
		int idx = index*K;
		for (int k = 0; k < K; ++k)
			item_loss_data[index] += joint_res_data[idx++];
		item_loss_data[index] = -log_joint_max_data[index] - log(item_loss_data[index]);
	}
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* log_posterior_data = bottom[0]->gpu_data();
	const Dtype* prior_data = bottom[1]->gpu_data();
	Dtype* log_joint_data = log_joint_.mutable_gpu_data();
	ClusterLossForwardLogJoint<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
		log_posterior_data, prior_data, log_joint_data);

	Dtype* log_joint_max_data = log_joint_max_.mutable_gpu_data();
	ClusterLossForwardLogJointMax<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_,
		log_joint_data, log_joint_max_data);

	Dtype* joint_res_data = joint_res_.mutable_gpu_data();
	ClusterLossForwardJointRes<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_,
		log_joint_data, log_joint_max_data, joint_res_data);

	Dtype* item_loss_data = item_loss_.mutable_gpu_data();
	ClusterLossForwardItemLoss<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_,
		log_joint_max_data, joint_res_data, item_loss_data);

	item_loss_data = item_loss_.mutable_cpu_data();
	Dtype loss = 0;
	for (int n = 0; n < N_; ++n)
		loss += item_loss_data[n];
	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
__global__ void ClusterLossBackwardJointResSum(const int nthreads, const int K,
	const Dtype* joint_res_data, Dtype* joint_res_sum_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		joint_res_sum_data[index] = 0;
		int idx = index * K;
		for (int k = 0; k < K; ++k)
			joint_res_sum_data[index] += joint_res_data[idx++];
	}
}

template <typename Dtype>
__global__ void ClusterLossBackwardLogJoint(const int nthreads, const int K, const Dtype scale,
	const Dtype* joint_res_data, const Dtype* joint_res_sum_data, Dtype* log_joint_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		log_joint_diff[index] = -joint_res_data[index] / joint_res_sum_data[index / K] * scale;
	}
}

template <typename Dtype>
__global__ void ClusterLossBackwardPrior(const int nthreads, const int N,
	const Dtype* log_joint_diff, const Dtype* prior_data, Dtype* prior_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		prior_diff[index] = 0;
		int idx = index;
		for (int n = 0; n < N; ++n) {
			prior_diff[index] += log_joint_diff[idx];
			idx += nthreads;
		}
		prior_diff[index] /= max(Dtype(1e-12), prior_data[index]);
	}
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if ((!propagate_down[0]) && (!propagate_down[1]))
		return;

	const Dtype* joint_res_data = joint_res_.gpu_data();
	const Dtype scale = top[0]->cpu_diff()[0] / N_;
	Dtype* joint_res_sum_data = item_loss_.mutable_gpu_diff();
	ClusterLossBackwardJointResSum<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, K_,
		joint_res_data, joint_res_sum_data);

	Dtype* log_joint_diff = log_joint_.mutable_gpu_diff();
	ClusterLossBackwardLogJoint<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>(N_*K_, K_, scale,
		joint_res_data, joint_res_sum_data, log_joint_diff);

	if (propagate_down[0])
		caffe_copy<Dtype>(bottom[0]->count(), log_joint_diff, bottom[0]->mutable_gpu_diff());
	if (propagate_down[1]) {
		const Dtype* bottom1_data = bottom[1]->gpu_data();
		Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();
		ClusterLossBackwardPrior<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>(K_, N_,
			log_joint_diff, bottom1_data, bottom1_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ClusterLossLayer);

}  // namespace caffe
