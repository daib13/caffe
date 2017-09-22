#include <vector>

#include "caffe/layers/self_assign_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SelfAssignLossDeDiagonal(const int nthreads, Dtype* weight_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		weight_data[index*nthreads, index] = Dtype(0);
	}
}

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* z_data = bottom[0]->gpu_data();
	Dtype* z_hat_data = z_hat_.mutable_gpu_data();
	Dtype* weight_data = this->blobs_[0]->mutable_gpu_data();
	SelfAssignLossDeDiagonal<Dtype><<<CAFFE_GET_BLOCKS(N_), CAFFE_CUDA_NUM_THREADS>>>(N_, weight_data);
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, D_, N_,
		Dtype(1), weight_data, z_data, Dtype(0), z_hat_data);

	Dtype* diff = z_hat_.mutable_gpu_diff();
	caffe_gpu_sub<Dtype>(N_*D_, z_hat_data, z_data, diff);
	Dtype loss_diff;
	caffe_gpu_dot<Dtype>(N_*D_, diff, diff, &loss_diff);
	loss_diff /= (N_ * 2);

	Dtype loss_l1;
	caffe_gpu_asum<Dtype>(N_*N_, weight_data, &loss_l1);
	loss_l1 *= (lambda_ / N_ / N_);
	top[0]->mutable_cpu_data()[0] = loss_diff + loss_l1;
}

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype* z_hat_diff = z_hat_.mutable_gpu_diff();
	const Dtype scale = top[0]->cpu_diff()[0] / Dtype(N_);
	caffe_gpu_scale<Dtype>(N_*D_, scale, z_hat_diff, z_hat_diff);

	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_gpu_diff();
		caffe_gpu_axpby<Dtype>(N_*D_, Dtype(-1), z_hat_diff, Dtype(0), z_diff);
		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, D_, N_,
			Dtype(1), weight_data, z_hat_diff, Dtype(1), z_diff);
	}

	if (param_propagate_down_[0]) {
		const Dtype* z_data = bottom[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, D_,
			Dtype(1), z_hat_diff, z_data, Dtype(1), weight_diff);

		const Dtype* weight_data = this->blobs_[0]->gpu_data();
		Dtype* w_sign_data = w_sign_.mutable_gpu_data();
		caffe_gpu_sign<Dtype>(N_*N_, weight_data, w_sign_data);
		caffe_gpu_axpby<Dtype>(N_*N_, lambda_ / N_ / N_, w_sign_data, Dtype(1), weight_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SelfAssignLossLayer);

}  // namespace caffe
