#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/self_assign_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);
	lambda_ = this->layer_param_.self_assign_loss_param().lambda();
	if (this->blobs_.size() > 0) {
		LOG(INFO) << "Skipping parameter initialization";
	}
	else {
		this->blobs_.resize(1);
		vector<int> weight_shape(2, N_);
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		// fill the weights
		shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
			this->layer_param_.self_assign_loss_param().weight_filler()));
		weight_filler->Fill(this->blobs_[0].get());
	}  // parameter initialization
	this->param_propagate_down_.resize(this->blobs_.size(), true);
	w_sign_.Reshape(vector<int>(2, N_));
}

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);
	z_hat_.ReshapeLike(*bottom[0]);
	CHECK_EQ(bottom[0]->shape(0), N_) << "The num of bottom should keep the same.";
}

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* z_data = bottom[0]->cpu_data();
	Dtype* z_hat_data = z_hat_.mutable_cpu_data();
	Dtype* weight_data = this->blobs_[0]->mutable_cpu_data();
	for (int n = 0; n < N_; ++n)
		weight_data[n*N_ + n] = Dtype(0);
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_, D_, N_,
		Dtype(1), weight_data, z_data, Dtype(0), z_hat_data);

	Dtype* diff = z_hat_.mutable_cpu_diff();
	caffe_sub<Dtype>(N_*D_, z_hat_data, z_data, diff);
	Dtype loss_diff = caffe_cpu_dot<Dtype>(N_*D_, diff, diff) / N_ / 2;

	Dtype loss_l1 = caffe_cpu_asum<Dtype>(N_*N_, weight_data) * lambda_ / N_ / N_;
	top[0]->mutable_cpu_data()[0] = loss_diff + loss_l1;
}

template <typename Dtype>
void SelfAssignLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	Dtype* z_hat_diff = z_hat_.mutable_cpu_diff();
	const Dtype scale = top[0]->cpu_diff()[0] / Dtype(N_);
	caffe_cpu_scale<Dtype>(N_*D_, scale, z_hat_diff, z_hat_diff);
	
	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_cpu_diff();
		caffe_cpu_axpby<Dtype>(N_*D_, Dtype(-1), z_hat_diff, Dtype(0), z_diff);
		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, D_, N_,
			Dtype(1), weight_data, z_hat_diff, Dtype(1), z_diff);
	}

	if (param_propagate_down_[0]) {
		const Dtype* z_data = bottom[0]->cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
		caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, N_, N_, D_,
			Dtype(1), z_hat_diff, z_data, Dtype(1), weight_diff);

		const Dtype* weight_data = this->blobs_[0]->cpu_data();
		Dtype* w_sign_data = w_sign_.mutable_cpu_data();
		caffe_cpu_sign<Dtype>(N_*N_, weight_data, w_sign_data);
		caffe_cpu_axpby<Dtype>(N_*N_, lambda_ / N_ / N_, w_sign_data, Dtype(1), weight_diff);
	}
}

#ifdef CPU_ONLY
STUB_GPU(SelfAssignLossLayer);
#endif

INSTANTIATE_CLASS(SelfAssignLossLayer);
REGISTER_LAYER_CLASS(SelfAssignLoss);

}  // namespace caffe
