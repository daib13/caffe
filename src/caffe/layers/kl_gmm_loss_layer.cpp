#include <vector>

#include "caffe/layers/kl_gmm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLGMMLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	if (this->layer_param_.loss_weight_size() == 0) {
		if (top.size() == 1)
			this->layer_param_.add_loss_weight(Dtype(1));
		else if (top.size() == 2) {
			this->layer_param_.add_loss_weight(Dtype(1));
			this->layer_param_.add_loss_weight(Dtype(0));
		}
	}
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);
	CHECK_EQ(bottom[1]->shape(0), N_) << "The num of mu_z and z should be the same.";
	CHECK_EQ(bottom[1]->count(1), D_) << "The dim of mu_z and z should be the same.";
	CHECK_EQ(bottom[2]->shape(0), N_) << "The num of sd_z and z should be the same.";
	CHECK_EQ(bottom[2]->count(1), D_) << "The dim of sd_z and z should be the same.";
	K_ = bottom[3]->count();
	CHECK_EQ(bottom[4]->shape(0), K_) << "The num of mu_c and the count of the prior should be the same.";
	CHECK_EQ(bottom[4]->count(1), D_) << "The dim of mu_c and z should be the same.";
	CHECK_EQ(bottom[5]->shape(0), K_) << "The num of sd_c and the count of the prior should be the same.";
	CHECK_EQ(bottom[5]->count(1), D_) << "The dim of sd_c and z should be the same.";

	vector<int> logp_dim_shape;
	logp_dim_shape.push_back(N_);
	logp_dim_shape.push_back(D_);
	logp_dim_.Reshape(logp_dim_shape);

	logp_.Reshape(vector<int>(1, N_));

	vector<int> logq_dim_shape;
	logq_dim_shape.push_back(N_);
	logq_dim_shape.push_back(K_);
	logq_dim_shape.push_back(D_);
	logq_dim_.Reshape(logq_dim_shape);

	vector<int> logq_shape;
	logq_shape.push_back(N_);
	logq_shape.push_back(K_);
	logq_.Reshape(logq_shape);

	logq_max_.Reshape(vector<int>(1, N_));
	resq_.Reshape(logq_shape);
	resq_sum_.Reshape(vector<int>(1, N_));
	
	if (top.size() == 2)
		top[1]->Reshape(logq_shape);
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	const Dtype* z_data = bottom[0]->cpu_data();
	const Dtype* mu_z_data = bottom[1]->cpu_data();
	const Dtype* sd_z_data = bottom[2]->cpu_data();
	
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	
}

#ifdef CPU_ONLY
STUB_GPU(KLGMMLossLayer);
#endif

INSTANTIATE_CLASS(KLGMMLossLayer);
REGISTER_LAYER_CLASS(KLGMMLoss);

}  // namespace caffe
