#include <vector>

#include "caffe/layers/log_probability_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void LogProbabilityLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LayerParameter concat_param(this->layer_param_);
	concat_param.set_type("Concat");
	concat_param.mutable_concat_param()->set_axis(0);
	concat_layer_ = LayerRegistry<Dtype>::CreateLayer(concat_param);
	concat_layer_bottom_.clear();
	K_ = bottom[0]->shape(0);
	for (int k = 0; k < K_; ++k)
		concat_layer_bottom_.push_back(bottom[1]);
	concat_layer_top_.clear();
	concat_layer_top_.push_back(&x_duplicate_);
	concat_layer_->SetUp(concat_layer_bottom_, concat_layer_top_);
}

template <typename Dtype>
void LogProbabilityLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	K_ = bottom[0]->shape(0);
	N_ = bottom[0]->shape(1);
	D_ = bottom[0]->count(2);
	CHECK_EQ(bottom[1]->shape(0), N_) << "The num of x_hat and x should be the same.";
	CHECK_EQ(bottom[1]->count(1), D_) << "The dim of x_hat and x should be the same.";
	vector<int> top_shape;
	top_shape.push_back(N_);
	top_shape.push_back(K_);
	top[0]->Reshape(top_shape);
	vector<int> p_dim_shape;
	p_dim_shape.push_back(K_);
	p_dim_shape.push_back(N_);
	p_dim_shape.push_back(D_);
	p_dim_.Reshape(p_dim_shape);
}

template <typename Dtype>
void LogProbabilityLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	concat_layer_->Forward(concat_layer_bottom_, concat_layer_top_);
	const Dtype* x_hat_data = bottom[0]->cpu_data();
	const Dtype* x_data = x_duplicate_.cpu_data();
	Dtype* p_dim_data = p_dim_.mutable_cpu_data();
	Dtype* p_data = top[0]->mutable_cpu_data();
	int idx = 0;
	for (int k = 0; k < K_; ++k)
	for (int n = 0; n < N_; ++n) {
		int p_idx = n*K_ + k;
		p_data[p_idx] = 0;
		for (int d = 0; d < D_; ++d) {
			p_dim_data[idx] = x_data[idx] * log(max(Dtype(1e-12), x_hat_data[idx]))
				+ (1 - x_data[idx]) * log(max(Dtype(1e-12), Dtype(1 - x_hat_data[idx])));
			p_data[p_idx] += p_dim_data[idx++];
		}
	}
}

template <typename Dtype>
void LogProbabilityLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const Dtype* p_diff = top[0]->cpu_diff();
	const Dtype* x_hat_data = bottom[0]->cpu_data();
	const Dtype* x_data = x_duplicate_.cpu_data();
	if (propagate_down[0]) {
		Dtype* x_hat_diff = bottom[0]->mutable_cpu_diff();
		int idx = 0;
		for (int k = 0; k < K_; ++k)
		for (int n = 0; n < N_; ++n) {
			int p_idx = n*K_ + k;
			for (int d = 0; d < D_; ++d) {
				x_hat_diff[idx] = (x_data[idx] - x_hat_data[idx]) /
					max(Dtype(1e-12), x_hat_data[idx] * (1 - x_hat_data[idx]));
				x_hat_diff[idx++] *= p_diff[p_idx];
			}
		}
	}
	if (propagate_down[1]) {
		Dtype* x_diff = x_duplicate_.mutable_cpu_diff();
		int idx = 0;
		for (int k = 0; k < K_; ++k)
		for (int n = 0; n < N_; ++n) {
			int p_idx = n*K_ + k;
			for (int d = 0; d < D_; ++d) {
				x_diff[idx] = log(max(Dtype(1e-12), x_hat_data[idx]))
					- log(max(Dtype(1e-12), Dtype(1) - x_hat_data[idx]));
				x_diff[idx++] *= p_diff[p_idx];
			}
		}
		Dtype* bottom_diff = bottom[1]->mutable_cpu_diff();
		caffe_set<Dtype>(bottom[1]->count(), Dtype(0), bottom_diff);
		for (int k = 0; k < K_; ++k)
			caffe_cpu_axpby<Dtype>(N_*D_, Dtype(1), x_diff + k*N_*D_, Dtype(1), bottom_diff);
	}
} 

#ifdef CPU_ONLY
STUB_GPU(LogProbabilityLayer);
#endif

INSTANTIATE_CLASS(LogProbabilityLayer);
REGISTER_LAYER_CLASS(LogProbability);

}  // namespace caffe
