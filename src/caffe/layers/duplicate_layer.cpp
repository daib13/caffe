#include <vector>

#include "caffe/layers/duplicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DuplicateLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);
	K_ = this->layer_param_.duplicate_param().duplicate();
	vector<int> top_shape;
	top_shape.push_back(K_);
	top_shape.push_back(N_);
	top_shape.push_back(K_*D_);
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set<Dtype>(top[0]->count(), Dtype(0), top_data);
	for (int k = 0; k < K_; ++k)
	for (int n = 0; n < N_; ++n) {
		caffe_copy<Dtype>(D_, bottom_data + n*D_, top_data + (k*N_ + n)*K_*D_ + k*D_);
	}
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	caffe_set<Dtype>(bottom[0]->count(), Dtype(0), bottom_diff);
	for (int k = 0; k < K_; ++k)
	for (int n = 0; n < N_; ++n) {
		caffe_cpu_axpby<Dtype>(D_, Dtype(1), top_diff + (k*N_ + n)*K_*D_ + k*D_, Dtype(1), bottom_diff + n*D_);
	}
}

#ifdef CPU_ONLY
STUB_GPU(DuplicateLayer);
#endif

INSTANTIATE_CLASS(DuplicateLayer);
REGISTER_LAYER_CLASS(Duplicate);

}  // namespace caffe
