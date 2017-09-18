#include <vector>

#include "caffe/layers/separate_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SeparateLossLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	delta_ = this->layer_param_.separate_loss_param().delta();
	CHECK_GT(delta_, 0) << "Delta in separate loss should be greater than 0.";
}

template <typename Dtype>
void SeparateLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	K_ = bottom[0]->shape(0);
	N_ = bottom[0]->shape(1);
	D_ = bottom[0]->count(2);

	vector<int> dis_shape;
	dis_shape.push_back(N_);
	dis_shape.push_back(K_);
	dis_shape.push_back(K_);
	dis_.Reshape(dis_shape);
	loss_.Reshape(dis_shape);
}

template <typename Dtype>
void SeparateLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* x_data = bottom[0]->cpu_data();
	Dtype* dis_data = dis_.mutable_cpu_data();
	Dtype* loss_data = loss_.mutable_cpu_data();
	Dtype loss = 0;
	for (int n = 0; n < N_; ++n)
	for (int k1 = 0; k1 < K_; ++k1)
	for (int k2 = 0; k2 < K_; ++k2) {
		int dis_idx = (n * K_ + k1) * K_ + k2;
		int x_idx1 = (k1*N_ + n)*D_;
		int x_idx2 = (k2*N_ + n)*D_;
		dis_data[dis_idx] = 0;
		for (int d = 0; d < D_; ++d)
			dis_data[dis_idx] += abs(x_data[x_idx1++] - x_data[x_idx2++]);
		if (k1 == k2)
			loss_data[dis_idx] = 0;
		else
			loss_data[dis_idx] = max(Dtype(0), delta_ - dis_data[dis_idx]);
		loss += loss_data[dis_idx];
	}
	top[0]->mutable_cpu_data()[0] = loss / N_ / (K_*K_ - K_);
}

template <typename Dtype>
void SeparateLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;
	const Dtype scale = top[0]->cpu_diff()[0] / N_ / (K_*K_ - K_);
	const Dtype* x_data = bottom[0]->cpu_data();
	const Dtype* loss_data = loss_.cpu_data();
	Dtype* dis_diff = dis_.mutable_cpu_diff();
	for (int n = 0; n < N_; ++n)
	for (int k1 = 0; k1 < K_; ++k1)
	for (int k2 = 0; k2 < K_; ++k2) {
		int dis_idx = (n * K_ + k1) * K_ + k2;
		if (loss_data[dis_idx] == 0)
			dis_diff[dis_idx] = 0;
		else
			dis_diff[dis_idx] = -scale;
	}
	Dtype* x_diff = bottom[0]->mutable_cpu_diff();
	for (int k = 0; k < K_; ++k)
	for (int n = 0; n < N_; ++n)
	for (int d = 0; d < D_; ++d) {
		int x_idx = (k*N_ + n)*D_ + d;
		x_diff[x_idx] = 0;
		int dis_idx = (n*K_ + k)*K_;
		for (int k_prime = 0; k_prime < K_; ++k_prime) {
			int x_idx_prime = (k_prime*N_ + n)*D_ + d;
			x_diff[x_idx] += dis_diff[dis_idx++] * (2 * (x_data[x_idx] > x_data[x_idx_prime]) - 1);
		}
		x_diff[x_idx] *= 2;
	}
}

#ifdef CPU_ONLY
STUB_GPU(SeparateLossLayer);
#endif

INSTANTIATE_CLASS(SeparateLossLayer);
REGISTER_LAYER_CLASS(SeparateLoss);

}  // namespace caffe
