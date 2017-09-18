#include <vector>

#include "caffe/layers/dpp_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DppLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	K_ = bottom[0]->shape(0);
	N_ = bottom[0]->shape(1);
	D_ = bottom[0]->count(2);

	vector<int> correlate_shape;
	correlate_shape.push_back(N_);
	correlate_shape.push_back(K_);
	correlate_shape.push_back(K_);
	correlate_matrix_.Reshape(correlate_shape);
	
	vector<int> norm_shape;
	norm_shape.push_back(N_);
	norm_shape.push_back(K_);
	x_norm_.Reshape(norm_shape);

	log_det_.Reshape(vector<int>(1, N_));
}

template <typename Dtype>
void DppLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* x_data = bottom[0]->cpu_data();
	Dtype* x_norm_data = x_norm_.mutable_cpu_data();
	for (int n = 0; n < N_; ++n)
	for (int k = 0; k < K_; ++k) {
		int x_norm_idx = n*K_ + k;
		int x_idx = (k*N_ + n)*D_;
		x_norm_data[x_norm_idx] = 0;
		for (int d = 0; d < D_; ++d)
			x_norm_data[x_norm_idx] += pow(x_data[x_idx++], 2);
		x_norm_data[x_norm_idx] = sqrt(x_norm_data[x_norm_idx]);
	}
	Dtype* correlate_data = correlate_matrix_.mutable_cpu_data();
	for (int n = 0; n < N_; ++n)
	for (int k1 = 0; k1 < K_; ++k1)
	for (int k2 = k1; k2 < K_; ++k2) {
		int x_idx1 = (k1*N_ + n)*D_;
		int x_idx2 = (k2*N_ + n)*D_;
		Dtype correlate_var = 0;
		for (int d = 0; d < D_; ++d)
			correlate_var += x_data[x_idx1++] * x_data[x_idx2++];
		correlate_var /= x_norm_data[n*K_ + k1];
		correlate_var /= x_norm_data[n*K_ + k2];
		correlate_data[(n*K_ + k1)*K_ + k2] = correlate_var;
		correlate_data[(n*K_ + k2)*K_ + k1] = correlate_var;
	}
	Dtype* correlate_diff = correlate_matrix_.mutable_cpu_diff();
	Dtype* log_det_data = log_det_.mutable_cpu_data();
	caffe_cpu_matrix_inverse<Dtype>(N_, K_, correlate_data, correlate_diff, log_det_data);
	Dtype loss = 0;
	for (int n = 0; n < N_; ++n)
		loss -= log_det_data[n];
	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
void DppLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0])
		return;
	const Dtype* correlate_diff = correlate_matrix_.cpu_diff();
	const Dtype* correlate_data = correlate_matrix_.cpu_data();
	const Dtype* x_data = bottom[0]->cpu_data();
	const Dtype* x_norm_data = x_norm_.cpu_data();
	Dtype* x_diff = bottom[0]->mutable_cpu_diff();
	for (int k = 0; k < K_; ++k)
	for (int n = 0; n < N_; ++n)
	for (int d = 0; d < D_; ++d) {
		int x_norm_idx = n * K_ + k;
		int x_idx = (k * N_ + n) * D_ + d;
		x_diff[x_idx] = 0;
		Dtype xd_norm = x_data[x_idx] / pow(x_norm_data[x_norm_idx], 2);
		for (int k_prime = 0; k_prime < K_; ++k_prime) {
			int c_idx = (n*K_ + k)*K_ + k_prime;
			int x_norm_prime_idx = n * K_ + k_prime;
			int x_prime_idx = (k_prime * N_ + n) * D_ + d;
			x_diff[x_idx] += (x_data[x_prime_idx] / x_norm_data[x_norm_idx] / x_norm_data[x_norm_prime_idx]
				- xd_norm * correlate_data[c_idx]) * correlate_diff[c_idx];
		}
		x_diff[x_idx] *= 2;
	}
	caffe_cpu_scale<Dtype>(bottom[0]->count(), -top[0]->cpu_diff()[0] / Dtype(N_), x_diff, x_diff);
}

#ifdef CPU_ONLY
STUB_GPU(DppLossLayer);
#endif

INSTANTIATE_CLASS(DppLossLayer);
REGISTER_LAYER_CLASS(DppLoss);

}  // namespace caffe
