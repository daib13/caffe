#include <vector>

#include "caffe/layers/kl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	CHECK_EQ(bottom[0]->count(), bottom[1]->count())
		<< "Inputs must have the same dimension.";
	if (bottom.size() > 2) {
		CHECK_EQ(bottom.size(), 4) << "Reference mu and sd must be specified simultanuously.";
		if (bottom[2]->count() != 1)
			CHECK_EQ(bottom[2]->count(), bottom[0]->count()) << "Reference mu must either be a scalar or have the same count as input mu.";
		if (bottom[3]->count() != 1)
			CHECK_EQ(bottom[3]->count(), bottom[1]->count()) << "Reference var must either be a scalar or have the same count as input var.";
	}
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);

	vector<int> distribution_shape;
	distribution_shape.push_back(N_);
	distribution_shape.push_back(D_);
	mu_.Reshape(distribution_shape);
	var_.Reshape(distribution_shape);

	const int count = N_*D_;
	if (bottom.size() == 2) {
		caffe_set<Dtype>(count, this->layer_param().klloss_param().mu(), mu_.mutable_cpu_data());
		caffe_set<Dtype>(count, this->layer_param().klloss_param().var(), var_.mutable_cpu_data());
	}
	else {
		if (bottom[2]->count() == 1) {
			caffe_set<Dtype>(count, bottom[2]->cpu_data()[0], mu_.mutable_cpu_data());
			caffe_set<Dtype>(count, bottom[3]->cpu_data()[0], var_.mutable_cpu_data());
		}
		else {
			mu_.ShareData(*bottom[2]);
			mu_.ShareDiff(*bottom[2]);
			var_.ShareData(*bottom[3]);
			var_.ShareDiff(*bottom[3]);
		}
	}
}

template <typename Dtype>
void KLLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	
	const Dtype* mu1 = bottom[0]->cpu_data();
	const Dtype* var1 = bottom[1]->cpu_data();
	const Dtype* mu2 = mu_.cpu_data();
	const Dtype* var2 = var_.cpu_data();

	Dtype loss = 0;
	const int count = N_*D_;
	for (int i = 0; i < count; ++i) {
		loss = loss + var1[i] / (var2[i] + 1e-6)
			+ pow(mu1[i] - mu2[i], 2) / (var2[i] + 1e-6)
			+ log((var2[i] + 1e-6) / (var1[i] + 1e-6));
	}
	loss = (loss - count) / 2 / N_;

	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void KLLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* mu1 = bottom[0]->cpu_data();
	const Dtype* var1 = bottom[1]->cpu_data();
	Dtype* mu1_diff = bottom[0]->mutable_cpu_diff();
	Dtype* var1_diff = bottom[1]->mutable_cpu_diff();
	const Dtype* mu2 = mu_.cpu_data();
	const Dtype* var2 = var_.cpu_data();
	Dtype* mu2_diff = mu_.mutable_cpu_diff();
	Dtype* var2_diff = var_.mutable_cpu_diff();

	const Dtype loss_weight = top[0]->cpu_diff()[0];
	const int count = N_*D_;
	for (int i = 0; i < count; ++i) {
		mu1_diff[i] = (mu1[i] - mu2[i]) / (var2[i] + 1e-6) / N_ * loss_weight;
		var1_diff[i] = (Dtype(1) / (var2[i] + 1e-6) - Dtype(1) / (var1[i] + 1e-6)) / N_ / 2 * loss_weight;
		mu2_diff[i] = -mu1_diff[i];
		var2_diff[i] = (var2[i] - var1[i] + pow(mu1[i] - mu2[i], 2)) / (pow(var2[i], 2) + 1e-6) / N_ / 2 * loss_weight;
	}

	if (bottom.size() == 4 && bottom[2]->count() == 0) {
		Dtype* mu2_sum_diff = bottom[2]->mutable_cpu_diff();
		Dtype* var2_sum_diff = bottom[3]->mutable_cpu_diff();
		mu2_sum_diff[0] = 0;
		var2_sum_diff[0] = 0;
		for (int i = 0; i < count; ++i) {
			mu2_sum_diff[0] += mu2_diff[i];
			var2_sum_diff[0] += var2_diff[i];
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(KLLossLayer);
#endif

INSTANTIATE_CLASS(KLLossLayer);
REGISTER_LAYER_CLASS(KLLoss);

}  // namespace caffe
