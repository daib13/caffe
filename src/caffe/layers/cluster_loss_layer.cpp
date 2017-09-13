#include <vector>

#include "caffe/layers/cluster_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ClusterLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	N_ = bottom[0]->shape(0);
	K_ = bottom[0]->count(1);
	CHECK_EQ(bottom[1]->count(), K_) << "The dim of prior and posterior should be the same.";
	vector<int> item_shape(1, N_);
	log_joint_max_.Reshape(item_shape);
	item_loss_.Reshape(item_shape);
	log_joint_.ReshapeLike(*bottom[0]);
	joint_res_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	const Dtype* log_posterior_data = bottom[0]->cpu_data();
	const Dtype* prior_data = bottom[1]->cpu_data();
	Dtype* log_joint_data = log_joint_.mutable_cpu_data();
	Dtype* log_joint_max_data = log_joint_max_.mutable_cpu_data();
	Dtype* joint_res_data = joint_res_.mutable_cpu_data();
	Dtype* item_loss_data = item_loss_.mutable_cpu_data();
	Dtype loss = 0;
	for (int n = 0; n < N_; ++n) {
		int index = n*K_;
		log_joint_max_data[n] = -INT_MAX;
		for (int k = 0; k < K_; ++k) {
			log_joint_data[index] = log_posterior_data[index] + log(max(Dtype(1e-12), prior_data[k]));
			log_joint_max_data[n] = max(log_joint_data[n], log_joint_data[index++]);
		}
		index = n*K_;
		item_loss_data[n] = 0;
		for (int k = 0; k < K_; ++k) {
			joint_res_data[index] = exp(log_joint_data[index] - log_joint_max_data[n]);
			item_loss_data[n] += joint_res_data[index++];
		}
		item_loss_data[n] = -log_joint_max_data[n] - log(item_loss_data[n]);
		loss += item_loss_data[n];
	}
	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
void ClusterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if ((!propagate_down[0]) && (!propagate_down[1]))
		return;

	const Dtype* joint_res_data = joint_res_.cpu_data();
	const Dtype scale = top[0]->cpu_diff()[0] / N_;
	Dtype* joint_res_sum_data = item_loss_.mutable_cpu_diff();
	Dtype* log_joint_diff = log_joint_.mutable_cpu_diff();
	for (int n = 0; n < N_; ++n) {
		int index = n*K_;
		joint_res_sum_data[n] = 0;
		for (int k = 0; k < K_; ++k)
			joint_res_sum_data[n] += joint_res_data[index++];
		index = n*K_;
		for (int k = 0; k < K_; ++k) {
			log_joint_diff[index] = -joint_res_data[index] / joint_res_sum_data[n] * scale;
			++index;
		}
	}
	if (propagate_down[0])
		caffe_copy<Dtype>(bottom[0]->count(), log_joint_diff, bottom[0]->mutable_cpu_diff());
	if (propagate_down[1]) {
		const Dtype* bottom1_data = bottom[1]->cpu_data();
		Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();
		caffe_set<Dtype>(K_, Dtype(0), bottom1_diff);
		const int count = N_*K_;
		for (int i = 0; i < count; ++i)
			bottom1_diff[i%K_] += log_joint_diff[i];
		for (int k = 0; k < K_; ++k)
			bottom1_diff[k] /= max(Dtype(1e-12), bottom1_data[k]);
	}
}

#ifdef CPU_ONLY
STUB_GPU(ClusterLossLayer);
#endif

INSTANTIATE_CLASS(ClusterLossLayer);
REGISTER_LAYER_CLASS(ClusterLoss);

}  // namespace caffe
