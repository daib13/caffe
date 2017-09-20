#include <vector>

#include "caffe/layers/kl_subspace_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KLSubspaceLossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);
	CHECK_EQ(bottom[1]->shape(0), N_) << "The num of z and mu_z should be the same.";
	CHECK_EQ(bottom[1]->count(1), D_) << "The dim of z and mu_z should be the same.";
	CHECK_EQ(bottom[2]->shape(0), N_) << "The num of z and sd_z should be the same.";
	CHECK_EQ(bottom[2]->count(1), D_) << "The dim of z and sd_z should be the same.";
	K_ = bottom[3]->count();
	CHECK_EQ(bottom[4]->count(), 1) << "The count of noise_sd should be 1.";
	CHECK_EQ(D_%K_, 0) << "The cluster number should be an exact divion of the dim of z.";

	vector<int> logp_dim_shape;
	logp_dim_shape.push_back(N_);
	logp_dim_shape.push_back(D_);
	logp_dim_.Reshape(logp_dim_shape);

	vector<int> logp_shape;
	logp_shape.push_back(N_);
	logp_.Reshape(logp_shape);

	vector<int> logq_dim_shape;
	logq_dim_shape.push_back(N_);
	logq_dim_shape.push_back(K_);
	logq_dim_shape.push_back(D_);
	logq_dim_.Reshape(logq_dim_shape);

	vector<int> logq_shape;
	logq_shape.push_back(N_);
	logq_shape.push_back(K_);
	logq_.Reshape(logq_shape);

	max_logq_.Reshape(logp_shape);
	res_q_.Reshape(logq_shape);
	sum_res_q_.Reshape(logp_shape);

	item_loss_.Reshape(vector<int>(1, N_));

	vector<int> gt_sd_shape;
	gt_sd_shape.push_back(K_);
	gt_sd_shape.push_back(D_);
	gt_sd_.Reshape(gt_sd_shape);
}

template <typename Dtype>
void KLSubspaceLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	Dtype loss = 0;
	const Dtype* z_data = bottom[0]->cpu_data();
	const Dtype* mu_z_data = bottom[1]->cpu_data();
	const Dtype* sd_z_data = bottom[2]->cpu_data();
	Dtype* logp_dim_data = logp_dim_.mutable_cpu_data();
	Dtype* logp_data = logp_.mutable_cpu_data();
	int idx = 0;
	for (int n = 0; n < N_; ++n) {
		logp_data[n] = 0;
		for (int d = 0; d < D_; ++d) {
			const Dtype safe_sd = max(Dtype(1e-6), sd_z_data[idx]);
			logp_dim_data[idx] = -pow((z_data[idx] - mu_z_data[idx]) / safe_sd, 2) / Dtype(2)
				- LOG_TWO_PI / Dtype(2) - log(safe_sd);
			logp_data[n] += logp_dim_data[idx++];
		}
		loss += logp_data[n];
	}

	const Dtype noise_sd = bottom[4]->cpu_data()[0];
	Dtype* sd_gt = gt_sd_.mutable_cpu_data();
	caffe_set<Dtype>(K_*D_, Dtype(1), sd_gt);
	const int dim_per_cluster = D_ / K_;
	for (int d = 0; d < D_; ++d) {
		sd_gt[d / dim_per_cluster*D_ + d] += noise_sd;
	}

	const Dtype* prior_data = bottom[3]->cpu_data();
	Dtype* logq_dim_data = logq_dim_.mutable_cpu_data();
	Dtype* logq_data = logq_.mutable_cpu_data();
	Dtype* max_logq_data = max_logq_.mutable_cpu_data();
	Dtype* res_q_data = res_q_.mutable_cpu_data();
	Dtype* sum_res_q_data = sum_res_q_.mutable_cpu_data();
	idx = 0;
	for (int n = 0; n < N_; ++n) {
		int sd_idx = 0;
		max_logq_data[n] = -INT_MAX;
		for (int k = 0; k < K_; ++k) {
			int logq_idx = n*K_ + k;
			logq_data[logq_idx] = 0;
			for (int d = 0; d < D_; ++d) {
				logq_dim_data[idx] = -pow(z_data[n*D_ + d] / sd_gt[sd_idx], 2) / Dtype(2)
					- LOG_TWO_PI / Dtype(2) - log(sd_gt[sd_idx]);
				++sd_idx;
				logq_data[logq_idx] += logq_dim_data[idx++];
			}
			logq_data[logq_idx] += log(max(Dtype(1e-6), prior_data[k]));
			max_logq_data[n] = max(max_logq_data[n], logq_data[logq_idx]);
		}
		loss -= max_logq_data[n];
		sum_res_q_data[n] = 0;
		for (int k = 0; k < K_; ++k) {
			int logq_idx = n*K_ + k;
			res_q_data[logq_idx] = exp(logq_data[logq_idx] - max_logq_data[n]);
			sum_res_q_data[n] += res_q_data[logq_idx];
		}
		loss -= log(sum_res_q_data[n]);
	}

	top[0]->mutable_cpu_data()[0] = loss / N_;
}

template <typename Dtype>
void KLSubspaceLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype scale = top[0]->mutable_cpu_diff()[0] / Dtype(N_);

	const Dtype* res_q_data = res_q_.cpu_data();
	const Dtype* sum_res_q_data = sum_res_q_.cpu_data();
	Dtype* logq_diff = logq_.mutable_cpu_diff();
	for (int n = 0; n < N_; ++n)
	for (int k = 0; k < K_; ++k)
		logq_diff[n*K_ + k] = -res_q_data[n*K_ + k] / sum_res_q_data[n] * scale;

	Dtype* logp_diff = logp_.mutable_cpu_diff();
	caffe_set<Dtype>(N_, scale, logp_diff);

	const Dtype* z_data = bottom[0]->cpu_data();
	const Dtype* mu_z_data = bottom[1]->cpu_data();
	const Dtype* sd_z_data = bottom[2]->cpu_data();
	const Dtype* sd_gt = gt_sd_.cpu_data();
	const Dtype* prior_data = bottom[3]->cpu_data();
	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_cpu_diff();
		int idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int d = 0; d < D_; ++d) {
			z_diff[idx] = 0;
			z_diff[idx] += (mu_z_data[idx] - z_data[idx]) / pow(max(Dtype(1e-6), sd_z_data[idx]), 2) * logp_diff[n];
			for (int k = 0; k < K_; ++k)
				z_diff[idx] -= z_data[idx] / pow(sd_gt[k*D_ + d], 2) * logq_diff[n*K_ + k];
			++idx;
		}
	}
	if (propagate_down[1]) {
		Dtype* mu_z_diff = bottom[1]->mutable_cpu_diff();
		int idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int d = 0; d < D_; ++d) {
			mu_z_diff[idx] = (z_data[idx] - mu_z_data[idx]) / pow(max(Dtype(1e-6), sd_z_data[idx]), 2) * logp_diff[n];
			++idx;
		}
	}
	if (propagate_down[2]) {
		Dtype* sd_z_diff = bottom[2]->mutable_cpu_diff();
		int idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int d = 0; d < D_; ++d) {
			const Dtype safe_sd = max(Dtype(1e-6), sd_z_data[idx]);
			sd_z_diff[idx] = (pow((z_data[idx] - mu_z_data[idx]) / safe_sd, 2) - 1) / safe_sd * logp_diff[n];
			++idx;
		}
	} 
	if (propagate_down[3]) {
		Dtype* prior_diff = bottom[3]->mutable_cpu_diff();
		caffe_set<Dtype>(K_, Dtype(0), prior_diff);
		int idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int k = 0; k < K_; ++k)
			prior_diff[k] += logq_diff[idx++];
		for (int k = 0; k < K_; ++k)
			prior_diff[k] /= max(Dtype(1e-6), prior_data[k]);
	}
	if (propagate_down[4]) {
		Dtype* sd_gt_diff = gt_sd_.mutable_cpu_diff();
		for (int k = 0; k < K_; ++k)
		for (int d = 0; d < D_; ++d) {
			int sd_gt_idx = k*D_ + d;
			sd_gt_diff[sd_gt_idx] = 0;
			for (int n = 0; n < N_; ++n) {
				sd_gt_diff[sd_gt_idx] += (pow(z_data[n*D_ + d] / sd_gt[sd_gt_idx], 2) - 1) / sd_gt[sd_gt_idx] * logq_diff[n*K_ + k];
			}
		}
		Dtype noise_sd_diff = 0;
		const int dim_per_cluster = D_ / K_;
		for (int d = 0; d < D_; ++d) {
			noise_sd_diff += sd_gt_diff[d / dim_per_cluster*D_ + d];
		}
		bottom[4]->mutable_cpu_diff()[0] = noise_sd_diff;
	}
}

#ifdef CPU_ONLY
STUB_GPU(KLSubspaceLossLayer);
#endif

INSTANTIATE_CLASS(KLSubspaceLossLayer);
REGISTER_LAYER_CLASS(KLSubspaceLoss);

}  // namespace caffe
