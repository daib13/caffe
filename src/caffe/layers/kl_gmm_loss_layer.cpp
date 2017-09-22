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

	Dtype* loss_data = item_loss_.mutable_cpu_data();

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
			logp_dim_data[idx] = (-pow((z_data[idx] - mu_z_data[idx]) / safe_sd, 2) - LOG_TWO_PI) / Dtype(2) - log(safe_sd);
			logp_data[n] += logp_dim_data[idx++];
		}
		loss_data[n] = logp_data[n];
	}

	Dtype loss = 0;
	const Dtype* prior_data = bottom[3]->cpu_data();
	const Dtype* mu_c_data = bottom[4]->cpu_data();
	const Dtype* sd_c_data = bottom[5]->cpu_data();
	Dtype* logq_dim_data = logq_dim_.mutable_cpu_data();
	Dtype* logq_data = logq_.mutable_cpu_data();
	Dtype* logq_max_data = logq_max_.mutable_cpu_data();
	Dtype* resq_data = resq_.mutable_cpu_data();
	Dtype* resq_sum_data = resq_.mutable_cpu_data();
	for (int n = 0; n < N_; ++n) {
		logq_max_data[n] = -INT_MAX;
		for (int k = 0; k < K_; ++k) {
			int q_idx = n*K_ + k;
			logq_data[q_idx] = log(max(Dtype(1e-6), prior_data[k]));
			idx = q_idx*D_;
			for (int d = 0; d < D_; ++d) {
				int c_idx = k*D_ + d;
				const Dtype safe_sd = max(Dtype(1e-6), sd_c_data[c_idx]);
				logq_dim_data[idx] = (-pow((z_data[n*D_ + d] - mu_c_data[c_idx]) / safe_sd, 2) - LOG_TWO_PI) / Dtype(2) - log(safe_sd);
				logq_data[q_idx] += logq_dim_data[idx++];
			}
			logq_max_data[n] = max(logq_data[q_idx], logq_max_data[n]);
		}
		loss_data[n] -= logq_max_data[n];

		resq_sum_data[n] = Dtype(0);
		for (int k = 0; k < K_; ++k) {
			int q_idx = n*K_ + k;
			resq_data[q_idx] = exp(logq_data[q_idx] - logq_max_data[n]);
			resq_sum_data[n] += resq_data[q_idx];
		}
		loss_data[n] -= log(resq_sum_data[n]);
		loss += loss_data[n];
	}

	top[0]->mutable_cpu_data()[0] = loss / N_;
	Dtype* posterior_data = logq_.mutable_cpu_diff();
	for (int n = 0; n < N_; ++n)
	for (int k = 0; k < K_; ++k) {
		idx = n*K_ + k;
		posterior_data[idx] = resq_data[idx] / resq_sum_data[n];
	}
	if (top.size() == 2)
		caffe_copy<Dtype>(N_*K_, posterior_data, top[1]->mutable_cpu_data());
}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype scale = top[0]->cpu_diff()[0] / Dtype(N_);
	Dtype* logp_diff = logp_.mutable_cpu_diff();
	caffe_set<Dtype>(N_, scale, logp_diff);
	Dtype* logq_diff = logq_.mutable_cpu_diff();
	caffe_cpu_scale<Dtype>(N_*K_, -scale, logq_diff, logq_diff);

	const Dtype* z_data = bottom[0]->cpu_data();
	const Dtype* mu_z_data = bottom[1]->cpu_data();
	const Dtype* sd_z_data = bottom[2]->cpu_data();
	const Dtype* prior_data = bottom[3]->cpu_data();
	const Dtype* mu_c_data = bottom[4]->cpu_data();
	const Dtype* sd_c_data = bottom[5]->cpu_data();
	if (propagate_down[0]) {
		Dtype* z_diff = bottom[0]->mutable_cpu_diff();
		int z_idx = 0;
		for (int n = 0; n < N_; ++n) 
		for (int d = 0; d < D_; ++d) {
			Dtype safe_sd = max(Dtype(1e-6), sd_z_data[z_idx]);
			z_diff[z_idx] = logp_diff[n] * (mu_z_data[z_idx] - z_data[z_idx]) / pow(safe_sd, 2);
			for (int k = 0; k < K_; ++k) {
				int q_idx = n*K_ + k;
				int c_idx = k*D_ + d;
				safe_sd = max(Dtype(1e-6), sd_c_data[c_idx]);
				z_diff[z_idx] += logq_diff[q_idx] * (mu_c_data[c_idx] - z_data[z_idx]) / pow(safe_sd, 2);
			}
			++z_idx;
		}
	}
	if (propagate_down[1]) {
		Dtype* mu_z_diff = bottom[1]->mutable_cpu_diff();
		int z_idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int d = 0; d < D_; ++d) {
			Dtype safe_sd = max(Dtype(1e-6), sd_z_data[z_idx]);
			mu_z_diff[z_idx] = logp_diff[n] * (z_data[z_idx] - mu_z_data[z_idx]) / pow(safe_sd, 2);
			++z_idx;
		}
	}
	if (propagate_down[2]) {
		Dtype* sd_z_diff = bottom[2]->mutable_cpu_diff();
		int z_idx = 0;
		for (int n = 0; n < N_; ++n)
		for (int d = 0; d < D_; ++d) {
			Dtype safe_sd = max(Dtype(1e-6), sd_z_data[z_idx]);
			sd_z_diff[z_idx] = logp_diff[n] * (pow((mu_z_data[z_idx] - z_data[z_idx]) / safe_sd, 2) - 1) / safe_sd;
			++z_idx;
		}
	}
	if (propagate_down[3]) {
		Dtype* prior_diff = bottom[3]->mutable_cpu_diff();
		for (int k = 0; k < K_; ++k) {
			prior_diff[k] = 0;
			for (int n = 0; n < N_; ++n)
				prior_diff[k] += logq_diff[n*K_ + k];
			prior_diff[k] /= max(Dtype(1e-6), prior_data[k]);
		}
	}
	if (propagate_down[4]) {
		Dtype* mu_c_diff = bottom[4]->mutable_cpu_diff();
		int c_idx = 0;
		for (int k = 0; k < K_; ++k)
		for (int d = 0; d < D_; ++d) {
			mu_c_diff[c_idx] = Dtype(0);
			const Dtype safe_sd = max(Dtype(1e-6), sd_c_data[c_idx]);
			const Dtype safe_sd_square = pow(safe_sd, 2);
			for (int n = 0; n < N_; ++n) {
				mu_c_diff[c_idx] += logq_diff[n*K_ + k] * (z_data[n*D_ + d] - mu_c_data[c_idx]) / safe_sd_square;
			}
			++c_idx;
		}
	}
	if (propagate_down[5]) {
		Dtype* sd_c_diff = bottom[5]->mutable_cpu_diff();
		int c_idx = 0;
		for (int k = 0; k < K_; ++k)
		for (int d = 0; d < D_; ++d) {
			sd_c_diff[c_idx] = Dtype(0);
			const Dtype safe_sd = max(Dtype(1e-6), sd_c_data[c_idx]);
			for (int n = 0; n < N_; ++n) {
				sd_c_diff[c_idx] += logq_diff[n*K_ + k]
					* (pow((z_data[n*D_ + d] - mu_c_data[c_idx]) / safe_sd, 2) - 1) / safe_sd;
			}
			++c_idx;
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(KLGMMLossLayer);
#endif

INSTANTIATE_CLASS(KLGMMLossLayer);
REGISTER_LAYER_CLASS(KLGMMLoss);

}  // namespace caffe
