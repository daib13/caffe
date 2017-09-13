#include <vector>

#include "caffe/layers/vae_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void VAELossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	LossLayer::LayerSetUp(bottom, top);
	variance_type_ = this->layer_param_.vae_loss_param().type();
	epsilon_ = this->layer_param_.vae_loss_param().epsilon();
	if (epsilon_ > 0)
		alpha_ = 0;
	else {
		alpha_ = -epsilon_;
		this->blobs_.resize(1);
		vector<int> weight_shape(0);
		this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
		this->blobs_[0]->mutable_cpu_data()[0] = 1;
	}
//	CHECK_GT(epsilon_, 0) << "Epsilon of VAE loss must be larger than 1.";
}

template <typename Dtype>
void VAELossLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
	top[0]->Reshape(loss_shape);
	CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << "Inputs must have the same num.";
	CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1)) << "Inputs must have the same dimension.";
	diff_.ReshapeLike(*bottom[0]);
	N_ = bottom[0]->shape(0);
	D_ = bottom[0]->count(1);

	// if input variance, check the dimension of variance blob
	// The shape of variance blob should be
	// 1) pixel wise case: N * D
	// 2) item wise case: N
	if (bottom.size() > 2) {
		CHECK_EQ(bottom[2]->shape(0), N_) << "The num of sigma should equal to the input.";
		if (variance_type_ == VAELossParameter_VarianceType_PIXEL)
			CHECK_EQ(bottom[2]->count(1), D_) << "The dim of sigma should either be the dim of input in the pixelwise case.";
		else
			CHECK_EQ(bottom[2]->count(1), 1) << "The dim of sigma should be 1 in itemwise case.";
	}

	// reshape dist in the item wise case
	if (variance_type_ == VAELossParameter_VarianceType_ITEM) {
		vector<int> dist_shape(1, N_);
		dist_.Reshape(dist_shape);
	}
}

template <typename Dtype>
void VAELossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  // diff = x - x_hat
  int count = bottom[0]->count();
  caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());
  if (alpha_ > 0) {
	  Dtype l2_dis = caffe_cpu_dot<Dtype>(diff_.count(), diff_.cpu_data(), diff_.cpu_data());
	  l2_dis /= diff_.count();
	  mean_l2_ = this->blobs_[0]->cpu_data()[0];
	  mean_l2_ = 0.99 * mean_l2_ + 0.01 * l2_dis;
	  epsilon_ = alpha_ * mean_l2_;
	  this->blobs_[0]->mutable_cpu_data()[0] = mean_l2_;
  }

  // calculate dist in the item wise case
  // dist = \sum_d pow(diff[d], 2)
  const Dtype* diff_data = diff_.cpu_data();
  if (variance_type_ == VAELossParameter_VarianceType_ITEM) {
	  Dtype* dist_data = dist_.mutable_cpu_data();
	  for (int n = 0; n < N_; ++n) {
		  dist_data[n] = 0;
		  int diff_idx = n*D_;
		  for (int d = 0; d < D_; ++d)
			  dist_data[n] += pow(diff_data[diff_idx++], 2);
	  }
  }

  Dtype loss = 0;
  // 1. if input variance
  // loss = log(2*pi) + log(variance) + (x-x_hat)**2/variance
  // to protect the log and divide operation: variance -> variance + epsilon
  if (bottom.size() == 3) {
	  const Dtype* variance_data = bottom[2]->cpu_data();
	  int denominator = (variance_type_ == VAELossParameter_VarianceType_ITEM) ? D_ : 1;
	  for (int i = 0; i < count; ++i) {
		  Dtype variance = variance_data[i / denominator];
		  loss += (log(variance + epsilon_) + pow(diff_data[i], 2) / (variance + epsilon_) + LOG_TWO_PI);
	  }
  }
  // 2. if no input variance
  else {
	  // 1) pixel wise case: loss = log(2*pi) + log(diff**2) + 1
	  //    to protect log operation: diff**2 -> diff**2 + epsilon
	  if (variance_type_ == VAELossParameter_VarianceType_PIXEL){
		  for (int i = 0; i < count; ++i)
			  loss += (log(pow(diff_data[i], 2) + epsilon_) + LOG_TWO_PI + 1);
	  }
	  // 2) item wise case: loss = log(2*pi) + log(dist) + 1/D
	  //    to protect log operation: dist -> dist + epsilon
	  else {
		  const Dtype* dist_data = dist_.cpu_data();
		  for (int n = 0; n < N_; ++n)
			  loss += (D_*LOG_TWO_PI + D_*log(dist_data[n] + epsilon_) + 1);
	  }
  }

  top[0]->mutable_cpu_data()[0] = loss / N_ / 2;
}

template <typename Dtype>
void VAELossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
	const Dtype* diff_data = diff_.cpu_data();
	const int count = bottom[0]->count();

	const Dtype loss_weight = top[0]->cpu_diff()[0];
	Dtype* bottom0_diff = bottom[0]->mutable_cpu_diff();
	Dtype* bottom1_diff = bottom[1]->mutable_cpu_diff();
	// 1. if input variance (variance -> variance + epsilon)
	if (bottom.size() == 3) {
		// backward to x and x_hat
		// dL/dx = (x-x_hat) / variance
		const Dtype* variance_data = bottom[2]->cpu_data();
		int denominator = (variance_type_ == VAELossParameter_VarianceType_ITEM) ? D_ : 1;
		for (int i = 0; i < count; ++i) {
			Dtype variance = variance_data[i / denominator];
			bottom0_diff[i] = diff_data[i] / (variance + epsilon_) / N_ * loss_weight;
			bottom1_diff[i] = -bottom0_diff[i];
		}

		// backward to variance
		// 1) pixel wise case
		//    dL/dvariance = 1/variance - (x-x_hat)**2/variance**2
		Dtype* variance_diff = bottom[2]->mutable_cpu_diff();
		if (variance_type_ == VAELossParameter_VarianceType_PIXEL) {
			for (int i = 0; i < count; ++i) {
				variance_diff[i] = Dtype(1) / (variance_data[i] + epsilon_);
				variance_diff[i] -= pow(diff_data[i] / (variance_data[i] + epsilon_), 2);
				variance_diff[i] /= (2 * N_ / loss_weight);
			}	
		}
		// 2) item wise case
		//    dL/dvariance = D/variance - dist/variance**2 
		else {
			const Dtype* dist_data = dist_.cpu_data();
			for (int n = 0; n < N_; ++n) {
				variance_diff[n] = -dist_data[n] / pow(variance_data[n] + epsilon_, 2);
				variance_diff[n] += Dtype(D_) / (variance_data[n] + epsilon_);
				variance_diff[n] /= (2 * N_ / loss_weight);
			}
		}
	}
	// 2. if no input variance (only need to backward to x and x_hat)
	else {
		// 1) pixel wise case (diff**2 -> diff**2 + epsilon)
		//    dL/dx = (x-x_hat)/(x-x_hat)**2
		if (variance_type_ == VAELossParameter_VarianceType_PIXEL) {
			for (int i = 0; i < count; ++i) {
				bottom0_diff[i] = diff_data[i] / (pow(diff_data[i], 2) + epsilon_) / N_ * loss_weight;
				bottom1_diff[i] = -bottom0_diff[i];
			}
		}
		// 2) item wise case (dist -> dist + epsilon)
		//    dL/dx = D*(x-x_hat)/dist
		else {
			const Dtype* dist_data = dist_.cpu_data();
			for (int i = 0; i < count; ++i) {
				bottom0_diff[i] = D_ * diff_data[i] / (dist_data[i / D_] + epsilon_) / N_ * loss_weight;
				bottom1_diff[i] = -bottom0_diff[i];
			}
		}
	}
}

#ifdef CPU_ONLY
STUB_GPU(VAELossLayer);
#endif

INSTANTIATE_CLASS(VAELossLayer);
REGISTER_LAYER_CLASS(VAELoss);

}  // namespace caffe
