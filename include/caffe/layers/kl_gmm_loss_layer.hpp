#ifndef CAFFE_KL_GMM_LOSS_LAYER_HPP_
#define CAFFE_KL_GMM_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class KLGMMLossLayer : public LossLayer<Dtype> {
public:
	explicit KLGMMLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param) {}
	virtual void LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "KLGMMLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}

	virtual inline int ExactNumBottomBlobs() const { return 6; }
	virtual inline int ExactNumTopBlobs() const { return -1; }
	virtual inline int MinNumTopBlobs() const { return 1; }
	virtual inline int MaxNumTopBlobs() const { return 2; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int N_, K_, D_;
	Blob<Dtype> logp_dim_, logp_;
	Blob<Dtype> logq_dim_, logq_, logq_max_, resq_, resq_sum_, item_loss_;
};

}  // namespace caffe

#endif  // CAFFE_KL_GMM_LOSS_LAYER_HPP_
