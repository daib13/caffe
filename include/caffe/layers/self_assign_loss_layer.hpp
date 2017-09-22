#ifndef CAFFE_SELF_ASSIGN_LAYER_HPP_
#define CAFFE_SELF_ASSIGN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SelfAssignLossLayer : public LossLayer<Dtype> {
public:
	explicit SelfAssignLossLayer(const LayerParameter& param)
		: LossLayer<Dtype>(param) {}
	virtual void LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "SelfAssignLoss"; }
	virtual inline bool AllowForceBackward(const int bottom_index) const {
		return true;
	}
	virtual inline int ExactNumBottomBlobs() const { return 1; }

protected:
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	int N_, D_;
	Dtype lambda_;
	Blob<Dtype> z_hat_, w_sign_;
};

}  // namespace caffe

#endif  // CAFFE_SELF_ASSIGN_LOSS_LAYER_HPP_
