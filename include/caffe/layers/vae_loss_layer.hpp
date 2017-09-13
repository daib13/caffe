#ifndef CAFFE_VAE_LOSS_LAYER_HPP_
#define CAFFE_VAE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class VAELossLayer : public LossLayer<Dtype> {
 public:
  explicit VAELossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "VAELoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MaxNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  VAELossParameter_VarianceType variance_type_;
  int N_, D_;
  Blob<Dtype> diff_, dist_;
  Dtype epsilon_, mean_l2_;
  Dtype alpha_;
};

}  // namespace caffe

#endif  // CAFFE_VAE_LOSS_LAYER_HPP_
