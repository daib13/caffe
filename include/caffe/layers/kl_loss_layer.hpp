#ifndef CAFFE_KL_LOSS_LAYER_HPP_
#define CAFFE_KL_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class KLLossLayer : public LossLayer<Dtype> {
 public:
  explicit KLLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KLLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MaxNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> mu_, var_;
  int N_, D_;
};

}  // namespace caffe

#endif  // CAFFE_KL_LOSS_LAYER_HPP_
