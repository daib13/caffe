#ifndef CAFFE_KL_SUBSPACE_LOSS_LAYER_HPP_
#define CAFFE_KL_SUBSPACE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class KLSubspaceLossLayer : public LossLayer<Dtype> {
 public:
  explicit KLSubspaceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "KLLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

  virtual inline int ExactNumBottomBlobs() const { return 5; }
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

  int N_, K_, D_;
  Blob<Dtype> logp_dim_, logp_;
  Blob<Dtype> logq_dim_, logq_, max_logq_, res_q_, sum_res_q_;
  Blob<Dtype> gt_sd_;
  Blob<Dtype> item_loss_;
};

}  // namespace caffe

#endif  // CAFFE_KL_SUBSPACE_LOSS_LAYER_HPP_
