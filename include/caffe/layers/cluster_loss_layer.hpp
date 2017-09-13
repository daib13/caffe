#ifndef CAFFE_CLUSTER_LOSS_LAYER_HPP_
#define CAFFE_CLUSTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Clustering loss @f$
 *          E = log \sum_y p(x|y)p(y)
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times K) @f$ log p(x|y) 
 *   -# @f$ (N) @f$ p(y)
 */
template <typename Dtype>
class ClusterLossLayer : public LossLayer<Dtype> {
 public:
  explicit ClusterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ClusterLoss"; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> log_joint_, log_joint_max_, joint_res_, item_loss_;
  int N_, K_;
};

}  // namespace caffe

#endif  // CAFFE_CLUSTER_LOSS_LAYER_HPP_
