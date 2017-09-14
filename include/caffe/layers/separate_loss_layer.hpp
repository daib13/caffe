#ifndef CAFFE_SEPARATE_LOSS_LAYER_HPP_
#define CAFFE_SEPARATE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

/**
 * @brief Computes the Clustering loss @f$
 *          L = sum_{k_1 \neq k_2} max(0, delta - dis(x(k_1) - x(k_2)))
 *
 * @param bottom input Blob vector (length 1)
 *   -# @f$ (K \times N \times D) @f$ x 
 */
template <typename Dtype>
class SeparateLossLayer : public LossLayer<Dtype> {
 public:
  explicit SeparateLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SeparateLoss"; }
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
  
  Blob<Dtype> dis_, loss_;
  int N_, K_, D_;
  Dtype delta_;
};

}  // namespace caffe

#endif  // CAFFE_SEPARATE_LOSS_LAYER_HPP_
