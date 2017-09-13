#ifndef CAFFE_DUPLICATE_LAYER_HPP_
#define CAFFE_DUPLICATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Prepare a blob for clustering useage
 * The original blob: N*D
 * The specified cluster number: K
 * The output blob: K*N*(K*D)
 * output(k1, n, k2, d) = 0 if k1 != k2
 * output(k1, n, k2, d) = input(n, d) if k1 == k2
 */
template <typename Dtype>
class DuplicateLayer : public Layer<Dtype> {
 public:
  explicit DuplicateLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Duplicate"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
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

  unsigned int N_, D_, K_;
};

}  // namespace caffe

#endif  // CAFFE_DUPLICATE_LAYER_HPP_
