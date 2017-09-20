#ifndef CAFFE_LOG_PROBABILITY_LAYER_HPP_
#define CAFFE_LOG_PROBABILITY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Produce the log probability of each sample given a reconstructed sample x_hat and groundtruth sample x
 * bottom[0]: x_hat: K*N*D
 * bottom[1]: x: N*D
 * top[0]: y: N*K (gives the log probability of p(x|x_hat) (notice that the order of N and K changes comparing to x_hat)
 *
 * Only for Bernoulli distribution (todo: Gaussian distribution, adaptive variance Gaussian distribution)
 */
template <typename Dtype>
class LogProbabilityLayer : public Layer<Dtype> {
 public:
  explicit LogProbabilityLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "LogProbability"; }
  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinNumBottomBlobs() const { return 2; }
  virtual inline int MaxNumBottomBlobs() const { return 3; }
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
  Blob<Dtype> x_duplicate_, p_dim_;
  shared_ptr<Layer<Dtype> > concat_layer_;
  vector<Blob<Dtype>*> concat_layer_bottom_;
  vector<Blob<Dtype>*> concat_layer_top_;
  Dtype sigma_, two_var_, log_sigma_, epsilon_;
  LogProbabilityParameter_DistanceType distance_type_;
};

}  // namespace caffe

#endif  // CAFFE_LOG_PROBABILITY_LAYER_HPP_
