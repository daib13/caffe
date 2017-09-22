#include <vector>

#include "caffe/layers/kl_gmm_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void KLGMMLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void KLGMMLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


}

INSTANTIATE_LAYER_GPU_FUNCS(KLGMMLossLayer);

}  // namespace caffe
