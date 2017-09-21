#ifndef CAFFE_LABEL_DATA_LAYER_HPP_
#define CAFFE_LABEL_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LabelDataLayer : public BasePrefetchingDataLayer<Dtype> {
public:
	explicit LabelDataLayer(const LayerParameter& param)
		: BasePrefetchingDataLayer<Dtype>(param) {}
	virtual ~LabelDataLayer();
	virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);

	virtual inline const char* type() const { return "LabelData"; }
	virtual inline int ExactNumBottomBlobs() const { return 0; }
	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
	virtual void load_batch(Batch<Dtype>* batch);
	vector<Dtype> label_;
	int label_id_;
};


}  // namespace caffe

#endif  // CAFFE_LABEL_DATA_LAYER_HPP_
