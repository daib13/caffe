#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/label_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
LabelDataLayer<Dtype>::~LabelDataLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void LabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	const string label_file = this->layer_param_.data_param().source();
	FILE* fin = fopen(label_file.c_str(), "rb");
	int N;
	fread(&N, sizeof(int), 1, fin);
	label_.resize(N);
	fread(label_.data(), sizeof(Dtype), N, fin);
	fclose(fin);

	vector<int> top_shape;
	top_shape.push_back(this->layer_param_.data_param().batch_size());
	top_shape.push_back(1);
	top[0]->Reshape(top_shape);
	label_id_ = 0;

	for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
		this->prefetch_[i].data_.Reshape(top_shape);
	}
}

// This function is called on prefetch thread
template <typename Dtype>
void LabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
	CPUTimer batch_timer;
	batch_timer.Start();
	CHECK(batch->data_.count());
	vector<int> top_shape;
	const int batch_size = this->layer_param_.data_param().batch_size();
	top_shape.push_back(this->layer_param_.data_param().batch_size());
	top_shape.push_back(1);
	batch->data_.Reshape(top_shape);

	Dtype* prefetch_data = batch->data_.mutable_cpu_data();

	// datum scales
	for (int item_id = 0; item_id < batch_size; ++item_id) {
		prefetch_data[item_id] = label_[label_id_++];
		if (label_id_ >= label_.size()) {
			// We have reached the end. Restart from the first.
			DLOG(INFO) << "Restarting data prefetching from start.";
			label_id_ = 0;
		}
	}
	batch_timer.Stop();
	DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
}

INSTANTIATE_CLASS(LabelDataLayer);
REGISTER_LAYER_CLASS(LabelData);

}  // namespace caffe
