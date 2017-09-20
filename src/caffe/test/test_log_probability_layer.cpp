#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/log_probability_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class LogProbabilityLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  LogProbabilityLayerTest()
      : blob_bottom1_(new Blob<Dtype>(2, 3, 4, 1)),
	  blob_bottom2_(new Blob<Dtype>(3, 4, 1, 1)),
	  blob_bottom3_(new Blob<Dtype>(2, 4, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
	filler_param.set_min(0.01);
	filler_param.set_max(0.99);
	UniformFiller<Dtype> filler(filler_param);
	filler.Fill(this->blob_bottom1_);
	filler.Fill(this->blob_bottom2_);
	filler.Fill(this->blob_bottom3_);
	blob_bottom_vec_.push_back(blob_bottom1_);
	blob_bottom_vec_.push_back(blob_bottom2_);

    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~LogProbabilityLayerTest() {
    delete blob_bottom1_;
	delete blob_bottom2_;
	delete blob_bottom3_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_bottom3_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LogProbabilityLayerTest, TestDtypesAndDevices);

TYPED_TEST(LogProbabilityLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
	layer_param.add_propagate_down(true);
	layer_param.add_propagate_down(true);
	blob_bottom_vec_.clear();
	blob_bottom_vec_.push_back(blob_bottom1_);
	blob_bottom_vec_.push_back(blob_bottom2_);
	layer_param.mutable_log_probability_param()->set_distance_type(LogProbabilityParameter_DistanceType_BERNOULLI);
    LogProbabilityLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 5e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

TYPED_TEST(LogProbabilityLayerTest, TestGradient2) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
		sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		layer_param.add_propagate_down(true);
		layer_param.add_propagate_down(true);
		blob_bottom_vec_.clear();
		blob_bottom_vec_.push_back(blob_bottom1_);
		blob_bottom_vec_.push_back(blob_bottom2_);
		layer_param.mutable_log_probability_param()->set_distance_type(LogProbabilityParameter_DistanceType_GAUSSIAN);
		layer_param.mutable_log_probability_param()->set_sigma(1.2);
		LogProbabilityLayer<Dtype> layer(layer_param);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom1_);
		filler.Fill(this->blob_bottom2_);
		GradientChecker<Dtype> checker(1e-2, 5e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
	else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

TYPED_TEST(LogProbabilityLayerTest, TestGradient3) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
		sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		layer_param.add_propagate_down(true);
		layer_param.add_propagate_down(true);
		blob_bottom_vec_.clear();
		blob_bottom_vec_.push_back(blob_bottom1_);
		blob_bottom_vec_.push_back(blob_bottom2_);
		layer_param.mutable_log_probability_param()->set_distance_type(LogProbabilityParameter_DistanceType_OPT_GAUSSIAN);
		layer_param.mutable_log_probability_param()->set_epsilon(0.02);
		LogProbabilityLayer<Dtype> layer(layer_param);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom1_);
		filler.Fill(this->blob_bottom2_);
		GradientChecker<Dtype> checker(1e-2, 5e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
	else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

TYPED_TEST(LogProbabilityLayerTest, TestGradient4) {
	typedef typename TypeParam::Dtype Dtype;
	bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
	IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
	if (Caffe::mode() == Caffe::CPU ||
		sizeof(Dtype) == 4 || IS_VALID_CUDA) {
		LayerParameter layer_param;
		layer_param.add_propagate_down(true);
		layer_param.add_propagate_down(true);
		layer_param.add_propagate_down(true);
		blob_bottom_vec_.clear();
		blob_bottom_vec_.push_back(blob_bottom1_);
		blob_bottom_vec_.push_back(blob_bottom2_);
		blob_bottom_vec_.push_back(blob_bottom3_);
		layer_param.mutable_log_probability_param()->set_distance_type(LogProbabilityParameter_DistanceType_GAUSSIAN);
		layer_param.mutable_log_probability_param()->set_sigma(1.2);
		LogProbabilityLayer<Dtype> layer(layer_param);
		FillerParameter filler_param;
		GaussianFiller<Dtype> filler(filler_param);
		filler.Fill(this->blob_bottom1_);
		filler.Fill(this->blob_bottom2_);
		GradientChecker<Dtype> checker(1e-2, 5e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_);
	}
	else {
		LOG(ERROR) << "Skipping test due to old architecture.";
	}
}

}  // namespace caffe
