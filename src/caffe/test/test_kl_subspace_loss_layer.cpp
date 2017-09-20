#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/kl_subspace_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class KLSubspaceLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  KLSubspaceLossLayerTest()
      : blob_bottom1_(new Blob<Dtype>(2, 12, 1, 1)),
	  blob_bottom2_(new Blob<Dtype>(2, 12, 1, 1)),
	  blob_bottom3_(new Blob<Dtype>(2, 12, 1, 1)),
	  blob_bottom4_(new Blob<Dtype>(3, 1, 1, 1)),
	  blob_bottom5_(new Blob<Dtype>(1, 1, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
	filler_param.set_min(0.01);
	filler_param.set_max(0.99);
	UniformFiller<Dtype> filler1(filler_param);
	GaussianFiller<Dtype> filler2(filler_param);
	filler2.Fill(this->blob_bottom1_);
	filler2.Fill(this->blob_bottom2_);
	filler1.Fill(this->blob_bottom3_);
	blob_bottom4_->mutable_cpu_data()[0] = 0.3;
	blob_bottom4_->mutable_cpu_data()[1] = 0.2;
	blob_bottom4_->mutable_cpu_data()[2] = 0.5;
	blob_bottom5_->mutable_cpu_data()[0] = 1.2;
	blob_bottom_vec_.push_back(blob_bottom1_);
	blob_bottom_vec_.push_back(blob_bottom2_);
	blob_bottom_vec_.push_back(blob_bottom3_);
	blob_bottom_vec_.push_back(blob_bottom4_);
	blob_bottom_vec_.push_back(blob_bottom5_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~KLSubspaceLossLayerTest() {
    delete blob_bottom1_;
	delete blob_bottom2_;
	delete blob_bottom3_;
	delete blob_bottom4_;
	delete blob_bottom5_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom1_;
  Blob<Dtype>* const blob_bottom2_;
  Blob<Dtype>* const blob_bottom3_;
  Blob<Dtype>* const blob_bottom4_;
  Blob<Dtype>* const blob_bottom5_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(KLSubspaceLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(KLSubspaceLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
	layer_param.add_loss_weight(1.2);
    KLSubspaceLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 2e-2);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
