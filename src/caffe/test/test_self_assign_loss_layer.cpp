#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/self_assign_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

template <typename TypeParam>
class SelfAssignLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
 protected:
  SelfAssignLossLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 3, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
	GaussianFiller<Dtype> filler(filler_param);
	filler.Fill(this->blob_bottom_);
	blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SelfAssignLossLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(SelfAssignLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(SelfAssignLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
  IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
  if (Caffe::mode() == Caffe::CPU ||
      sizeof(Dtype) == 4 || IS_VALID_CUDA) {
    LayerParameter layer_param;
	layer_param.add_propagate_down(true);
	layer_param.add_loss_weight(1.2);
	layer_param.mutable_self_assign_loss_param()->set_lambda(0.1);
	layer_param.mutable_self_assign_loss_param()->mutable_weight_filler()->set_type("gaussian");
	layer_param.mutable_self_assign_loss_param()->mutable_weight_filler()->set_std(0.1);
    SelfAssignLossLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-2, 5e-3);
    checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test due to old architecture.";
  }
}

}  // namespace caffe
