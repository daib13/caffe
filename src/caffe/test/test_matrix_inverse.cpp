#include <stdint.h>  // for uint32_t & uint64_t
#include <time.h>
#include <cmath>  // for std::fabs

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename TypeParam>
class MatrixInverseTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MatrixInverseTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()) {
  }

  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    this->blob_bottom_->Reshape(2, 5, 5, 1);
    this->blob_top_->Reshape(2, 5, 5, 1);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_top_);
  }

  virtual ~MatrixInverseTest() {
    delete blob_bottom_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
};

TYPED_TEST_CASE(MatrixInverseTest, TestDtypesAndDevices);

TYPED_TEST(MatrixInverseTest, TestGradient) {
	typedef typename TypeParam::Dtype Dtype;

	std::ofstream outfile("result.txt", std::ios::app);
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 5, 5, 5, 
		1, blob_bottom_->cpu_data(), blob_bottom_->cpu_data(), 
		0, blob_bottom_->mutable_cpu_diff());
	caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 5, 5, 5,
		1, blob_bottom_->cpu_data() + 25, blob_bottom_->cpu_data() + 25,
		0, blob_bottom_->mutable_cpu_diff() + 25);
	const Dtype* bottom_data = blob_bottom_->cpu_diff();
	for (int i = 0; i < 2; ++i){
		outfile << "[";
		for (int j = 0; j < 5; ++j) {
			outfile << "[";
			for (int k = 0; k < 5; ++k)
				outfile << bottom_data[i * 25 + j * 5 + k] << ", ";
			outfile << "]; ";
		}
		outfile << "]\n";
	}

	caffe_cpu_matrix_inverse<Dtype>(2, 5, bottom_data, blob_top_->mutable_cpu_data(), blob_top_->mutable_cpu_diff());
	const Dtype* inv_mat = blob_top_->cpu_data();
	const Dtype* det_mat = blob_top_->cpu_diff();
	for (int i = 0; i < 2; ++i) {
		outfile << "[";
		for (int j = 0; j < 5; ++j) {
			outfile << "[";
			for (int k = 0; k < 5; ++k)
				outfile << inv_mat[i * 25 + j * 5 + k] << ", ";
			outfile << "]; ";
		}
		outfile << "]\n" << det_mat[i] << '\n';
	}

	caffe_gpu_matrix_inverse<Dtype>(2, 5, blob_bottom_->gpu_diff(), blob_top_->mutable_gpu_data(), blob_top_->mutable_gpu_diff());
	inv_mat = blob_top_->cpu_data();
	det_mat = blob_top_->cpu_diff();
	for (int i = 0; i < 2; ++i) {
		outfile << "[";
		for (int j = 0; j < 5; ++j) {
			outfile << "[";
			for (int k = 0; k < 5; ++k)
				outfile << inv_mat[i * 25 + j * 5 + k] << ", ";
			outfile << "]; ";
		}
		outfile << "]\n" << det_mat[i] << '\n';
	}
	outfile.close();

}

}