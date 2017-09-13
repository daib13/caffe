#include <vector>

#include "caffe/layers/duplicate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DuplicateForward(const int nthreads, const int K, const int N, const int D,
	const Dtype* bottom_data, Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int d = index % D;
		int n = index / D % N;
		int k = index / D / N;
		top_data[(k*N + n)*K*D + k*D + d] = bottom_data[n*D + d];
	}
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	caffe_gpu_set<Dtype>(top[0]->count(), Dtype(0), top_data);
	DuplicateForward<Dtype><<<CAFFE_GET_BLOCKS(K_*N_*D_), CAFFE_CUDA_NUM_THREADS>>>(K_*N_*D_, K_, N_, D_,
		bottom_data, top_data);
}

template <typename Dtype>
__global__ void DuplicateBackward(const int nthreads, const int K, const int N, const int D,
	const Dtype* top_diff, Dtype* bottom_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		bottom_diff[index] = 0;
		int n = index / D;
		int d = index % D;
		for (int k = 0; k < K; ++k)
			bottom_diff[index] += top_diff[(k*N + n)*K*D + k*D + d];
	}
}

template <typename Dtype>
void DuplicateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (!propagate_down[0]) { return; }
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	DuplicateBackward<Dtype><<<CAFFE_GET_BLOCKS(N_*D_), CAFFE_CUDA_NUM_THREADS>>>(N_*D_, K_, N_, D_,
		top_diff, bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(DuplicateLayer);

}  // namespace caffe
