#include <vector>

#include "caffe/layers/kl_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void KLLossForward(const int nthreads, const Dtype* mu1, const Dtype* var1,
	const Dtype* mu2, const Dtype* var2, Dtype* loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		loss[index] = var1[index] / (var2[index] + 1e-6)
			+ pow(mu1[index] - mu2[index], 2) / (var2[index] + 1e-6)
			+ log((var2[index] + 1e-6) / (var1[index] + 1e-6))
			- 1;
	}
}

template <typename Dtype>
void KLLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	const Dtype* mu1 = bottom[0]->gpu_data();
	const Dtype* var1 = bottom[1]->gpu_data();
	const Dtype* mu2 = mu_.gpu_data();
	const Dtype* var2 = var_.gpu_data();

	Dtype* loss_data = bottom[0]->mutable_gpu_diff();
	const int count = N_*D_;
	KLLossForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, mu1, var1,
		mu2, var2, loss_data);
	Dtype loss;
	caffe_gpu_asum<Dtype>(count, loss_data, &loss);

	top[0]->mutable_cpu_data()[0] = loss / N_ / 2;
}

template <typename Dtype>
__global__ void KLLossBackward(const int nthreads, const int N, const Dtype loss_weight,
	const Dtype* mu1, const Dtype* var1, const Dtype* mu2, const Dtype* var2,
	Dtype* mu1_diff, Dtype* var1_diff, Dtype* mu2_diff, Dtype* var2_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		mu1_diff[index] = (mu1[index] - mu2[index]) / (var2[index] + 1e-6) / N * loss_weight;
		var1_diff[index] = (Dtype(1) / (var2[index] + 1e-6) - Dtype(1) / (var1[index] + 1e-6)) / 2 / N * loss_weight;
		mu2_diff[index] = -mu1_diff[index];
		var2_diff[index] = (var2[index] - var1[index] - pow(mu1[index] - mu2[index], 2)) / (pow(var2[index], 2) + 1e-6) / N / 2 * loss_weight;
	}
}

template <typename Dtype>
void KLLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* mu1 = bottom[0]->gpu_data();
	const Dtype* var1 = bottom[1]->gpu_data();
	Dtype* mu1_diff = bottom[0]->mutable_gpu_diff();
	Dtype* var1_diff = bottom[1]->mutable_gpu_diff();
	const Dtype* mu2 = mu_.gpu_data();
	const Dtype* var2 = var_.gpu_data();
	Dtype* mu2_diff = mu_.mutable_gpu_diff();
	Dtype* var2_diff = var_.mutable_gpu_diff();

	const int count = N_*D_;
	const Dtype loss_weight = top[0]->cpu_diff()[0];
	KLLossBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, N_, loss_weight, 
		mu1, var1, mu2, var2,
		mu1_diff, var1_diff, mu2_diff, var2_diff);

	if (bottom.size() == 4 && bottom[2]->count() == 0) {
		Dtype* mu2_sum_diff = bottom[2]->mutable_cpu_diff();
		Dtype* var2_sum_diff = bottom[3]->mutable_cpu_diff();
		mu2_sum_diff[0] = 0;
		var2_sum_diff[0] = 0;
		mu2_diff = mu_.mutable_cpu_diff();
		var2_diff = var_.mutable_cpu_diff();
		for (int i = 0; i < count; ++i) {
			mu2_sum_diff[0] += mu2_diff[i];
			var2_sum_diff[0] += var2_diff[i];
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(KLLossLayer);

}  // namespace caffe
