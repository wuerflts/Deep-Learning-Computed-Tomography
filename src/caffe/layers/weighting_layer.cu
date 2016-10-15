#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

 const Dtype* bottom_data = bottom[0]->gpu_data();
 Dtype* top_data = top[0]->mutable_gpu_data();
 const Dtype* weights = this->blobs_[0]->gpu_data();

 const size_t layerSize = top[0]->shape()[2] * top[0]->shape()[3];

 for( size_t i = 0; i <  top[0]->shape()[0]; ++i)
 {
  caffe_gpu_mul( layerSize, ( bottom_data + i * layerSize ) , weights, ( top_data + i * layerSize ) );
 }
}

template <typename Dtype>
void WeightingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
 if (this->param_propagate_down_[0]) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* weightUpdate = this->blobs_[0]->mutable_gpu_diff();

   caffe_gpu_mul( top[0]->count(), top_diff, bottom_data, weightUpdate );
  }
  if (propagate_down[0]) {
   const Dtype* top_diff = top[0]->gpu_diff();
   const Dtype* weights = this->blobs_[0]->gpu_data();
   Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

   const size_t layerSize = top[0]->shape()[2] * top[0]->shape()[3];

   for( size_t i = 0; i <  top[0]->shape()[0]; ++i)
   {
    caffe_gpu_mul( layerSize, ( top_diff + i * layerSize ), weights, ( bottom_diff + i * layerSize ) );
   }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightingLayer);

}  // namespace caffe
