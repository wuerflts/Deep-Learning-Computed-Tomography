#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 if (this->blobs_.size() > 0) {
      LOG(INFO) << "Skipping parameter initialization";
    } else {
   this->blobs_.resize(1);

   // Intialize the weight
   vector<int> weight_shape(1);
   weight_shape[0] = bottom[0]->shape()[2] * bottom[0]->shape()[3];

   this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

   // fill the weights
   shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
     this->layer_param_.weighting_param().weight_filler()));
   weight_filler->Fill(this->blobs_[0].get());

   this->param_propagate_down_.resize(this->blobs_.size(), true);
  }
}

template <typename Dtype>
void WeightingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 vector<int> top_shape = bottom[0]->shape();
 top[0]->Reshape(top_shape);
}

template <typename Dtype>
void WeightingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
 const Dtype* bottom_data = bottom[0]->cpu_data();
 Dtype* top_data = top[0]->mutable_cpu_data();
 const Dtype* weights = this->blobs_[0]->cpu_data();

 for( size_t i = 0; i <  top[0]->count(); ++i)
 {
  top_data[ i ] = bottom_data[ i ] * weights[ i % ( top[0]->shape()[2] * top[0]->shape()[3] ) ];
 }
}

template <typename Dtype>
void WeightingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
 if (this->param_propagate_down_[0]) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* weightUpdate = this->blobs_[0]->mutable_cpu_diff();

  for( size_t i = 0; i <  top[0]->count(); ++i)
  {
   weightUpdate[ i ] = top_diff[ i ] * bottom_data[ i ];
  }
 }
 if (propagate_down[0]) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* weights = this->blobs_[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  for( size_t i = 0; i < top[0]->count(); ++i)
  {
   bottom_diff[ i ] = top_diff[ i ] * weights[ i % ( top[0]->shape()[2] * top[0]->shape()[3] ) ];
  }
 }
}

#ifdef CPU_ONLY
STUB_GPU(WeightingLayer);
#endif

INSTANTIATE_CLASS(WeightingLayer);
REGISTER_LAYER_CLASS(Weighting);

}  // namespace caffe
