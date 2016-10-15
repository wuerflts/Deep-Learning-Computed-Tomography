
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/interpolator.hpp"

namespace caffe{

template< typename Dtype >
void FanBackprojectionLayer< Dtype >::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {

 const int detectorWidth = bottom[0]->shape()[3];

 this->detector_spacing = this->layer_param_.backprojection_param().detector_spacing();
 this->focal_length = this->layer_param_.backprojection_param().focal_length();

 this->ptDetectorX = 0;
 this->ptDetectorY = - static_cast< Dtype >( detectorWidth / 2. ) * this->detector_spacing;
 this->ptSourceX = this->focal_length;
 this->ptSourceY = 0;

 // just do an equal spacing of angles for now
 this->startAngle = ( this->layer_param_.backprojection_param().start_angle() * M_PI ) / 180;
 const double end_angle = ( this->layer_param_.backprojection_param().end_angle() * M_PI ) / 180;

 this->stepWidth = ( end_angle - this->startAngle ) / bottom[0]->shape()[2];
 for( int i = 0; i < bottom[0]->shape()[2]; ++i ) {
  double angle = -(this->startAngle + i * this->stepWidth);
  this->angles.push_back( angle );
 }
}

template< typename Dtype >
void FanBackprojectionLayer< Dtype >::Reshape(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 vector<int> top_shape;
 top_shape.push_back(bottom[0]->shape()[0]);
 top_shape.push_back(1);
 top_shape.push_back( bottom[0]->shape()[3] );
 top_shape.push_back( bottom[0]->shape()[3] );
 top[0]->Reshape( top_shape );
}

template< typename Dtype >
void FanBackprojectionLayer< Dtype >::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
 // get the information about the inputdata from the blobs
 const Dtype* bottom_data = bottom[0]->cpu_data();
 const int batchSize = bottom[0]->shape()[0];
 const int detectorWidth = bottom[0]->shape()[3];
 const int countOfProjections = bottom[0]->shape()[2];

 // get the information about the outputdata from the outputblob
 Dtype* top_data = top[0]->mutable_cpu_data();
 const int outputHeight = top[0]->shape()[3];
 const int outputWidth = top[0]->shape()[2];

 LinearInterpolator< Dtype > interpolator( bottom_data, SizeType( detectorWidth, countOfProjections ) );

 caffe_set(top[0]->count(), Dtype(0), top_data);

 for( int l = 0; l < batchSize; ++l )
 {
  const size_t batchOffsetOutput = l * outputWidth * outputHeight;
  const size_t batchOffsetInput = l * detectorWidth * countOfProjections;

  // for every projection
  for( int k = 0; k < countOfProjections; ++k ) {

   std::vector< Dtype > rotationMatrix;
   fillRotationMatrix( rotationMatrix, this->angles[k] );

   const int halfWidth = outputWidth / 2.;
   const int halfHeight = outputHeight /2.;

   // walk over every pixel of the output
   for (int i = 0; i < outputHeight; ++i) {
    for (int j = 0; j < outputWidth; ++j) {

     double position = backprojectPoint( GridPositionType(j - halfWidth,i - halfHeight) , rotationMatrix, PositionType(0 , -halfWidth ), detectorWidth );

     if( position < detectorWidth-1 && position > 0.00001 )
     {
      top_data[batchOffsetOutput + j+(outputWidth*i)] += interpolator.Interpolate( PositionType( position, k ), batchOffsetInput );
     }
    }
   }
  }
  double scalingFactor = M_PI / ( countOfProjections );
  for (int i = 0; i < outputHeight; ++i) {
   for (int j = 0; j < outputWidth; ++j) {
    top_data[batchOffsetOutput + j+(outputWidth*i)] *= scalingFactor;
   }
  }
 }
}

template< typename Dtype >
inline Dtype FanBackprojectionLayer< Dtype >::backprojectPoint( GridPositionType position, Dtype angle, PositionType detectorPosition,int detectorLength ) {

 std::vector< Dtype > rotationMatrix;
 fillRotationMatrix( rotationMatrix, angle );

 return backprojectPoint( position, rotationMatrix, detectorPosition , detectorLength );
}

template< typename Dtype >
inline Dtype FanBackprojectionLayer< Dtype >::backprojectPoint( GridPositionType position, const std::vector< Dtype >& rotationMatrix , PositionType detectorPosition,int detectorLength ) {

 PositionType direction, intersection;

 direction.first = detectorPosition.first * rotationMatrix[ 0 ] + detectorPosition.second * rotationMatrix[ 1 ];
 direction.second = detectorPosition.first * rotationMatrix[ 2 ] + detectorPosition.second * rotationMatrix[ 3 ];

 intersection.first =  ( position.first * direction.first + position.second * direction.second ) /  ( pow( direction.first, 2 ) + pow( direction.second, 2 ) ) * direction.first;
 intersection.second =  ( position.first * direction.first + position.second * direction.second ) /  ( pow( direction.first, 2 ) + pow( direction.second, 2 ) ) * direction.second;

 //TODO: this probably does not work for angles above 180
 if( intersection.first > direction.first )
 {
  return detectorLength;
 }

 return sqrt( pow( intersection.first - direction.first, 2 ) + pow( intersection.second - direction.second, 2 ) );
}

template< typename Dtype >
void FanBackprojectionLayer< Dtype >::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
 if (propagate_down[0]) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // for every pixel, calculate the weights and redistribute the diff backwards with their share
  // according to the weights

  // get the information about the inputdata from the blobs
  const int batchSize = bottom[0]->shape()[0];
  const int detectorWidth = bottom[0]->shape()[3];
  const int countOfProjections = bottom[0]->shape()[2];

  LinearInterpolator< Dtype > interpolator( bottom_diff, SizeType( detectorWidth, countOfProjections ) );

  // get the information about the outputdata from the outputblob
  const int outputHeight = top[0]->shape()[3];
  const int outputWidth = top[0]->shape()[2];

  const int halfWidth = outputWidth / 2.;
  const int halfHeight = outputHeight /2.;

 for( int l = 0; l < batchSize; ++l )
 {
  const size_t batchOffsetOutput = l * outputWidth * outputHeight;
  const size_t batchOffsetInput = l * detectorWidth * countOfProjections;

   // walk over every pixel of the output
   for (int i = 0; i < outputHeight; ++i) {
    for (int j = 0; j < outputWidth; ++j) {

     vector< PositionsAndWeights > positions;

     // find the corresponding pixels on the projections - this is wastefull because were doing the rotation alot !
     for( int k = 0; k < countOfProjections; ++k ) {

      double position = backprojectPoint( GridPositionType(j - halfWidth,i - halfHeight) , this->angles[k], PositionType( 0, - halfWidth ), detectorWidth );

      if( position < detectorWidth-1 && position > 0.00001 )
      {
       PositionsAndWeights current = interpolator.GetPositionsAndWeights( PositionType( position, k ) );
       positions.push_back( current );
      }
      else
      {
       PositionAndWeight dummy;
       dummy.position = 0;
       dummy.weight = 0;
       positions.push_back( PositionsAndWeights( dummy, dummy ) );
      }
     }

     for( int k = 0; k < countOfProjections; ++k ) {
      PositionsAndWeights current = positions[ k ];
      size_t depthIndex = k * detectorWidth + batchOffsetInput;

      Dtype currentError = top_diff[batchOffsetOutput + j+(outputWidth*i)];
      bottom_diff[ current.first.position + depthIndex ] += current.first.weight * currentError;
      bottom_diff[ current.second.position + depthIndex ] += current.second.weight * currentError;
     }
    }
   }
   Dtype scalingFactor = 1. / static_cast< Dtype >( outputHeight * outputWidth );
   for (int i = 0; i < countOfProjections; ++i) {
    for (int j = 0; j < detectorWidth; ++j) {
     bottom_diff[batchOffsetInput + j+(detectorWidth*i)] *= scalingFactor;
    }
   }
  }
 }
}

#ifdef CPU_ONLY
STUB_GPU(FanBackprojectionLayer);
#endif

INSTANTIATE_CLASS(FanBackprojectionLayer);

}  // namespace caffe
