
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
__device__ Dtype backprojectPointDevice( int j, int i, Dtype* rotationMatrix, Dtype detectorX, Dtype detectorY , int detectorLength )
{
  Dtype directionFirst = detectorX * rotationMatrix[ 0 ] + detectorY * rotationMatrix[ 1 ];
  Dtype directionSecond = detectorX * rotationMatrix[ 2 ] + detectorY * rotationMatrix[ 3 ];

  Dtype intersectionFirst =  ( j * directionFirst + i * directionSecond ) /  ( pow( directionFirst, 2 ) + pow( directionSecond, 2 ) ) * directionFirst;
  Dtype intersectionSecond =  ( j * directionFirst + i * directionSecond ) /  ( pow( directionFirst, 2 ) + pow( directionSecond, 2 ) ) * directionSecond;

  //TODO: this probably does not work for angles above 180
  if( intersectionFirst > directionFirst )
  {
   return detectorLength;
  }

  return sqrt( pow( intersectionFirst - directionFirst, 2 ) + pow( intersectionSecond - directionSecond, 2 ) );
}

template <typename Dtype>
__device__ Dtype interpolateDevice( const Dtype* in, Dtype position, size_t projection, size_t detectorLength, size_t batchOffsetInput )
{
 int firstPosition = static_cast< int >( position );
 int secondPosition = firstPosition + 1;

 // instead of calculating "1 - distance" we can just switch the two distances because they add up to 1
 Dtype firstWeight = -1 * ( position - static_cast< Dtype >( secondPosition ) );
 Dtype secondWeight = position - static_cast< Dtype >( firstPosition );

 size_t depthIndex = projection * detectorLength + batchOffsetInput;

 return firstWeight * in[ firstPosition + depthIndex ] + secondWeight * in[ secondPosition + depthIndex ];
}

template <typename Dtype>
__global__ void BackprojectionForward(const int count, const Dtype* in,
    const int halfOutputWidth, const int halfOutputHeight, const int outputWidth,
    const int outputHeight, const int detectorWidth, const int countOfProjections,
    const int outputSize,
    const Dtype scalingFactor,
    Dtype* rotmatrices,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {

   const int l = index / outputSize;
   const size_t batchOffsetInput = l * (detectorWidth * countOfProjections);
   const int interBatchIndex = index - l * outputSize;
   int i = interBatchIndex / outputWidth;
   int j = interBatchIndex - ( i * outputWidth );
   i -= halfOutputHeight;
   j -= halfOutputWidth;

   out[index] = 0;

   // for every projection
   #pragma unroll
   for( size_t k = 0; k < countOfProjections; ++k ) {

    Dtype position = backprojectPointDevice(j, i, rotmatrices + ( k * static_cast< size_t >( 4 ) ), static_cast< Dtype >( 0. ), -static_cast< Dtype >( halfOutputWidth ), detectorWidth );

    if( position < detectorWidth-1 && position > 0.00001 )
    {
     out[ index ] += interpolateDevice( in, position, k, detectorWidth, batchOffsetInput );
    }

   }

   out[ index ] *= scalingFactor;
  }
}

template <typename Dtype>
void BackprojectionLayer< Dtype >::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{
 const int count = top[0]->count();

 const int outputHeight = top[0]->shape()[3];
 const int outputWidth = top[0]->shape()[2];
 const int halfHeight = outputHeight /2.;
 const int halfWidth = outputWidth / 2.;

 const int detectorWidth = bottom[0]->shape()[3];
 const int countOfProjections = bottom[0]->shape()[2];
 const int outputSize = outputWidth * outputHeight;

 const Dtype scalingFactor = M_PI / ( countOfProjections );

 std::vector< Dtype > rotationMatrices;
 for( int k = 0; k < countOfProjections; ++k )
 {
     fillRotationMatrix( rotationMatrices, this->angles[k] );
 }

 // send matrices to device
 Dtype *deviceRotationMatrices;
 cudaMalloc( (void**) &deviceRotationMatrices, sizeof(Dtype) * rotationMatrices.size() );
 cudaMemcpy(deviceRotationMatrices, &rotationMatrices[0], sizeof(Dtype) * rotationMatrices.size(), cudaMemcpyHostToDevice);

 // calculate
 BackprojectionForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
           count,
           bottom[0]->gpu_data(),
           halfWidth,
           halfHeight,
           outputWidth,
           outputHeight,
           detectorWidth,
           countOfProjections,
           outputSize,
           scalingFactor,
           deviceRotationMatrices,
           top[0]->mutable_gpu_data()
           );

 CUDA_POST_KERNEL_CHECK;

}

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ float atomicAdd(float* address, float val)
{
 return __fAtomicAdd(address, val);
}

template <typename Dtype>
__global__ void BackprojectionBackward(const int count, const Dtype* __restrict__ top,
  const int halfTopWidth, const int halfTopHeight, const int topWidth,
  const int topHeight, const int detectorWidth, const int countOfProjections,
  const int topSize,
  Dtype* __restrict__ rotmatrices,
  Dtype* __restrict__ bottom ){

  CUDA_KERNEL_LOOP(index, count) {

   const int l = index / topSize;

   const size_t batchOffsetBottom = l * (detectorWidth * countOfProjections);
   const size_t batchOffsetTop = l  * topSize;

   const int interBatchIndex = index - batchOffsetTop;
   int i = interBatchIndex / topWidth;
   int j = interBatchIndex - ( i * topWidth );
   i -= halfTopHeight;
   j -= halfTopWidth;

   Dtype currentError = top[ index ];
   size_t depthIndex = batchOffsetBottom;

   // for every projection
   #pragma unroll
   for( size_t k = 0; k < countOfProjections; ++k ) {

    // determine the weights and the input positions
    Dtype position = backprojectPointDevice(j, i, rotmatrices + ( k * static_cast< size_t >( 4 ) ), static_cast< Dtype >( 0. ), -static_cast< Dtype >( halfTopWidth ), detectorWidth );

    if( position < detectorWidth-1 && position > 0.00001 )
    {
     int firstPosition = static_cast< int >( position );
     int secondPosition = firstPosition + 1;

     // instead of calculating "1 - distance" we can just switch the two distances because they add up to 1
     Dtype firstWeight = -1 * ( position - static_cast< Dtype >( secondPosition ) );
     Dtype secondWeight = position - static_cast< Dtype >( firstPosition );

     Dtype* currentFloorPosition = bottom + firstPosition + depthIndex;
     Dtype value = firstWeight * currentError;
     atomicAdd( currentFloorPosition, value );

     Dtype* currentCeilPosition = currentFloorPosition + static_cast< size_t >( 1 );
     value = secondWeight * currentError;
     atomicAdd( currentCeilPosition, value );
    }

    depthIndex += detectorWidth;
   }
  }
}


template <typename Dtype>
void BackprojectionLayer< Dtype >::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
 const int count = top[0]->count();

 const int topHeight = top[0]->shape()[3];
 const int topWidth = top[0]->shape()[2];
 const int halfTopHeight = topHeight /2.;
 const int halfTopWidth = topWidth / 2.;

 const int detectorWidth = bottom[0]->shape()[3];
 const int countOfProjections = bottom[0]->shape()[2];
 const int topSize = topWidth * topHeight;

 std::vector< Dtype > rotationMatrices;
 for( int k = 0; k < countOfProjections; ++k )
 {
     fillRotationMatrix( rotationMatrices, this->angles[k] );
 }

 // send matrices to device
 Dtype *deviceRotationMatrices;
 cudaMalloc( (void**) &deviceRotationMatrices, sizeof(Dtype) * rotationMatrices.size() );
 cudaMemcpy(deviceRotationMatrices, &rotationMatrices[0], sizeof(Dtype) * rotationMatrices.size(), cudaMemcpyHostToDevice);

 BackprojectionBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
           count,
           top[0]->gpu_diff(),
           halfTopWidth,
           halfTopHeight,
           topWidth,
           topHeight,
           detectorWidth,
           countOfProjections,
           topSize,
           deviceRotationMatrices,
           bottom[0]->mutable_gpu_diff()
           );

 CUDA_POST_KERNEL_CHECK;

 Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

 Dtype scalingFactor = 1. / static_cast< Dtype >( topSize );
 const int batchSize = bottom[0]->shape()[0];

 for( int l = 0; l < batchSize; ++l )
 {
  const size_t batchOffsetBottom = l * detectorWidth * countOfProjections;
  for (int i = 0; i < countOfProjections; ++i) {
   for (int j = 0; j < detectorWidth; ++j) {
    bottom_diff[batchOffsetBottom + j+(detectorWidth*i)] *= scalingFactor;
   }
  }
 }
}

INSTANTIATE_LAYER_GPU_FUNCS(BackprojectionLayer);

}  // namespace caffe
