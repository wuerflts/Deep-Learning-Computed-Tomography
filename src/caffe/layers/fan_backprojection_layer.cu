
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
__device__ Dtype backprojectPointDevice( Dtype ptRecoX, Dtype ptRecoY, Dtype* rotationMatrix, Dtype ptDetX, Dtype ptDetY, Dtype ptSourceX, Dtype ptSourceY, Dtype detectorSpacing )
{
 // calculate the straight line from the detectorend to the origin

 // calculate the detectorend p2
 const Dtype ptRotatedDetX = ptDetX * rotationMatrix[ 0 ] + ptDetY * rotationMatrix[1];
 const Dtype ptRotatedDetY = ptDetX * rotationMatrix[ 2 ] + ptDetY * rotationMatrix[3];

 // calculate the direction d2
 const Dtype dirRotatedDetX = - ptRotatedDetX;
 const Dtype dirRotatedDetY = - ptRotatedDetY;

 // calculate the straight line from the source to the reconstruction point

 // calculate the source point p1
 const Dtype ptRotatedSourceX = ptSourceX * rotationMatrix[ 0 ] + ptSourceY * rotationMatrix[1];
 const Dtype ptRotatedSourceY = ptSourceX * rotationMatrix[ 2 ] + ptSourceY * rotationMatrix[3];

 // calculate the direction d1
 const Dtype dirRayX = ptRecoX - ptRotatedSourceX;
 const Dtype dirRayY = ptRecoY - ptRotatedSourceY ;

 // intersect the two lines A x = b,
 // where b = p2 - p1 and A = ( d1, d2 )

 // calculate the determinand of the matrix inversion
 const Dtype determinand = ( dirRayX * ptRotatedDetY ) - ( ptRotatedDetX * dirRayY );

 // if no intersection is possible return an invalid value
 if( std::abs( determinand ) < 1E-6 )
 {
  return -1;
 }

 // calculate A^-1 * b to receive the factor where the ray intersects the detector

// calculate b = p2 - p1
const Dtype bX = ptRotatedDetX - ptRotatedSourceX;
const Dtype bY = ptRotatedDetY - ptRotatedSourceY;

// use cramers rule and calculate the matrixproduct with b
const Dtype lamda1 = ( ptRotatedDetY * bX - ptRotatedDetX * bY ) / determinand;

// use the factor to calculate the actual point of intersection
const Dtype ptIntersectionX = lamda1 * dirRayX + ptRotatedSourceX;
const Dtype ptIntersectionY = lamda1 * dirRayY + ptRotatedSourceY;

// calculate the distance to the detectorend to be able to map to the 1d detector
const Dtype ptOnDetX = ptIntersectionX - ptRotatedDetX;
const Dtype ptOnDetY = ptIntersectionY - ptRotatedDetY;

 Dtype length = std::sqrt( std::pow( ptOnDetX , 2 ) + std::pow( ptOnDetY, 2 ) );

 //deal with truncation
 if( ( ptOnDetX * dirRotatedDetX + ptOnDetY * dirRotatedDetY ) < 0. )
 {
  length = 0.;
 }

  return ( length - 0.5 ) / detectorSpacing;
}

//TODO: unify interpolators
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
    const Dtype halfOutputWidth, const Dtype halfOutputHeight, const int outputWidth,
    const int outputHeight, const int detectorWidth, const int countOfProjections,
    const int outputSize,
    const Dtype scalingFactor,
    const Dtype halfDetectorLength,
    const Dtype focalLength,
    const Dtype detectorSpacing,
    const Dtype ptDetectorX,
    const Dtype ptDetectorY,
    const Dtype ptSourceX,
    const Dtype ptSourceY,
    const Dtype startAngle,
    const Dtype stepWidth,
    Dtype* rotmatrices,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, count) {

   const int l = index / outputSize;
   const size_t batchOffsetInput = l * (detectorWidth * countOfProjections);
   const int interBatchIndex = index - l * outputSize;
   int i = interBatchIndex / outputWidth;
   int j = interBatchIndex - ( i * outputWidth );
   Dtype reconstructionX = j - halfOutputWidth + 0.5;
   Dtype reconstructionY = i - halfOutputHeight + 0.5;

   out[index] = 0;

   // for every projection
   #pragma unroll
   for( size_t k = 0; k < countOfProjections; ++k ) {

     Dtype position = backprojectPointDevice(reconstructionX,
                                             reconstructionY,
                                             rotmatrices + ( k * static_cast< size_t >( 4 ) ),
                                             ptDetectorX,
                                             ptDetectorY,
                                             ptSourceX,
                                             ptSourceY,
                                             detectorSpacing );

    if( position < detectorWidth-1 && position > 0.00001 )
    {
     Dtype radius = std::sqrt( std::pow( reconstructionX, 2 ) + std::pow( reconstructionY, 2 ) );
     Dtype phi = M_PI / 2. + std::atan2( reconstructionY, reconstructionX);
     Dtype beta = startAngle + k * stepWidth;
     Dtype distanceWeight = std::pow( ( focalLength + radius * std::sin( beta - phi ) ) / focalLength , 2 );

     out[ index ] += interpolateDevice( in, position, k, detectorWidth, batchOffsetInput ) / distanceWeight;
    }

   }

   out[ index ] *= scalingFactor;
  }
}

template <typename Dtype>
void FanBackprojectionLayer< Dtype >::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top)
{

 const int count = top[0]->count();

 const int outputHeight = top[0]->shape()[3];
 const int outputWidth = top[0]->shape()[2];
 const Dtype halfHeight = outputHeight /2.;
 const Dtype halfWidth = outputWidth / 2.;

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
           static_cast< Dtype >( detectorWidth / 2. ) * this->detector_spacing,
           this->focal_length,
           this->detector_spacing,
           this->ptDetectorX,
           this->ptDetectorY,
           this->ptSourceX,
           this->ptSourceY,
           this->startAngle,
           this->stepWidth,
           deviceRotationMatrices,
           top[0]->mutable_gpu_data()
           );

 CUDA_POST_KERNEL_CHECK;

}

__device__ double fanAtomicAdd(double* address, double val)
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

__device__ float fanAtomicAdd(float* address, float val)
{
 return __fAtomicAdd(address, val);
}

template <typename Dtype>
__global__ void BackprojectionBackward(const int count, const Dtype* __restrict__ top,
  const int halfTopWidth, const int halfTopHeight, const int topWidth,
  const int topHeight, const int detectorWidth, const int countOfProjections,
  const int topSize,
  const Dtype halfDetectorLength,
  const Dtype focalLength,
  const Dtype detectorSpacing,
  const Dtype ptDetectorX,
  const Dtype ptDetectorY,
  const Dtype ptSourceX,
  const Dtype ptSourceY,
  const Dtype startAngle,
  const Dtype stepWidth,
  Dtype* __restrict__ rotmatrices,
  Dtype* __restrict__ bottom ){

  //Dtype maxDifference;

  CUDA_KERNEL_LOOP(index, count) {

   const int l = index / topSize;

   const size_t batchOffsetBottom = l * (detectorWidth * countOfProjections);
   const size_t batchOffsetTop = l  * topSize;

   const int interBatchIndex = index - batchOffsetTop;
   int i = interBatchIndex / topWidth;
   int j = interBatchIndex - ( i * topWidth );
   Dtype reconstructionX = j - halfTopWidth + 0.5;
   Dtype reconstructionY = i - halfTopHeight + 0.5;

   Dtype currentError = top[ index ];
   size_t depthIndex = batchOffsetBottom;

   // for every projection
   #pragma unroll
   for( size_t k = 0; k < countOfProjections; ++k ) {

    // determine the weights and the input positions
    Dtype position = backprojectPointDevice(reconstructionX,
                                            reconstructionY,
                                            rotmatrices + ( k * static_cast< size_t >( 4 ) ),
                                            ptDetectorX,
                                            ptDetectorY,
                                            ptSourceX,
                                            ptSourceY,
                                            detectorSpacing );

    if( position < detectorWidth-1 && position > 0.00001 )
    {
     Dtype radius = std::sqrt( std::pow( reconstructionX, 2 ) + std::pow( reconstructionY, 2 ) );
     Dtype phi = M_PI / 2. + std::atan2( reconstructionY, reconstructionX);
     Dtype beta = startAngle + k * stepWidth;
     Dtype distanceWeight = std::pow( ( focalLength + radius * std::sin( beta - phi ) ) / focalLength , 2 );

     Dtype weightedCurrentError = currentError / distanceWeight;

     int firstPosition = static_cast< int >( position );
     int secondPosition = firstPosition + 1;

     // instead of calculating "1 - distance" we can just switch the two distances because they add up to 1
     Dtype firstWeight = -1 * ( position - static_cast< Dtype >( secondPosition ) );
     Dtype secondWeight = position - static_cast< Dtype >( firstPosition );

     Dtype* currentFloorPosition = bottom + firstPosition + depthIndex;
     Dtype value = firstWeight * weightedCurrentError;
     fanAtomicAdd( currentFloorPosition, value );

     Dtype* currentCeilPosition = currentFloorPosition + static_cast< size_t >( 1 );
     value = secondWeight * weightedCurrentError;
     fanAtomicAdd( currentCeilPosition, value );
    }

    depthIndex += detectorWidth;
   }
  }
}


template <typename Dtype>
void FanBackprojectionLayer< Dtype >::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
           static_cast< Dtype >( detectorWidth / 2. ) * this->detector_spacing,
           this->focal_length,
           this->detector_spacing,
           this->ptDetectorX,
           this->ptDetectorY,
           this->ptSourceX,
           this->ptSourceY,
           this->startAngle,
           this->stepWidth,
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

INSTANTIATE_LAYER_GPU_FUNCS(FanBackprojectionLayer);

}  // namespace caffe
