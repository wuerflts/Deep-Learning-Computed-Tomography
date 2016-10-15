// Interpolators are what their name says

#ifndef CAFFE_INTERPOLATOR_HPP
#define CAFFE_INTERPOLATOR_HPP

#include "caffe/blob.hpp"
#include "caffe/common.hpp"

namespace caffe {

struct PositionAndWeight
{
  int position;
  double weight;
};

typedef pair< PositionAndWeight, PositionAndWeight > PositionsAndWeights;

typedef pair< int, int > SizeType;
typedef pair< double, double > PositionType;

/// @brief Performs a concrete operation.
template <typename Dtype>
class LinearInterpolator{
 private:
  const SizeType _size;
  Dtype const* const _data;
 public:
  explicit LinearInterpolator( Dtype const* const data , const SizeType size ) :
  _size( size ), _data( data )
  {}

  Dtype Interpolate( const PositionType position, const size_t batchOffset )
  {
   PositionsAndWeights values = this->GetPositionsAndWeights( position );

   size_t depthIndex = static_cast< size_t >( position.second ) * this->_size.first + batchOffset;
   return values.first.weight * this->_data[ values.first.position + depthIndex ] +
     values.second.weight*this->_data[ values.second.position + depthIndex ];
  }

  PositionsAndWeights GetPositionsAndWeights( const PositionType position )
  {
   PositionAndWeight first;
   PositionAndWeight second;

   //first.position = std::max(0, std::min( ( this->size.first-1), static_cast< int >( position.first )));
   //second.position = std::min( static_cast< ptrdiff_t >(this->size.first-1), first.position+1);

   first.position = static_cast< int >( position.first );
   second.position = first.position + 1;

   // instead of calculating "1 - distance" we can just switch the two distances because they add up to 1
   first.weight = -1 * ( position.first - static_cast< double >( second.position ) );
   second.weight = position.first - static_cast< double >( first.position );

   return PositionsAndWeights( first, second );
  }
};

}  // namespace caffe

#endif  // CAFFE_INTERPOLATOR_HPP_
