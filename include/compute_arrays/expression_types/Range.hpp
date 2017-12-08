#pragma once
#include "Expression.hpp"
#include "../internal/sequence.hpp"
namespace compute_arrays{
  /// An expression representing a numerical range of the form `Range[i] = start+i*delta`
  template<typename T>
  class Range : public Expression<Range<T>>{
    const T mStart;
    const T mDelta;
    const std::size_t mSize;
    template<int... Is>
    constexpr auto create_vector(int start, internal::seq<Is...> seq) const{
        //might be sub-optimal if Range is not constexpr
        return vec::vectorized_type<T>(std::initializer_list<T>({(T)(start+Is)...}).begin());
    }
  public:
    ///Construct a new range [start, end] with size nodes. If size == 1, then only one point will be included halfway between start and end.
    constexpr Range(T start, T end, std::size_t size) : mStart(start), mDelta(size > 1 ? (end-start)/(size-1) : (start+end)/2), mSize(size){}
    constexpr T operator[](std::size_t index) const {
      assert(index < mSize);
      return mStart + index*mDelta;
    }
    constexpr vec::vectorized_type<T> getVec(std::size_t index) const{
      assert(index + vec::vectorized_type<T>::Width -1 < mSize);
      vec::vectorized_type<T> offset =
          create_vector(
                        index,
                        typename internal::gens<vec::vectorized_type<T>::Width>::type()
                        );
      return vec::vectorized_type<T>(mStart)+ mDelta*offset;
    }
    template<typename I>
    constexpr auto gather(const Expression<I>& indices) const{
      return Gather<Range, I>(*this, indices);
    }
    /// Returns the size of the expression.
    constexpr std::size_t size() const{return mSize;}
    ///returns the stepsize
    constexpr T delta() const{return mDelta;}
    using Type = T;
  };
}
