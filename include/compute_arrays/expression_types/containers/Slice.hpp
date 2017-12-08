#pragma once
#include "../Expression.hpp"
namespace compute_arrays{
  /// A Slice is a _borrowed_ array wrapper that functions
  /// as an expression. Care has to be taken that the underlying array outlives
  /// the Slice (and that the owner eventually frees it). Varray is the data-owning
  /// alternative and can be used in any situation where a Slice could be used.
  ///  The underlying pointer has to be aligned w.r.t. vec::vectorized_type<T>.
  template<typename T>
  class Slice : public Expression<Slice<T>> {
  protected:
    using vec_t = vec::vectorized_type<T>;
    T* mData;
    std::size_t mSize;
  public:

    using Type = T;
    ///The default constructor creates a Slice of size 0, pointing to null.
    constexpr Slice() : mData(nullptr), mSize(0){}
    /// Construct a slice from a pointer and size.
    /// The pointer *has to* have the alignment of vec::vectorized_type<T>.
    constexpr Slice(T* data, std::size_t size) : mData(data), mSize(size){}
    ///Virtual dtor, does nothing because a slice does not own its data.
    virtual ~Slice(){}
    /// how many scalar elements are in this Slice.
    constexpr std::size_t size() const{ return mSize; }
    /// How many vector elements are in this slice.
    ///
    /// If for example size() is 17, and vec::vectorized_type<T>::Width is 4,
    /// The vectorSize is = floor(17/4) = 4 and the tailSize() = 1.
    constexpr std::size_t vecSize() const{ return mSize/vec_t::Width; }
    /// How many vector scalar elements trail the data, not fitting in a vector
    /// element.
    ///
    /// If for example size() is 17, and vec::vectorized_type<T>::Width is 4,
    /// The vectorSize is = floor(17/4) = 4 and the tailSize() = 1.
    constexpr std::size_t tailSize() const{ return mSize%vec_t::Width;}
    /// The scalar index of the first element that can not be mapped to a vector
    /// element.
    constexpr std::size_t tailStartIndex() const{ return vecSize()*vec_t::Width;}

    constexpr const T* data() const {return mData;}

    constexpr T operator[](std::size_t index) const {return mData[index];}
    vec_t getVec(std::size_t index) const{
      return vec_t(mData+index);
    }

    template<typename I>
    constexpr auto gather(const Expression<I>& indices) const{
      return Gather<Slice<T>, I>(*this, indices);
    }
  };
}
