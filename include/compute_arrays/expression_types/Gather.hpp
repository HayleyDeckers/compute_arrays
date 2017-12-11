#pragma once
#include "Expression.hpp"
namespace compute_arrays{
template<typename T>
class Slice;
/** Gather expressions take two expressions as input, a source and a set of indices.
 ** Upon evaluation at index `i` this expression returns `source[indices[i]]`.
 ** Both the source and indices can be any type of expression but
 ** the indices have to produce an integral type when evaluated.
 **
 ** Currently, vectorized access is limited to the case where the type of
 ** Source and Indices have the same native vector length.
*/
template<typename Source, typename Indices>
class Gather : public Expression<Gather<Source, Indices>>{
  const Source mSource;
  const Indices mIndices;
public:
  using Type = typename Source::Type;
  constexpr Gather(const Expression<Source> &source, const Expression<Indices>& indices) : mSource(inner(source)), mIndices(inner(indices)){
      static_assert(std::is_integral<typename Indices::Type>::value, "Gather(const Expression<T> t, const Expression<I> indices) requires that I produce an integral type");
  }
  constexpr Type operator[](std::size_t index) const {
    return mSource[mIndices[index]];
  }

  /// `get_vec` optimized for the situation where the `Source` derives from Slice
  /// In this case we can gather data directly from memory using the gather function
  /// defined in vec::vectorized_type.
  template<
  typename slice_t = Slice<typename Source::Type>,
  typename std::enable_if<
    std::is_base_of<slice_t, Source>::value
  && (vec::vectorized_type<Type>::Width <= vec::vectorized_type<typename Indices::Type>::Width)
   >::type* = nullptr
  >
  constexpr vec::vectorized_type<Type> getVec(std::size_t index) const{
    auto indices = mIndices.getVec(index).inner();
    return vec::vectorized_type<Type>::gather(mSource.data(), (const typename Indices::Type *)&indices);
  }
  /// `get_vec` optimized for the situation where the `Source` derives from Slice
  /// and the source has a wider vectorization than the index.
  /// In this case we can gather data directly from memory using the gather function
  /// defined in vec::vectorized_type.
  template<
  typename slice_t = Slice<typename Source::Type>,
  typename std::enable_if<
    std::is_base_of<slice_t, Source>::value
  && (vec::vectorized_type<Type>::Width > vec::vectorized_type<typename Indices::Type>::Width)
   >::type* = nullptr
  >
  constexpr vec::vectorized_type<Type> getVec(std::size_t index) const{
    constexpr int width = vec::vectorized_type<Type>::Width;
    typename Indices::Type indices[width];
    //TODO: optimize: vectorize index grabbing, specialize for the case where mIndices : Slice.
    for(int i = 0; i < width; i++){
      indices[i] = mIndices[index*width+i];
    }
    return vec::vectorized_type<Type>::Gather(mSource.data(), mIndices.getVec(index));;
  }
 /// `get_vec` for when the Source does not derinve from Slice (i.e. is not
 /// a plain array). In this case we have to evaluate the source at each index
 ///  in a scalar fashion.
  template<
    typename slice_t = Slice<typename Source::Type>,
    typename std::enable_if<
      !std::is_base_of<slice_t, Source>::value
      && (vec::vectorized_type<Type>::Width <= vec::vectorized_type<typename Indices::Type>::Width)
     >::type* = nullptr
  >
  constexpr  vec::vectorized_type<Type> getVec(std::size_t index) const{
    constexpr int width = vec::vectorized_type<Type>::Width;
    constexpr vec::vectorized_type<typename Indices::Type> indices = mIndices.getVec(index);
    constexpr vec::vectorized_type<typename Source::Type> ret;
    for(int i =0; i < width ; i++){
      ret.set(i, mSource[indices[i]]);
    }
    return ret;
  }
  /// `get_vec` for when the Source does not derinve from Slice (i.e. is not
  /// a plain array).
  /// and the source has a wider vectorization than the index.
  /// In this case we have to evaluate the source at each index
  ///  in a scalar fashion.
   template<
     typename slice_t = Slice<typename Source::Type>,
     typename std::enable_if<
       !std::is_base_of<slice_t, Source>::value
       && (vec::vectorized_type<Type>::Width > vec::vectorized_type<typename Indices::Type>::Width)
      >::type* = nullptr
   >
   constexpr  vec::vectorized_type<Type> getVec(std::size_t index) const{
     constexpr int width = vec::vectorized_type<Type>::Width;
     constexpr vec::vectorized_type<typename Source::Type> ret;
     for(int i =0; i < width ; i++){
       ret.set(i, mSource[mIndices[i]]);
     }
     return ret;
   }

  constexpr std::size_t size() const{return inner(mIndices).size();}
  ///gather's can be chained (but this should be avoided for performance reasons).
  template<typename I>
  constexpr auto gather(const Expression<I>& indices) const{
    return Gather<Source, decltype(std::declval<Indices>().gather(indices))>(mSource, mIndices.gather(indices));
  }
  constexpr const Source& source() const {return mSource;}
  constexpr const Indices& indices() const {return mIndices;}
};
}
