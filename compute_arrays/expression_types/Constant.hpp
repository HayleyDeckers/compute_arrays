#pragma once
#include "Expression.hpp"
namespace compute_arrays{
  /// An expression representing an array of constant values.
  template<typename T>
  class Constant : public Expression<Constant<T>>{
    const T mData; ///The value produced by the expression.
    const std::size_t mSize; ///The size of the expression, used for consistency checks.
  public:
    ///Construct a new consant from a value and size.
    constexpr Constant(T t, std::size_t size) : mData(t), mSize(size){}
    constexpr const T operator[](std::size_t index) const {
      assert(index < mSize);
      return mData;
    }
    constexpr vec::vectorized_type<T> getVec(std::size_t index) const{
      assert(index + vec::vectorized_type<T>::Width - 1 < mSize);
      return vec::vectorized_type<T>(mData);
    }
    /// A constant gathered by [i_0,i_1,...i_n] is simply
    /// a new Constant of length n.
    template<typename I>
    constexpr auto gather(const Expression<I>& indices) const{
      return Constant(mData, inner(indices).size());
    }
    /// Returns the size of the expression.
    constexpr std::size_t size() const{return mSize;}
    using Type = T;
  };

  ///Create a Constant<T> from an object of Type T.
  ///
  /// note: this function copies T and can not convert standard containers such as a vectors.
  template<typename T, typename std::enable_if<!std::is_base_of<Expression<T>, T>::value>::type* = nullptr>
  constexpr inline const Constant<T> makeExpression(const T& expr, std::size_t size){return Constant<T>(expr, size);}
  //specialization for the case where T is already an expression.
  template<typename T, typename std::enable_if<std::is_base_of<Expression<T>, T>::value>::type* = nullptr>
  constexpr inline const auto& makeExpression(const Expression<T>& expr, std::size_t size){return expr;}
}
