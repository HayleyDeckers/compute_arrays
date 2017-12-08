#pragma once
#include "functors.hpp"
#include "apply_functor.hpp"
namespace compute_arrays{
//wrapper for mathematical functions
///Compute the elementwise sqrt of expr.
template<typename TVal>
constexpr inline auto sqrt(const Expression<TVal>& expr) {
  return applyFunctor<functors::Sqrt>(expr);
}
///Compute the elementwise log of expr.
template<typename TVal>
constexpr inline auto log(const Expression<TVal>& expr) {
  return applyFunctor<functors::Log>(expr);
}
///Compute the elementwise tan of expr.
template<typename TVal>
constexpr inline auto tan(const Expression<TVal>& expr) {
  return applyFunctor<functors::Tan>(expr);
}
///Compute the elementwise sin of expr.
template<typename TVal>
constexpr inline auto sin(const Expression<TVal>& expr) {
  return applyFunctor<functors::Sin>(expr);
}
///Compute the elementwise cos of expr.
template<typename TVal>
constexpr inline auto cos(const Expression<TVal>& expr) {
  return applyFunctor<functors::Cos>(expr);
}

///Compute the elementwise cos of expr.
template<typename TVal>
constexpr inline auto abs(const Expression<TVal>& expr) {
  return applyFunctor<functors::Abs>(expr);
}
}
