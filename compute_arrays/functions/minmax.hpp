#pragma once
#include "functors.hpp"
#include "apply_functor.hpp"
namespace compute_arrays{
template<typename TVal, typename TMin>
constexpr inline auto min(const Expression<TVal>& val, const TMin& min) {
  return applyFunctor<functors::Min>(val, makeExpression(min));
}
template<typename TVal, typename TMax>
constexpr inline auto max(const Expression<TVal>& val, const TMax& max) {
  return applyFunctor<functors::Max>(val, makeExpression(max));
}
template<typename TVal, typename TMin, typename TMax>
constexpr inline auto clamp(const Expression<TVal>& val, const TMin& min, const TMax& max) {
  return applyFunctor<functors::Clamp>(val, makeExpression(min), makeExpression(max));
}
}
