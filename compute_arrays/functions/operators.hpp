#pragma once
#include "apply_functor.hpp"
#include "functors.hpp"
namespace compute_arrays{
//wrappers for functors
/*two expressions*/
template<typename TLhs, typename TRhs>
constexpr inline auto
operator*(const Expression<TLhs>& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Mul>(lhs, rhs);
}
template<typename TLhs, typename TRhs>
constexpr inline auto
operator/(const Expression<TLhs>& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Div>(lhs, rhs);
}
template<typename TLhs, typename TRhs>
constexpr inline auto
operator+(const Expression<TLhs>& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Sum>(lhs, rhs);
}
template<typename TLhs, typename TRhs>
constexpr inline auto
operator-(const Expression<TLhs>& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Sub>(lhs, rhs);
}
/*expression on right, non on left*/
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TRhs>::value>::type* = nullptr>
constexpr inline auto
operator*(const Expression<TLhs>& lhs, const TRhs& rhs) {
  return applyFunctor<functors::Mul>(lhs, makeExpression(rhs, inner(lhs).size()));
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TRhs>::value>::type* = nullptr>
constexpr inline auto
operator/(const Expression<TLhs>& lhs, const TRhs& rhs) {
  return applyFunctor<functors::Div>(lhs, makeExpression(rhs, inner(lhs).size()));
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TRhs>::value>::type* = nullptr>
constexpr inline auto
operator+(const Expression<TLhs>& lhs, const TRhs& rhs) {
  return applyFunctor<functors::Sum>(lhs, makeExpression(rhs, lhs.size()));
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TRhs>::value>::type* = nullptr>
constexpr inline auto
operator-(const Expression<TLhs>& lhs, const TRhs& rhs) {
  return applyFunctor<functors::Sub>(lhs, makeExpression(rhs, lhs.size()));
}
/*expression on the left, non on right*/
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TLhs>::value>::type* = nullptr>
constexpr inline auto
operator*(const TLhs& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Mul>(makeExpression(lhs,inner(rhs).size()), rhs);
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TLhs>::value>::type* = nullptr>
constexpr inline auto
operator/(const TLhs& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Div>(makeExpression(lhs,inner(rhs).size()), rhs);
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TLhs>::value>::type* = nullptr>
constexpr inline auto
operator+(const TLhs& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Sum>(makeExpression(lhs,rhs.size()), rhs);
}
template<typename TLhs, typename TRhs, typename std::enable_if<!std::is_base_of<TraitExpression, TLhs>::value>::type* = nullptr>
constexpr inline auto
operator-(const TLhs& lhs, const Expression<TRhs>& rhs) {
  return applyFunctor<functors::Sub>(makeExpression(lhs,rhs.size()), rhs);
}

}
