#pragma once
#include "../expression_types/ExpressionMap.hpp"
namespace compute_arrays{
/// Creates an expression of type ExpressionMap<OP, Args...> representing
/// the result of calling functor OP with the element-wise arguments given
/// by the expressions Args...
template<typename OP, typename... Args>
constexpr inline auto
applyFunctor(const Expression<Args>&... args) {
  return ExpressionMap<OP, Args...>(inner(args)...);
}
}
