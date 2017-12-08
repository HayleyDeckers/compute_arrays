#pragma once
#include "apply_functor.hpp"
#include "functors.hpp"
namespace compute_arrays{

namespace reduce{
/// Reduce an expression using Op.
///
/// Automatically vectorized with OpenMP.
template<typename Op, typename E>
inline auto reduce(const Expression<E>& expr){
  using namespace vec;
  const E& subexpr = inner(expr);
  vectorized_type<typename E::Type> v_collector((typename E::Type)0);
  #pragma omp declare reduction(+:vectorized_type<typename E::Type>: omp_out = Op::Map(omp_out,omp_in) ) initializer(omp_priv = (typename E::Type)0)
  #pragma omp parallel for reduction(+:v_collector) schedule(static)
  for(int i = 0; i < subexpr.size()/decltype(v_collector)::Width; i++){
    v_collector = Op::Map(v_collector,subexpr.getVec(i));
  }
  typename E::Type scalar = v_collector[0];
  for(int i=1; i < decltype(v_collector)::Width;i++){
    scalar = Op::Map(scalar, v_collector[i]);
  }
  return scalar;
}

///Sums all the elements of expr, folding it into a scalar
/// while using vectorized operations internally.
///
/// Automatically vectorized with OpenMP.
template<typename TExpr>
constexpr inline auto sum(const Expression<TExpr>& expr){
  return reduce<functors::Sum>(expr);
}

///multiplies all the elements of expr, folding it into a scalar
/// while using vectorized operations internally.
///
/// Automatically vectorized with OpenMP.
template<typename TExpr>
constexpr inline auto multiply(const Expression<TExpr>& expr){
  return reduce<functors::Mul>(expr);
}

/// Finds the max element
///
/// Automatically vectorized with OpenMP.
template<typename TExpr>
constexpr inline auto max(const Expression<TExpr>& expr){
  return reduce<functors::Max>(expr);
}

///Finds the min element
///
/// Automatically vectorized with OpenMP.
template<typename TExpr>
constexpr inline auto min(const Expression<TExpr>& expr){
  return reduce<functors::Min>(expr);
}

}
}
