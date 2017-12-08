#pragma once
#include <utility>

//these headers are not used here, but this is the base include for all specific
// types, which all use these. So we include them here instead of in every single
//type implementation.
#include <assert.h>
#include <vectorized_types.h>

namespace compute_arrays{
  //The "Gather" type is used by most other types, so we forward declare it here
  // instead of in each file individually.
  template<typename Source, typename Index>
  class Gather;

  /// An empty helper class. All Expression deriving classes have this trait
  /// allowing the use of `std::is_base_of<X, TraitExpression>` to check
  /// if something is an expression when using template metaprogramming.
  class TraitExpression{};

  /// All types know to the Expression type system must derive from
  /// Expresssion<E> using the curriously-recurring-template-parameter pattern.
  /// That is, define new types as `class someExpression : public Expression<someExpression>`.
  /// Contains some constexpr functions
  template<typename SubType>
  class Expression : public TraitExpression{
  // public:
  //   using Type = typename SubType::Type;
  //   constexpr std::size_t simd_width() const {
  //     return vec::vectorized_type<Type>::Width;
  //   }
  //   constexpr std::size_t vec_size() const {
  //     return size()/simd_width();
  //   }
  //   constexpr std::size_t tail_size() const {
  //     return size()%simd_width();
  //   }
  //   SubType& inner() {return GetInner(*this);}
  //   const SubType& inner() const {return GetInner(*this);}
  //   constexpr std::size_t size() const {return inner().size();}
  //   constexpr auto operator[](std::size_t index) const{return inner()[index];}
  //   constexpr auto get_vec(std::size_t index) const{return inner().get_vec(index);}
  //   template<typename I>
  //   constexpr auto gather(const Expression<I>& indices) const {return inner().gather(indices);}
  };
  /// a function for casting a generic Expression<E> to it's subtype E.
  template<typename SubType, typename std::enable_if<std::is_base_of<Expression<SubType>, SubType>::value>::type* = nullptr>
  constexpr inline const SubType& inner(const Expression<SubType>& expr){
    return *static_cast<const SubType*>(&expr);
  }
}
