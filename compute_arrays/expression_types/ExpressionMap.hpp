#pragma once
#include "Expression.hpp"
#include "../internal/apply.hpp"
#include <tuple>
namespace compute_arrays{
  template<typename T>
  class Constant;
  /// An expression which performs an element-wise operation Op using arguments (Args...).
  /// If Args[i] is an expression it stores (and uses) a constant copy of this expression,
  /// if Args[i] is not an expression, an Constant<Args[i]> copy is made of it instead.
  /// these copies can stack up for deep structures but are O(1) when compared
  /// with number of elements.
  ///  ExpressionMap can take any number of arguments, but operations are always element-wise. Therefore,
  /// All arguments must be of equal length. Currently no checks are in place however.
  ///
  /// Note: copies are used instead of references to prevent temporary objects from being invalidated before
  /// they are evaluated.
  /// Note: Currently now size checks are in place.
  template<typename Op, typename... Args>
  class ExpressionMap : public Expression<ExpressionMap<Op, Args...>>{
    /// A tuple to hold the arguments.
    /// if it is not an expression yet it gets copied into a Constant<Arg> type.
    using tuple_type = std::tuple<typename
      std::conditional<
        std::is_base_of<TraitExpression, Args>::value,
        const Args,
        const Constant<Args>
      >::type...
    >;
    const tuple_type mArgs;

    //A helper function to gather, it creates a tuple consisting of the results of calling
    // `gather(indices)` on all members of mArgs.
    template<typename I, int... Is>
    constexpr auto gatherAll(const Expression<I>& indices, internal::seq<Is...>) const{
      return std::make_tuple(
        std::get<Is>(mArgs).gather(indices)...
      );
    }
    template<int... Is>
    constexpr auto gatherAllDep(internal::seq<Is...>) const{
      return std::tuple_cat(
        std::get<Is>(mArgs).getDep()...
      );
    }

  public:
    ///Construct and ExpressionMap from a parameter pack of arguments.
    /// None expression types specified in the template parameter pack `Args`
    /// should be given as `Constant<Args>`.
    constexpr ExpressionMap(typename
      std::conditional<
        std::is_base_of<TraitExpression, Args>::value,
        const Args,
        const Constant<Args>
      >::type... args) : mArgs(std::tie(args...)){};
    /// Construct an ExpressionMap directly from a tuple.
    constexpr ExpressionMap(const tuple_type tuple) : mArgs(tuple){};
    ///The resultant size of an ExpressionMap is the same as the size of
    /// of the first argument.
    constexpr std::size_t size() const{
      return std::get<0>(mArgs).size();
    }
    /// Evaluate the expression at index.
    ///
    ///see internal::invoke_at<Op>(std::size_t, const Tuple&)
    /// for implementation details.
    constexpr auto operator[](std::size_t index) const{
      return internal::invoke_at<Op>(index, mArgs);
    }
    /// Evaluate the expression as a vector expression at index.
    ///
    ///see internal::invoke_at_vec<Op>(std::size_t, const Tuple&)
    /// for implementation details.
    constexpr auto getVec(std::size_t index) const{
      return internal::invoke_at_vec<Op>(index, mArgs);
    }
    ///gather is special for an expression map. Say one wants to compute (a+b*c).gather([0,100,200]).
    /// if {a,b,c} are large arrays then computing a+b*c for all elements is an expensive operation.
    /// And all but three results end up being discared.
    /// The solution is that ExpressionMap::gather moves the gather instruction down to its arguments
    /// so it rewrites itself to a.gather([0,100,200])+b.gather([0,100,200])*c.gather([0,100,200]).
    /// if {a,b,c} is another ExpressionMap type this proccess is repeated.
    ///  If we look at the expression as a tree-like structure, this can be pictured as moving gather operations on a node down
    /// its children until they reach a leaf.
    template<typename I>
    constexpr auto gather(const Expression<I>& indices) const{
      return ExpressionMap<Op, const decltype(std::declval<Args>().gather(indices))...>(gatherAll(indices, typename internal::gens<sizeof...(Args)>::type()));
    }
    /// get a reference to one of the arguments.
    template<unsigned N>
    constexpr const auto& getArgs() const{
      return std::get<N>(mArgs);
    }
    constexpr auto getDep() const{
      return gatherAllDep(typename internal::gens<sizeof...(Args)>::type());
    }
    using Type = decltype(internal::invoke_at<Op>(0, mArgs));
  };
}
