#pragma once
#include "sequence.hpp"
#include <tuple>


namespace compute_arrays{
namespace internal{

namespace detail{
  ///Calls F::Map with arguments given by the Tuple members, each indexed by index.
  template<typename F, typename Tuple, int ...S>
  constexpr inline auto callFunc(std::size_t index, const Tuple & t, seq<S...>)
  {
    return F::Map(std::get<S>(t)[index]...);
  }
}

///Calls F::Map with arguments given by the Tuple members, each indexed by index.
template <typename F, typename Tuple>
  constexpr inline auto invoke_at(std::size_t index, const Tuple &t)
  {
    return detail::callFunc<F>(index, t, typename gens<std::tuple_size<Tuple>::value>::type());
  }


  namespace detail{
    /// same as callFunc but uses vector indexing.
    template<typename F, typename Tuple, int ...S>
    constexpr inline auto callFunc_vec(std::size_t index, const Tuple & t, seq<S...>)
    {
      return F::Map(std::get<S>(t).get_vec(index)...);
    }
  }

  template <typename F, typename Tuple>
    /// same as invoke_at but uses vector indexing.
    constexpr inline auto invoke_at_vec(std::size_t index, const Tuple &t)
    {
      return detail::callFunc_vec<F>(index, t, typename gens<std::tuple_size<Tuple>::value>::type());
    }

}
}
