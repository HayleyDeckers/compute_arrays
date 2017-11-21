#pragma once
#include <vectorized_type.h>
namespace compute_arrays{
namespace functors{
struct Sum{
template<typename Lhs, typename Rhs>
  static auto Map(Lhs l, Rhs r){
    return l+r;
  }
};
struct Sub{
  template<typename Lhs, typename Rhs>
  static auto Map(Lhs l, Rhs r){
    return l-r;
  }
};
struct Mul{
template<typename Lhs, typename Rhs>
  static auto Map(Lhs l, Rhs r){
    return l*r;
  }
};
struct Div{
template<typename Lhs, typename Rhs>
  static auto Map(Lhs l, Rhs r){
    return l/r;
  }
};
//(possibly) vectorized mathetical operations, see <vectorized_types.h>
// for details.
struct Sqrt{
template<typename T>
  static auto Map(T val){
    return vec::sqrt(val);
  }
};
struct Log{
template<typename T>
  static auto Map(T val){
    return vec::log(val);
  }
};
struct Sin{
template<typename T>
  static auto Map(T val){
    return vec::sin(val);
  }
};
struct Cos{
template<typename T>
  static auto Map(T val){
    return vec::cos(val);
  }
};
struct Tan{
template<typename T>
  static auto Map(T val){
    return vec::tan(val);
  }
};
//min, max, clamp operators
template<typename T>
struct Max{
  inline static auto Map(T a, T b) {
    return a > b ? a : b;
  }
};
template<typename T>
struct Min{
  inline static auto Map(T a, T b) {
    return a < b ? a : b;
  }
};
struct Clamp{
  template<typename TVal>
  inline static TVal Map(TVal val, decltype(val) min, decltype(val) max) {
    assert(min <= max);
    return val < min? min : (val > max? max : val);
  }
};
}
}
