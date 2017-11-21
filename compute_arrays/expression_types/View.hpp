#pragma once
#include "Expression.hpp"
namespace compute_arrays{
  /// An expresion which is a subsection of another expression.
  template<typename E>
  class View : public Expression<View<E>>{
    const std::size_t mStart;
    const std::size_t mSize;
    const E mExpr;
  public:
    /// Construct a View from a start and end index.
    constexpr View(const Expression<E>& e, std::size_t start, std::size_t end) : mStart(start), mSize(end-start+1), mExpr(inner(e)){
      assert(start <= end);
    }
    constexpr auto operator[](std::size_t index) const{
      assert(index < size());
      return mExpr[mStart+index];
    }
    constexpr auto getVec(std::size_t index) const{
      assert(index < size());
      return mExpr.getVec(mStart + index);
    }
    constexpr auto size() const{
      return mSize;
    }
  };
}
