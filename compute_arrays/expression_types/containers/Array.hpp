#pragma once
#include <memory>
#include "Slice.hpp"
namespace compute_arrays{
/// A container which _owns_ the underlying data, similar to a std::vector,
/// and can be used as an expression. Can safely and cheaply be copied around.
template<typename T>
class Array : public Slice<T>{
protected:
  using vec_t = vec::vectorized_type<T>;
  std::shared_ptr<vec_t> mManagedData;
public:
  using Type = T;
  ///Construct a Array from an Expression where there the expression produces a type E::type != T but which can
  /// be converted into a T.
  ///
  /// Uses OpenMp to fill the array in parallel.
  template<typename E, typename std::enable_if<!std::is_same<typename E::Type, T>::value && std::is_convertible<typename E::Type, T>::value>::type* = nullptr>
  constexpr Array(const Expression<E> &v) {
    constexpr const E& subexpr = inner(v);
    this->mSize = subexpr.size();
    //Allocate an owned array for our data. rounding the allocated size up to a multiple of the SIMD width.
    mManagedData = std::shared_ptr<vec_t>(new vec_t[this->tailSize()? this->vecSize() + 1 : this->vecSize()], std::default_delete<vec_t[]>());
    this->mData = (T*)mManagedData.get();
    #pragma omp parallel for
    for(int index = 0;index < this->size(); index++){
      this->data()[index] = subexpr[index];
    }
  }
  ///Construct a Array from an Expression where there the expression produces a type E::type == T.
  ///
  /// Uses OpenMp to fill the array in parallel.
template<typename E, typename std::enable_if<std::is_same<typename E::Type, T>::value>::type* = nullptr>
  constexpr Array(const Expression<E> &v) {
    const E& subexpr = inner(v);
    this->mSize = subexpr.size();
    //Allocate an owned array for our data. rounding the allocated size up to a multiple of the SIMD width.
    mManagedData = std::shared_ptr<vec_t>(new vec_t[this->tailSize()? this->vecSize() + 1 : this->vecSize()], std::default_delete<vec_t[]>());
    this->mData = (T*)mManagedData.get();
    #pragma omp parallel for
    for(int index=0; index < this->vecSize(); index++){
      mManagedData.get()[index] = subexpr.get_vec(index);
    }
    for(int index = this->tailStartIndex();index < this->size(); index++){
      this->mData[index] = subexpr[index];
    }
  }
  ///Create an empty Array of given size.
constexpr Array(size_t size){
  this->mSize = size;
  mManagedData = std::shared_ptr<vec_t>(new vec_t[this->vecSize()+1], std::default_delete<vec_t[]>());
  this->mData = (T*)mManagedData.get();
}
  ///virtual dtor, does nothing because the std::shared_ptr handles data management.
  virtual ~Array(){}
  constexpr T& operator[](size_t index){return this->mData[index];}
  constexpr vec_t& getVec(size_t index){
    return mManagedData.get()[index];
  }
  constexpr vec_t getVec(size_t index) const{
    return mManagedData.get()[index];
  }
};
}
