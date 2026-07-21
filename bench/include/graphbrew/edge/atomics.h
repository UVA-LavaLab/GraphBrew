#ifndef GRAPHBREW_EDGE_ATOMICS_H_
#define GRAPHBREW_EDGE_ATOMICS_H_

#include <type_traits>

#include "platform_atomics.h"

namespace graphbrew::edge {

template <typename T>
T AtomicLoad(const T &target) {
  static_assert(std::is_integral<T>::value);
#ifdef _OPENMP
  return __atomic_load_n(&target, __ATOMIC_RELAXED);
#else
  return target;
#endif
}

template <typename T>
void AtomicStore(T &target, const T value) {
  static_assert(std::is_integral<T>::value);
#ifdef _OPENMP
  __atomic_store_n(&target, value, __ATOMIC_RELAXED);
#else
  target = value;
#endif
}

template <typename T>
bool AtomicMin(T &target, const T value) {
  static_assert(std::is_integral<T>::value);
  T observed = AtomicLoad(target);
  while (value < observed) {
    if (compare_and_swap(target, observed, value))
      return true;
    observed = AtomicLoad(target);
  }
  return false;
}

template <typename T>
bool AtomicMax(T &target, const T value) {
  static_assert(std::is_integral<T>::value);
  T observed = AtomicLoad(target);
  while (value > observed) {
    if (compare_and_swap(target, observed, value))
      return true;
    observed = AtomicLoad(target);
  }
  return false;
}

template <typename T>
bool AtomicAssignIfEqual(T &target, const T expected, const T value) {
  static_assert(std::is_integral<T>::value);
  return compare_and_swap(target, expected, value);
}

}  // namespace graphbrew::edge

#endif  // GRAPHBREW_EDGE_ATOMICS_H_
