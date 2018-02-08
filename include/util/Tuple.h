/*
 * Tuple.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_TUPLE_H_
#define INCLUDE_UTIL_TUPLE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <stddef.h>
#include <tgmath.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace wavepi {
namespace util {
using namespace dealii;

/**
 * `std::vector<T>` with vector space functionality and l2-norms and scalar products.
 * Its entries must also support vector operations and scalar products.
 */
template <typename T>
class Tuple {
 public:
  Tuple() {}

  explicit Tuple(size_t n) : elements(n) {}

  /**
   * Construct a tuple consisting of one given element
   */
  explicit Tuple(const T& t) : elements(1, t) {}

  Tuple(Tuple&& o) : elements(std::move(o.elements)) {}

  Tuple(const Tuple& o) : elements(o.elements) {}

  Tuple<T>& operator+=(const Tuple<T>& o) {
    add(1.0, o);

    return *this;
  }

  Tuple<T>& operator-=(const Tuple<T>& o) {
    add(-1.0, o);

    return *this;
  }

  Tuple<T>& operator*=(const double factor) {
    for (size_t i = 0; i < this->size(); i++) elements[i] *= factor;

    return *this;
  }

  Tuple<T>& operator/=(const double factor) { return *this *= (1.0 / factor); }

  double norm() const { return sqrt(this->dot(*this)); }

  double dot(const Tuple<T>& o) const { return (*this) * o; }

  double operator*(const Tuple<T>& o) const {
    AssertThrow(o.size() == this->size(), ExcInternalError());

    double tmp = 0.0;

    for (size_t i = 0; i < this->size(); i++) tmp += o[i] * elements[i];

    return tmp;
  }

  void add(const double a, const Tuple<T>& o) {
    AssertThrow(o.size() == this->size(), ExcInternalError());

    for (size_t i = 0; i < this->size(); i++) elements[i].add(a, o[i]);
  }

  /**
   * scale by s and add a*V
   */
  void sadd(const double s, const double a, const Tuple<T>& o) {
    AssertThrow(o.size() == this->size(), ExcInternalError());

    for (size_t i = 0; i < this->size(); i++) elements[i].sadd(s, a, o[i]);
  }

  Tuple<T>& operator=(Tuple<T>&& o) {
    elements = std::move(o.elements);

    return *this;
  }

  Tuple<T>& operator=(const Tuple<T>& o) {
    elements = o.elements;

    return *this;
  }

  static Tuple<T> noise(const Tuple<T>& like) {
    Tuple<T> res;

    for (size_t i = 0; i < like.size(); i++) res.push_back(T::noise(like[i]));

    return res;
  }

  static Tuple<T> noise(const Tuple<T>& like, double norm) {
    auto res = noise(like);
    res *= norm / res.norm();

    return res;
  }

  size_t size() const { return elements.size(); }

  T& operator[](size_t i) {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return elements[i];
  }

  const T& operator[](size_t i) const {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return elements[i];
  }

  void push_back(const T& value) { elements.push_back(value); }

  void push_back(T&& value) { elements.push_back(std::move(value)); }

  void write_pvd(std::string path, std::string filename, std::string name) const {
    AssertThrow(elements.size() < 100, ExcNotImplemented());  // 2 digits are ok

    for (size_t i = 0; i < elements.size(); i++)
      elements[i].write_pvd(path, filename + Utilities::int_to_string(i, 2), name);
  }

  void reserve(int num) { elements.reserve(num); }

 private:
  std::vector<T> elements;
};

}  // namespace util
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_ */
