/*
 * Tuple.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_TUPLE_H_
#define INCLUDE_MEASUREMENTS_TUPLE_H_

#include <deal.II/base/exceptions.h>
#include <stddef.h>
#include <vector>

namespace wavepi {
namespace measurements {
using namespace dealii;

/**
 * `std::vector<T>` with vector space functionality and l2-norms and scalar products.
 * Its entries must also support vector operations and scalar products.
 */
template<typename T>
class Tuple: public std::vector<T> {
   public:
      Tuple()
            : std::vector<T>() {
      }

      explicit Tuple(size_t n)
            : std::vector<T>(n) {
      }

      Tuple(Tuple&& o)
            : std::vector<T>(o) {
      }

      Tuple(const Tuple& o)
            : std::vector<T>(o) {
      }

      Tuple<T>& operator+=(const Tuple<T>& o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            *this[i] += o[i];

         return *this;
      }

      Tuple<T>& operator-=(const Tuple<T>& o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            *this[i] -= o[i];

         return *this;
      }

      Tuple<T>& operator*=(const double factor) {
         for (size_t i = 0; i < this->size(); i++)
            *this[i] *= factor;

         return *this;
      }

      Tuple<T>& operator/=(const double factor) {
         for (size_t i = 0; i < this->size(); i++)
            *this[i] /= factor;

         return *this;
      }

      double norm() const {
         return sqrt((*this) * (*this));
      }

      double dot(const Tuple<T>& o) const {
         return (*this) * o;
      }

      double operator*(const Tuple<T>& o) const {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         double tmp = 0.0;

         for (size_t i = 0; i < this->size(); i++)
            tmp += o[i] * (*this[i]);

         return tmp;
      }

};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_ */
