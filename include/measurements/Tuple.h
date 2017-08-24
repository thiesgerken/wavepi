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
#include <tgmath.h>
#include <random>

namespace wavepi {
namespace measurements {
using namespace dealii;

/**
 * `std::vector<T>` with vector space functionality and l2-norms and scalar products.
 * Its entries must also support vector operations and scalar products.
 */
template<typename T>
class Tuple {
   public:
      Tuple() {
      }

      explicit Tuple(size_t n)
            : elements(n) {
      }

      Tuple(Tuple&& o)
            : elements(std::move(o.elements)) {
      }

      Tuple(const Tuple& o)
            : elements(o.elements) {
      }

      Tuple<T>& operator+=(const Tuple<T>& o) {
         add(1.0, o);

         return *this;
      }

      Tuple<T>& operator-=(const Tuple<T>& o) {
         add(-1.0, o);

         return *this;
      }

      Tuple<T>& operator*=(const double factor) {
         for (size_t i = 0; i < this->size(); i++)
            elements[i] *= factor;

         return *this;
      }

      Tuple<T>& operator/=(const double factor) {
         return *this *= (1.0 / factor);
      }

      double norm() const {
         return sqrt(this->dot(*this));
      }

      double dot(const Tuple<T>& o) const {
         return (*this) * o;
      }

      double operator*(const Tuple<T>& o) const {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         double tmp = 0.0;

         for (size_t i = 0; i < this->size(); i++)
            tmp += o[i] * elements[i];

         return tmp;
      }

      void add(const double a, const Tuple<T>& o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i].add(a, o[i]);
      }

      /**
       * scale by s and add a*V
       */
      void sadd(const double s, const double a, const Tuple<T> &o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i].sadd(s, a, o[i]);
      }

      Tuple<T>& operator=(Tuple<T> && o) {
         elements = std::move(o.elements);

         return *this;
      }

      Tuple<T>& operator=(const Tuple<T> & o) {
         elements = o.elements;

         return *this;
      }

      static Tuple<T> noise(const Tuple<T>& like) {
         Tuple<T> res;

         for (size_t i = 0; i < like.size(); i++)
            res.push_back(T::noise(like[i]));

         return res;
      }

      static Tuple<T> noise(const Tuple<T>& like, double norm) {
         auto res = noise(like);
         res *= norm / res.norm();

         return res;
      }

      size_t size() const {
         return elements.size();
      }

      T& operator[](size_t i) {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      const T& operator[](size_t i) const {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      void push_back(const T& value) {
         elements.push_back(value);
      }

      void push_back(T&& value) {
         elements.push_back(std::move(value));
      }

   private:
      std::vector<T> elements;
};

/**
 * `std::vector<double>` with vector space functionality and l2-norms and scalar products.
 * Its entries must also support vector operations and scalar products.
 */
template<>
class Tuple<double> {
   public:
      Tuple() {
      }

      explicit Tuple(size_t n)
            : elements(n) {
      }

      Tuple(Tuple&& o)
            : elements(std::move(o.elements)) {
      }

      Tuple(const Tuple& o)
            : elements(o.elements) {
      }

      Tuple<double>& operator+=(const Tuple<double>& o) {
         add(1.0, o);

         return *this;
      }

      Tuple<double>& operator-=(const Tuple<double>& o) {
         add(-1.0, o);

         return *this;
      }

      Tuple<double>& operator*=(const double factor) {
         for (size_t i = 0; i < this->size(); i++)
            elements[i] *= factor;

         return *this;
      }

      Tuple<double>& operator/=(const double factor) {
         return *this *= (1.0 / factor);
      }

      double norm() const {
         return sqrt(this->dot(*this));
      }

      double dot(const Tuple<double>& o) const {
         return (*this) * o;
      }

      double operator*(const Tuple<double>& o) const {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         double tmp = 0.0;

         for (size_t i = 0; i < this->size(); i++)
            tmp += o[i] * elements[i];

         return tmp;
      }

      void add(const double a, const Tuple<double>& o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i] += a * o[i];
      }

      /**
       * scale by s and add a*V
       */
      void sadd(const double s, const double a, const Tuple<double> &o) {
         AssertThrow(o.size() == this->size(), ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i] = s * elements[i] + a * o[i];
      }

      Tuple<double>& operator=(Tuple<double> && o) {
         elements = std::move(o.elements);

         return *this;
      }

      Tuple<double>& operator=(const Tuple<double> & o) {
         elements = o.elements;

         return *this;
      }

      static Tuple<double> noise(const Tuple<double>& like) {
         Tuple<double> res(like.size());

         std::default_random_engine generator;
         std::uniform_real_distribution<double> distribution(-1, 1);

         for (size_t i = 0; i < like.size(); i++)
            res[i] = distribution(generator);

         return res;
      }

      static Tuple<double> noise(const Tuple<double>& like, double norm) {
         auto res = noise(like);
         res *= norm / res.norm();

         return res;
      }

      size_t size() const {
         return elements.size();
      }

      double operator[](size_t i) const {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      double& operator[](size_t i) {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      void push_back(const double value) {
         elements.push_back(value);
      }

   private:
      std::vector<double> elements;
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_ */
