/*
 * LightFunction.h
 *
 *  Created on: 01.09.2017
 *      Author: thies
 */

#ifndef INCLUDE_BASE_LIGHTFUNCTION_H_
#define INCLUDE_BASE_LIGHTFUNCTION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * no `set_time` s.t. evaluation can be thread safe
 */
template<int dim>
class LightFunction : public Function<dim> {
public:
   virtual ~LightFunction() = default;

   /**
    * evaluate function
    *
    * @param p point in space to evaluate at
    * @param time time to evaluate at
    */
   virtual double evaluate(const Point<dim> &p, const double time) const = 0;

   virtual double value(const Point<dim> &p, const unsigned int component __attribute__((unused))) const {
      return evaluate (p, this->get_time());
   }
};

template<int dim>
class ConstantFunction: public LightFunction<dim> {
public:
   virtual ~ConstantFunction() = default;

   ConstantFunction(double value = 0.0)
         : x(value) {
   }

   virtual double evaluate(const Point<dim> &p __attribute__((unused)),
         const double time __attribute__((unused))) const {
      return x;
   }

private:
   const double x;
};

}  // namespace base
} /* namespace wavepi */

#endif /* INCLUDE_BASE_LIGHTFUNCTION_H_ */
