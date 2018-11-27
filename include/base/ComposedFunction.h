/*
 * ComposedFunction.h
 *
 *  Created on: 16.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_BASE_COMPOSEDFUNCTION_H_
#define INCLUDE_BASE_COMPOSEDFUNCTION_H_

#include <base/LightFunction.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <memory>

namespace wavepi {
namespace base {

using namespace dealii;

template <int dim>
class ComposedFunction : public LightFunction<dim> {
 public:
  ComposedFunction(std::shared_ptr<LightFunction<dim>> inner, std::shared_ptr<LightFunction<1>> outer)
      : inner(inner), outer(outer) {
    AssertThrow(inner && outer, ExcNotInitialized());
  }
  virtual ~ComposedFunction() = default;


  virtual double evaluate(const Point<dim>& p, const double time) const override {
    Point<1> x(inner->evaluate(p, time));

    return outer->evaluate(x, 0);
  }

 private:
  std::shared_ptr<LightFunction<dim>> inner;
  std::shared_ptr<LightFunction<1>> outer;
};

} /* namespace base */
} /* namespace wavepi */

#endif /* INCLUDE_BASE_COMPOSEDFUNCTION_H_ */
