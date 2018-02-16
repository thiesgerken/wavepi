/*
 * ComposedFunction.h
 *
 *  Created on: 16.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_BASE_COMPOSEDFUNCTION_H_
#define INCLUDE_BASE_COMPOSEDFUNCTION_H_

#include <deal.II/base/function.h>
#include <memory>

namespace wavepi {
namespace base {

using namespace dealii;

template <int dim>
class ComposedFunction : public Function<dim> {
 public:
  ComposedFunction(std::shared_ptr<Function<dim>> inner, std::shared_ptr<Function<1>> outer)
      : inner(inner), outer(outer) {
    AssertThrow(inner && outer, ExcNotInitialized());
  }
  virtual ~ComposedFunction() = default;

  virtual void set_time(const double new_time) override {
    Function<dim>::set_time(new_time);
    inner->set_time(new_time);
    outer->set_time(new_time);
  }

  virtual void advance_time(const double delta_t) override {
    Function<dim>::advance_time(delta_t);
    inner->advance_time(delta_t);
    outer->advance_time(delta_t);
  }

  virtual double value(const Point<dim>& p, const unsigned int component = 0) const override {
    Assert(component == 0, ExcInternalError());

    Point<1> x(inner->value(p, 0));

    return outer->value(x, 0);
  }

 private:
  std::shared_ptr<Function<dim>> inner;
  std::shared_ptr<Function<1>> outer;
};

} /* namespace base */
} /* namespace wavepi */

#endif /* INCLUDE_BASE_COMPOSEDFUNCTION_H_ */
