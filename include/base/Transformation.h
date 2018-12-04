/*
 * Transformation.h
 *
 *  Created on: 15.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_BASE_TRANSFORMATION_H_
#define INCLUDE_BASE_TRANSFORMATION_H_

#include <base/DiscretizedFunction.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <cmath>
#include <memory>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * Transformation φ to apply before the forward operator S, i.e. S_{new}(φ(x)) := S(x).
 */
template <int dim>
class Transformation {
 public:
  Transformation()          = default;
  virtual ~Transformation() = default;

  /**
   * Apply φ to a given discretized function
   */
  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param) = 0;

  /**
   * Transform a function
   */
  virtual std::shared_ptr<LightFunction<dim>> transform(const std::shared_ptr<LightFunction<dim>> param) = 0;

  /**
   * Apply φ^{-1} to a given function
   */
  virtual DiscretizedFunction<dim> transform_inverse(const DiscretizedFunction<dim> &param) = 0;

  /**
   * Apply the linear operator φ^{-1}'(p) to a given function h
   */
  virtual DiscretizedFunction<dim> inverse_derivative(const DiscretizedFunction<dim> &param,
                                                      const DiscretizedFunction<dim> &h) = 0;
  /**
   * Apply the linear operator φ^{-1}'(p)* to a given function g,
   * where the adjoint should be formed regarding vector dot products.
   */
  virtual DiscretizedFunction<dim> inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                const DiscretizedFunction<dim> &g) = 0;
};

/**
 * Transformation with φ(x) = x.
 */
template <int dim>
class IdentityTransform : public Transformation<dim> {
 public:
  IdentityTransform()          = default;
  virtual ~IdentityTransform() = default;

  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param) override;

  virtual std::shared_ptr<LightFunction<dim>> transform(const std::shared_ptr<LightFunction<dim>> param) override;

  virtual DiscretizedFunction<dim> transform_inverse(const DiscretizedFunction<dim> &param) override;

  virtual DiscretizedFunction<dim> inverse_derivative(const DiscretizedFunction<dim> &param,
                                                      const DiscretizedFunction<dim> &h) override;

  virtual DiscretizedFunction<dim> inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                const DiscretizedFunction<dim> &g) override;
};

/**
 * Transformation with φ(x) = log(x) (pointwise).
 */
template <int dim>
class LogTransform : public Transformation<dim> {
 public:
  LogTransform()          = default;
  virtual ~LogTransform() = default;

  static void declare_parameters(ParameterHandler &prm);

  void get_parameters(ParameterHandler &prm);

  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param) override;

  virtual std::shared_ptr<LightFunction<dim>> transform(const std::shared_ptr<LightFunction<dim>> param) override;

  virtual DiscretizedFunction<dim> transform_inverse(const DiscretizedFunction<dim> &param) override;

  virtual DiscretizedFunction<dim> inverse_derivative(const DiscretizedFunction<dim> &param,
                                                      const DiscretizedFunction<dim> &h) override;

  virtual DiscretizedFunction<dim> inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                const DiscretizedFunction<dim> &g) override;

 private:
  double lower_bound = 0.0;

  class TransformFunction : public LightFunction<1> {
   public:
    TransformFunction(double lb) : lower_bound(lb){};
    virtual ~TransformFunction() = default;

    virtual double evaluate(const Point<1> &p, const double time __attribute__((unused))) const override {
          return std::log(p[0] - lower_bound);
    }

   private:
    double lower_bound;
  };
};

} /* namespace base */
} /* namespace wavepi */

#endif /* INCLUDE_BASE_TRANSFORMATION_H_ */
