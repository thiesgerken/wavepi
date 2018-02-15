/*
 * Transformation.h
 *
 *  Created on: 15.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_BASE_TRANSFORMATION_H_
#define INCLUDE_BASE_TRANSFORMATION_H_

#include <base/DiscretizedFunction.h>

namespace wavepi {
namespace base {

/**
 * Transformation φ to apply before the forward operator S, i.e. S_{new}(φ(x)) := S(x).
 */
template <int dim>
class Transformation {
 public:
  Transformation()          = default;
  virtual ~Transformation() = default;

  /**
   * Apply φ to a given function
   */
  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param) = 0;

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

  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param);

  virtual DiscretizedFunction<dim> transform_inverse(const DiscretizedFunction<dim> &param);

  virtual DiscretizedFunction<dim> inverse_derivative(const DiscretizedFunction<dim> &param,
                                                      const DiscretizedFunction<dim> &h);

  virtual DiscretizedFunction<dim> inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                const DiscretizedFunction<dim> &g);
};

/**
 * Transformation with φ(x) = log(x) (pointwise).
 */
template <int dim>
class LogTransform : public Transformation<dim> {
 public:
  LogTransform()          = default;
  virtual ~LogTransform() = default;

  virtual DiscretizedFunction<dim> transform(const DiscretizedFunction<dim> &param);

  virtual DiscretizedFunction<dim> transform_inverse(const DiscretizedFunction<dim> &param);

  virtual DiscretizedFunction<dim> inverse_derivative(const DiscretizedFunction<dim> &param,
                                                      const DiscretizedFunction<dim> &h);

  virtual DiscretizedFunction<dim> inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                const DiscretizedFunction<dim> &g);
};

} /* namespace base */
} /* namespace wavepi */

#endif /* INCLUDE_BASE_TRANSFORMATION_H_ */
