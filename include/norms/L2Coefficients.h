/*
 * L2Coefficients.h
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#ifndef INCLUDE_NORMS_L2COEFFICIENTS_H_
#define INCLUDE_NORMS_L2COEFFICIENTS_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <string>

namespace wavepi {
namespace norms {

using namespace wavepi::base;

/**
 * 2-norm on the underlying vectors.
 * Fast, but only a very crude approximation to the L²([0,T], L²(Ω)) norm (even in case of uniform space-time grids and
 * P1-elements)
 */
template <int dim>
class L2Coefficients : public Norm<DiscretizedFunction<dim>> {
 public:
  virtual ~L2Coefficients() = default;
  L2Coefficients()          = default;

  virtual double norm(const DiscretizedFunction<dim>& u) const override;

  virtual double dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const override;

  virtual void dot_transform(DiscretizedFunction<dim>& u) const override;

  virtual void dot_transform_inverse(DiscretizedFunction<dim>& u) const override;

  virtual void dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) const override;

  virtual void dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) const override;

  virtual bool hilbert() const override;

  virtual std::string name() const override;

  virtual std::string unique_id() const override;
};

} /* namespace norms */
} /* namespace wavepi */

#endif /* INCLUDE_NORMS_L2COEFFICIENTS_H_ */
