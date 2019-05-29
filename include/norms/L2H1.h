/*
 * L2H1.h
 *
 *  Created on: 28.05.2019
 *      Author: thies
 */

#ifndef INCLUDE_NORMS_L2H1_H_
#define INCLUDE_NORMS_L2H1_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <string>

namespace wavepi {
namespace norms {

using namespace wavepi::base;

/**
 * L²([0,T], L²(Ω)) norm, using the trapezoidal rule in time (approximation)
 * and the mass matrix in space (exact)
 */
template <int dim>
class L2H1 : public Norm<DiscretizedFunction<dim>> {
 public:
  virtual ~L2H1() = default;
  L2H1()          = default;

  static double absolute_error(const DiscretizedFunction<dim>& u, Function<dim>& v);

  virtual double norm(const DiscretizedFunction<dim>& u) const override;

  virtual double dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const override;

  virtual void dot_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual void dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual std::string name() const override;

  virtual std::string unique_id() const override;
};

} /* namespace norms */
} /* namespace wavepi */

#endif /* INCLUDE_NORMS_L2H1_H_ */
