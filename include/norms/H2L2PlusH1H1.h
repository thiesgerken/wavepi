/*
 * H2L2PlusH1H1.h
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#ifndef INCLUDE_NORMS_H2L2PLUSH1H1_H_
#define INCLUDE_NORMS_H2L2PLUSH1H1_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <string>

namespace wavepi {
namespace norms {

using namespace wavepi::base;

/**
 * H²([0,T], L²(Ω)) ∩ H¹([0,T], H¹(Ω)) norm, using the trapezoidal rule in time (approximation),
 * the mass matrix in space (exact) and finite differences of order h² (inner) and h (boundary)
 * Implements (u,v) = (u,v)_L² + ɣ  (∇u,∇v) + α (u',v')_L² + β (u'', v'')_L² with positive α, β and ɣ.
 *
 * Can handle adaptive meshes and therefore provides an alternative (albeit much slower) implementation of the other
 * Sobolev norms.
 */
template <int dim>
class H2L2PlusH1H1 : public Norm<DiscretizedFunction<dim>> {
 public:
  virtual ~H2L2PlusH1H1() = default;
  H2L2PlusH1H1(double alpha, double beta, double gamma);

  virtual double norm(const DiscretizedFunction<dim>& u) const override;

  virtual double dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const override;

  virtual void dot_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual void dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual bool hilbert() const override;

  virtual std::string name() const override;

  virtual std::string unique_id() const override;

  inline double alpha() const { return alpha_; }
  inline void alpha(double alpha) { alpha_ = alpha; }

  inline double beta() const { return beta_; }
  inline void beta(double beta) { beta_ = beta; }

  inline double gamma() const { return gamma_; }
  inline void gamma(double gamma) { gamma_ = gamma; }

 private:
  double alpha_;
  double beta_;
  double gamma_;
};

} /* namespace norms */
} /* namespace wavepi */

#endif /* INCLUDE_NORMS_H2L2PLUSH1H1_H_ */
