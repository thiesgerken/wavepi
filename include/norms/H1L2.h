/*
 * H1L2.h
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#ifndef INCLUDE_NORMS_H1L2_H_
#define INCLUDE_NORMS_H1L2_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <base/SpaceTimeMesh.h>
#include <deal.II/lac/sparse_direct.h>
#include <memory>
#include <string>

namespace wavepi {
namespace norms {

using namespace wavepi::base;
using namespace dealii;

/**
 * H¹([0,T], L²(Ω)) norm, using the trapezoidal rule in time (approximation),
 * the mass matrix in space (exact) and finite differences of order h² (inner) and h (boundary)
 * Implements (u,v) = (u,v)_L² + α (u',v')_L² with positive α.
 */
template <int dim>
class H1L2 : public Norm<DiscretizedFunction<dim>> {
 public:
  virtual ~H1L2() = default;
  H1L2(double alpha);

  virtual double norm(const DiscretizedFunction<dim>& u) const override;

  virtual double dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const override;

  virtual void dot_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual void dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual std::string name() const override;

  virtual std::string unique_id() const override;

  double alpha() const { return alpha_; }
  void alpha(double alpha) { alpha_ = alpha; }

 private:
  double alpha_;

  std::shared_ptr<SparseDirectUMFPACK> umfpack;

  void factorize_matrix(std::shared_ptr<SpaceTimeMesh<dim>> mesh);
};

} /* namespace norms */
} /* namespace wavepi */

#endif /* INCLUDE_NORMS_H1L2_H_ */
