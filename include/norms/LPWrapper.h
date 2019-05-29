/*
 * LPWrapper.h
 *
 *  Created on: 26.03.2018
 *      Author: thies
 */

#ifndef INCLUDE_NORMS_LPWRAPPER_H_
#define INCLUDE_NORMS_LPWRAPPER_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <deal.II/base/exceptions.h>

namespace wavepi {
namespace norms {

using namespace wavepi::base;
using namespace dealii;

/**
 * Transforms a given dot product (given by matrix X) into an L^p setting by using the norm ||Xf||_p.
 * Note that even for p=2 this yields not the same norm as X, because for that one would have to use ||X^{1/2}f||_p.
 */
template <int dim>
class LPWrapper : public Norm<DiscretizedFunction<dim>> {
 public:
  virtual ~LPWrapper() = default;
  LPWrapper(std::shared_ptr<Norm<DiscretizedFunction<dim>>> base, double p) : base(base), p_(p), q_(p / (p - 1)) {
    AssertThrow(p > 1, ExcMessage("LPWrapper: p<=1 not supported!"));
  }

  virtual double norm(const DiscretizedFunction<dim>& u) const override;
  virtual double norm_dual(const DiscretizedFunction<dim>& u) const override;

  virtual double dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const override;

  virtual void dot_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual void dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) override;

  virtual void dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) override;

  virtual bool hilbert() const override;

  virtual void duality_mapping(DiscretizedFunction<dim>& x, double p) override;
  virtual void duality_mapping_dual(DiscretizedFunction<dim>& x, double q) override;

  virtual std::string name() const override;

  virtual std::string unique_id() const override;

  double p() const { return p_; }
  void p(double p) {
    AssertThrow(p > 1, ExcMessage("LPWrapper: p<=1 not supported!"));
    p_ = p;
    q_ = p / (p - 1);
  }

 private:
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> base;
  double p_;
  double q_;  // = p / (p-1)
};

} /* namespace norms */
} /* namespace wavepi */

#endif /* INCLUDE_NORMS_LPWRAPPER_H_ */
