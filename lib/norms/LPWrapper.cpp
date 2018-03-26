/*
 * LPWrapper.cpp
 *
 *  Created on: 26.03.2018
 *      Author: thies
 */

#include <norms/LPWrapper.h>

namespace wavepi {
namespace norms {

template <int dim>
double LPWrapper<dim>::norm(const DiscretizedFunction<dim>& u) const {
  auto tmp = u;
  base->dot_transform(tmp);

  return tmp.norm_p(p_);
}

template <int dim>
double LPWrapper<dim>::norm_dual(const DiscretizedFunction<dim>& u) const {
  auto tmp = u;
  base->dot_transform_inverse(tmp);

  return tmp.norm_p(q_);
}

template <int dim>
double LPWrapper<dim>::dot(const DiscretizedFunction<dim>& u __attribute((unused)),
                           const DiscretizedFunction<dim>& v __attribute((unused))) const {
  AssertThrow(false, ExcMessage("call to dot on LPWrapper"));
  return 0.0;
}

template <int dim>
void LPWrapper<dim>::dot_transform(DiscretizedFunction<dim>& u __attribute((unused))) {
  AssertThrow(false, ExcMessage("call to dot_transform on LPWrapper"));
}

template <int dim>
void LPWrapper<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u __attribute((unused))) {
  AssertThrow(false, ExcMessage("call to dot_transform_inverse on LPWrapper"));
}

template <int dim>
void LPWrapper<dim>::dot_solve_mass_and_transform(DiscretizedFunction<dim>& u __attribute((unused))) {
  AssertThrow(false, ExcMessage("call to dot_solve_mass_and_transform on LPWrapper"));
}

template <int dim>
void LPWrapper<dim>::dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u __attribute((unused))) {
  AssertThrow(false, ExcMessage("call to dot_mult_mass_and_transform_inverse on LPWrapper"));
}

template <int dim>
bool LPWrapper<dim>::hilbert() const {
  return false;
}

template <int dim>
void LPWrapper<dim>::duality_mapping(DiscretizedFunction<dim>& x, double p) {
  base->dot_transform(x);
  x.duality_mapping_lp(p_);
  base->dot_transform(x);

  if (p != p_) x *= std::pow(x.norm(), (p - 1) / (p_ - 1));
}

template <int dim>
void LPWrapper<dim>::duality_mapping_dual(DiscretizedFunction<dim>& x, double q) {
  base->dot_transform_inverse(x);
  x.duality_mapping_lp(q_);
  base->dot_transform_inverse(x);

  if (q != q_) x *= std::pow(x.norm(), (q - 1) / (q_ - 1));
}

template <int dim>
std::string LPWrapper<dim>::name() const {
  return "p-norm on top of " + base->name() + " with p=" + std::to_string(p_);
}

template <int dim>
std::string LPWrapper<dim>::unique_id() const {
  return "p-norm on top of " + base->unique_id() + " with p=" + std::to_string(p_);
}

template class LPWrapper<1>;
template class LPWrapper<2>;
template class LPWrapper<3>;

} /* namespace norms */
} /* namespace wavepi */
