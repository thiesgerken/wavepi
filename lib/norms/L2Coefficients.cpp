/*
 * L2Coefficients.cpp
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#include <norms/L2Coefficients.h>

namespace wavepi {
namespace norms {

template <int dim>
double L2Coefficients<dim>::norm(const DiscretizedFunction<dim>& u) const {
  double result = 0;

  for (size_t i = 0; i < u.get_mesh()->length(); i++) {
    double nrm2 = u[i].norm_sqr();
    result += nrm2;
  }

  return std::sqrt(result);
}

template <int dim>
double L2Coefficients<dim>::dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const {
  double result = 0;

  for (size_t i = 0; i < u.get_mesh()->length(); i++) {
    Assert(u[i].size() == v[i].size(), ExcDimensionMismatch(u[i].size(), v[i].size()));
    result += u[i] * v[i];
  }

  return result;
}

template <int dim>
void L2Coefficients<dim>::dot_transform(DiscretizedFunction<dim>& u __attribute((unused))) const {}

template <int dim>
void L2Coefficients<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u __attribute((unused))) const {}

template <int dim>
void L2Coefficients<dim>::dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) const {
  u.solve_mass();
}

template <int dim>
void L2Coefficients<dim>::dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) const {
  u.mult_mass();
}

template <int dim>
bool L2Coefficients<dim>::hilbert() const {
  return true;
}

template <int dim>
std::string L2Coefficients<dim>::name() const {
  return "(ℝⁿ, ‖	·‖₂)";
}

template <int dim>
std::string L2Coefficients<dim>::unique_id() const {
  return "(ℝⁿ, ‖	·‖₂)";
}

template class L2Coefficients<1>;
template class L2Coefficients<2>;
template class L2Coefficients<3>;

} /* namespace norms */
} /* namespace wavepi */
