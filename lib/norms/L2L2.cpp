/*
 * L2L2.cpp
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#include <norms/L2L2.h>

namespace wavepi {
namespace norms {

template <int dim>
double L2L2<dim>::norm(const DiscretizedFunction<dim>& u) const {
  auto mesh     = u.get_mesh();
  double result = 0;

  // trapezoidal rule in time:
  for (size_t i = 0; i < mesh->length(); i++) {
    double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(u[i]);

    if (i > 0) result += nrm2 / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
    if (i < mesh->length() - 1) result += nrm2 / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
  }

  // assume that function is linear in time (consistent with crank-nicolson!)
  // and integrate that exactly (Simpson rule)
  // problem when mesh changes in time!
  //   for (size_t i = 0; i < mesh->length(); i++) {
  //      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);
  //
  //      if (i > 0)
  //         result += nrm2 / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
  //
  //      if (i < mesh->length() - 1)
  //         result += nrm2 / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
  //   }
  //
  //   for (size_t i = 0; i < mesh->length() - 1; i++) {
  //      double tmp = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
  //            function_coefficients[i + 1]);
  //
  //      result += tmp / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
  //   }

  return std::sqrt(result);
}

template <int dim>
double L2L2<dim>::dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const {
  auto mesh     = u.get_mesh();
  double result = 0.0;

  // trapezoidal rule in time:
  for (size_t i = 0; i < mesh->length(); i++) {
    double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(u[i], v[i]);

    if (i > 0) result += doti / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
    if (i < mesh->length() - 1) result += doti / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
  }

  // assume that both functions are linear in time (consistent with crank-nicolson!)
  // and integrate that exactly (Simpson rule)
  // problem when mesh changes in time!
  //   for (size_t i = 0; i < mesh->length(); i++) {
  //      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
  //            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));
  //
  //      double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
  //            V.function_coefficients[i]);
  //
  //      if (i > 0)
  //         result += doti / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
  //
  //      if (i < mesh->length() - 1)
  //         result += doti / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
  //   }
  //
  //   for (size_t i = 0; i < mesh->length() - 1; i++) {
  //      Assert(function_coefficients[i].size() == V.function_coefficients[i+1].size(),
  //            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i+1].size()));
  //      Assert(function_coefficients[i+1].size() == V.function_coefficients[i].size(),
  //             ExcDimensionMismatch (function_coefficients[i+1].size() , V.function_coefficients[i].size()));
  //
  //      double dot1 = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
  //            V.function_coefficients[i + 1]);
  //      double dot2 = mesh->get_mass_matrix(i + 1)->matrix_scalar_product(function_coefficients[i + 1],
  //            V.function_coefficients[i]);
  //
  //      result += (dot1 + dot2) / 6 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
  //   }

  return result;
}

template <int dim>
void L2L2<dim>::dot_transform(DiscretizedFunction<dim>& u) const {
  u.mult_mass();
  dot_solve_mass_and_transform(u);
}

template <int dim>
void L2L2<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u) const {
  u.solve_mass();
  dot_mult_mass_and_transform_inverse(u);
}

template <int dim>
void L2L2<dim>::dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) const {
  auto mesh = u.get_mesh();

  for (size_t i = 0; i < mesh->length(); i++) {
    double factor = 0.0;

    if (i > 0) factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;
    if (i < mesh->length() - 1) factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

    u[i] *= factor;
  }
}

template <int dim>
void L2L2<dim>::dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) const {
  auto mesh = u.get_mesh();

  // trapezoidal rule in time:
  for (size_t i = 0; i < mesh->length(); i++) {
    double factor = 0.0;

    if (i > 0) factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;
    if (i < mesh->length() - 1) factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

    u[i] /= factor;
  }
}

template <int dim>
bool L2L2<dim>::hilbert() const {
  return true;
}

template <int dim>
std::string L2L2<dim>::name() const {
  return "L²([0,T], L²(Ω))";
}

template <int dim>
std::string L2L2<dim>::unique_id() const {
  return "L²([0,T], L²(Ω))";
}

template class L2L2<1>;
template class L2L2<2>;
template class L2L2<3>;

} /* namespace norms */
} /* namespace wavepi */
