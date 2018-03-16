/*
 * H2L2PlusL2H1.cpp
 *
 *  Created on: 13.03.2018
 *      Author: thies
 */

#include <base/Util.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <norms/H1H1.h>
#include <norms/H1L2.h>
#include <norms/H2L2.h>
#include <norms/H2L2PlusL2H1.h>
#include <norms/L2Coefficients.h>
#include <stddef.h>
#include <cmath>
#include <iostream>
#include <vector>

namespace wavepi {
namespace norms {

using namespace dealii;

inline double square(const double x) { return x * x; }
inline double pow4(const double x) { return x * x * x * x; }

template <int dim>
H2L2PlusL2H1<dim>::H2L2PlusL2H1(double alpha, double beta, double gamma) : alpha_(alpha), beta_(beta), gamma_(gamma) {}

template <int dim>
double H2L2PlusL2H1<dim>::norm(const DiscretizedFunction<dim>& u) const {
  auto mesh = u.get_mesh();

  // we may be able to use v, but this might introduce inconsistencies in the adjoints
  // Note: this function works even for non-constant meshes.
  auto deriv = u.calculate_derivative();

  // using deriv.calculate_derivative feels wrong, better use a specialized formula.
  auto deriv2 = u.calculate_second_derivative();

  double result = 0;

  for (size_t i = 0; i < mesh->length(); i++) {
    double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(u[i]) +
                  gamma_ * mesh->get_laplace_matrix(i)->matrix_norm_square(u[i]);
    double nrm2_deriv  = mesh->get_mass_matrix(i)->matrix_norm_square(deriv[i]);
    double nrm2_deriv2 = mesh->get_mass_matrix(i)->matrix_norm_square(deriv2[i]);

    // + trapezoidal rule in time:
    if (i > 0)
      result += (nrm2 + alpha_ * nrm2_deriv + beta_ * nrm2_deriv2) / 2 *
                (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

    if (i < mesh->length() - 1)
      result += (nrm2 + alpha_ * nrm2_deriv + beta_ * nrm2_deriv2) / 2 *
                (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
  }

  return std::sqrt(result);
}

template <int dim>
double H2L2PlusL2H1<dim>::dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const {
  auto mesh     = u.get_mesh();
  double result = 0.0;

  // we may be able to use v, but this might introduce inconsistencies in the adjoints
  // Note: this function works even for non-constant meshes.
  auto deriv  = u.calculate_derivative();
  auto Vderiv = v.calculate_derivative();

  // using deriv.calculate_derivative feels wrong, better use a specialized formula.
  auto deriv2  = u.calculate_second_derivative();
  auto Vderiv2 = v.calculate_second_derivative();

  for (size_t i = 0; i < mesh->length(); i++) {
    double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(u[i], v[i]) +
                  gamma_ * mesh->get_laplace_matrix(i)->matrix_scalar_product(u[i], v[i]);
    double doti_deriv  = mesh->get_mass_matrix(i)->matrix_scalar_product(deriv[i], Vderiv[i]);
    double doti_deriv2 = mesh->get_mass_matrix(i)->matrix_scalar_product(deriv2[i], Vderiv2[i]);

    // + trapezoidal rule in time
    if (i > 0)
      result += (doti + alpha_ * doti_deriv + beta_ * doti_deriv2) / 2 *
                (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

    if (i < mesh->length() - 1)
      result += (doti + alpha_ * doti_deriv + beta_ * doti_deriv2) / 2 *
                (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
  }

  return result;
}

template <int dim>
void H2L2PlusL2H1<dim>::dot_transform(DiscretizedFunction<dim>& u) {
  auto mesh = u.get_mesh();

  // X = (T + \alpha D^t T D + \beta D_2^t T D_2) * M + gamma T L,
  // M = blocks of mass matrices, D = derivative, T = trapezoidal rule, L = blocks of laplace matrices
  DiscretizedFunction<dim> lap_u(mesh, u.get_norm());
  for (size_t i = 0; i < mesh->length(); i++)
    mesh->get_laplace_matrix(i)->vmult(lap_u[i], u[i]);

  u.mult_mass();

  auto dx  = u.calculate_derivative();
  auto d2x = u.calculate_second_derivative();

  // add laplace term
  u.add(gamma_, lap_u);

  // trapezoidal rule
  // (has to happen between D and D^t for dx)
  for (size_t i = 0; i < mesh->length(); i++) {
    double factor = 0.0;

    if (i > 0) factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;
    if (i < mesh->length() - 1) factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

    dx[i] *= factor;
    d2x[i] *= factor;
    u[i] *= factor;
  }

  auto dtdx   = dx.calculate_derivative_transpose();
  auto d2td2x = d2x.calculate_second_derivative_transpose();

  // add derivative terms
  u.add(alpha_, dtdx);
  u.add(beta_, d2td2x);
}

template <int dim>
void H2L2PlusL2H1<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u) {
  LogStream::Prefix p("h2l2plush1h1_transform_inverse");
  Timer timer;
  timer.start();
  // Use CG to invert `dot_transform` (the application of a symmetric+positive definite matrix A)

  // make sure we use standard dot products everywhere
  auto orig_norm = u.get_norm();
  u.set_norm(std::make_shared<L2Coefficients<dim>>());

  // auto precon = std::make_shared<L2Coefficients<dim>>(alpha_); // -> no preconditioning
  // auto precon = std::make_shared<H1L2<dim>>(alpha_);
  // auto precon = std::make_shared<H1H1<dim>>(alpha_, gamma_);
  auto precon = std::make_shared<H2L2<dim>>(alpha_, beta_);

  // use memory of u for the residual
  DiscretizedFunction<dim>& r = u;

  DiscretizedFunction<dim> h = r;
  precon->dot_mult_mass_and_transform_inverse(h);  // faster than just transform_inverse, results are better as well

  DiscretizedFunction<dim> d = h;
  DiscretizedFunction<dim> x(u.get_mesh(), u.get_norm());  // current estimate, initialize with 0
  DiscretizedFunction<dim> z(u.get_mesh(), u.get_norm());  // memory for A*d

  const double tol      = 1e-6;
  const double max_iter = 10000;
  const double norm_rhs = r.norm();
  double disc           = norm_rhs;
  double dot_rh         = r * h;
  size_t iter           = 0;

  // in this case the solution is zero, which u apparently already is
  if (disc == 0.0) return;

  while (disc / norm_rhs >= tol && iter++ < max_iter) {
    LogStream::Prefix p("CG");

    // z <- A d
    z = d;
    dot_transform(z);

    double cg_alpha = dot_rh / (d * z);
    x.add(cg_alpha, d);
    r.add(-cg_alpha, z);

    h = r;
    precon->dot_mult_mass_and_transform_inverse(h);

    // prepare direction for next step
    double dot_rh_next = r * h;
    double cg_beta     = dot_rh_next / dot_rh;
    d.sadd(cg_beta, 1.0, h);

    dot_rh = dot_rh_next;
    disc   = r.norm();

    deallog << "i=" << iter << ": rdisc = " << disc / norm_rhs << std::endl;
  }

  AssertThrow(disc / norm_rhs <= tol, ExcMessage("h2l2plush1h1_transform_inverse: no convergence in CG"));

  // finished
  u = x;

  // to be consistent with other norms, they do not change the norm setting as well (although using it after this
  // transform makes little sense)
  u.set_norm(orig_norm);
  deallog << "solved in " << Util::format_duration(timer.wall_time()) << " after " << iter << " CG steps" << std::endl;
}

// code without preconditioning:
/*
template <int dim>
void H2L2PlusL2H1<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u)  {
  LogStream::Prefix p("h2l2plush1h1_transform_inverse");
  // Use CG to invert `dot_transform` (the application of a symmetric+positive definite matrix A)

  // make sure we use standard dot products everywhere
  auto orig_norm = u.get_norm();
  u.set_norm(std::make_shared<L2Coefficients<dim>>());

  // use memory of u for the residual
  DiscretizedFunction<dim>& r = u;

  DiscretizedFunction<dim> d = r;
  DiscretizedFunction<dim> x(u.get_mesh(), u.get_norm());  // current estimate, initialize with 0
  DiscretizedFunction<dim> z(u.get_mesh(), u.get_norm());  // memory for A*d

  const double tol      = 1e-7;
  const double max_iter = 10000;
  const double norm_rhs = r.norm();
  double disc           = norm_rhs;
  size_t iter           = 0;

  // in this case the solution is zero, which u apparently already is
  if (disc == 0.0) return;

  while (disc / norm_rhs >= tol && iter++ < max_iter) {
    // z <- A d
    z = d;
    dot_transform(z);

    double cg_alpha = square(disc) / (d * z);
    x.add(cg_alpha, d);
    r.add(-cg_alpha, z);

    double disc_new = r.norm();

    // prepare direction for next step
    double cg_beta = square(disc_new / disc);
    d.sadd(cg_beta, 1.0, r);

    disc = disc_new;

    deallog << "i=" << iter << ": rdisc = " << disc / norm_rhs << std::endl;
  }

  AssertThrow(disc / norm_rhs <= tol, ExcMessage("h2l2plush1h1_transform_inverse: no convergence in CG"));

  // finished
  u = x;

  // to be consistent with other norms, they do not change the norm setting as well (although using it after this
  // transform makes little sense)
  u.set_norm(orig_norm);
}
 */

template <int dim>
void H2L2PlusL2H1<dim>::dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) {
  u.solve_mass();
  dot_transform(u);
}

template <int dim>
void H2L2PlusL2H1<dim>::dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) {
  u.mult_mass();
  dot_transform_inverse(u);
}

template <int dim>
bool H2L2PlusL2H1<dim>::hilbert() const {
  return true;
}

template <int dim>
std::string H2L2PlusL2H1<dim>::name() const {
  return "H²([0,T], L²(Ω)) ∩ L²([0,T], H¹(Ω))";
}

template <int dim>
std::string H2L2PlusL2H1<dim>::unique_id() const {
  return "H²([0,T], L²(Ω)) ∩ L²([0,T], H¹(Ω)) with α=" + std::to_string(alpha_) + ", β=" + std::to_string(beta_) +
         ", ɣ=" + std::to_string(gamma_);
}

template class H2L2PlusL2H1<1>;
template class H2L2PlusL2H1<2>;
template class H2L2PlusL2H1<3>;

} /* namespace norms */
} /* namespace wavepi */
