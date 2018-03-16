/*
 * H2L2.cpp
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
#include <norms/H2L2.h>
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
H2L2<dim>::H2L2(double alpha, double beta) : alpha_(alpha), beta_(beta) {}

template <int dim>
double H2L2<dim>::norm(const DiscretizedFunction<dim>& u) const {
  auto mesh = u.get_mesh();

  // we may be able to use v, but this might introduce inconsistencies in the adjoints
  // Note: this function works even for non-constant meshes.
  auto deriv = u.calculate_derivative();

  // using deriv.calculate_derivative feels wrong, better use a specialized formula.
  auto deriv2 = u.calculate_second_derivative();

  double result = 0;

  for (size_t i = 0; i < mesh->length(); i++) {
    double nrm2        = mesh->get_mass_matrix(i)->matrix_norm_square(u[i]);
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
double H2L2<dim>::dot(const DiscretizedFunction<dim>& u, const DiscretizedFunction<dim>& v) const {
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
    double doti        = mesh->get_mass_matrix(i)->matrix_scalar_product(u[i], v[i]);
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
void H2L2<dim>::dot_transform(DiscretizedFunction<dim>& u) {
  u.mult_mass();
  dot_solve_mass_and_transform(u);
}

template <int dim>
void H2L2<dim>::dot_transform_inverse(DiscretizedFunction<dim>& u) {
  u.solve_mass();
  dot_mult_mass_and_transform_inverse(u);
}

template <int dim>
void H2L2<dim>::dot_solve_mass_and_transform(DiscretizedFunction<dim>& u) {
  auto mesh = u.get_mesh();

  // X = (T + \alpha D^t T D + \beta D_2^t T D_2) * M,
  // M (blocks of mass matrices) is already taken care of, D = derivative, T = trapezoidal rule

  auto dx  = u.calculate_derivative();
  auto d2x = u.calculate_second_derivative();

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
void H2L2<dim>::factorize_matrix(std::shared_ptr<SpaceTimeMesh<dim>> mesh) {
  deallog << "factorizing matrix" << std::endl;

  SparsityPattern pattern(mesh->length(), mesh->length(), 5);

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      pattern.add(i, j);

  for (size_t i = 3; i < mesh->length() - 3; i++) {
    // fill row i and column i
    for (int j = -2; j <= 2; j++) {
      pattern.add(i, i + j);
      pattern.add(i + j, i);
    }
  }

  for (size_t i = 0; i < 3; i++)
    for (size_t j = 0; j < 3; j++)
      pattern.add(mesh->length() - 1 - i, mesh->length() - 1 - j);

  pattern.compress();

  // coefficients of trapezoidal rule
  std::vector<double> lambdas(mesh->length(), 0.0);

  for (size_t i = 0; i < mesh->length(); i++) {
    if (i > 0) lambdas[i] += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;

    if (i < mesh->length() - 1) lambdas[i] += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;
  }

  SparseMatrix<double> matrix(pattern);

  double p20 = 1.0 * 16 / pow4(mesh->get_time(2) - mesh->get_time(0));
  double p10 = 1.0 / pow4(mesh->get_time(1) - mesh->get_time(0));
  double p31 = 1.0 * 16 / pow4(mesh->get_time(3) - mesh->get_time(1));
  double p42 = 1.0 * 16 / pow4(mesh->get_time(4) - mesh->get_time(2));

  matrix.set(0, 0, lambdas[0] * p10 + lambdas[1] * p20);
  matrix.set(1, 1, 4 * lambdas[0] * p10 + 4 * lambdas[1] * p20 + lambdas[2] * p31);
  matrix.set(2, 2, lambdas[0] * p10 + lambdas[1] * p20 + 4 * lambdas[2] * p31 + lambdas[3] * p42);

  matrix.set(0, 1, -2 * lambdas[0] * p10 - 2 * lambdas[1] * p20);
  matrix.set(1, 0, -2 * lambdas[0] * p10 - 2 * lambdas[1] * p20);

  matrix.set(0, 2, lambdas[0] * p10 + lambdas[1] * p20);
  matrix.set(2, 0, lambdas[0] * p10 + lambdas[1] * p20);

  matrix.set(1, 2, -2 * lambdas[0] * p10 - 2 * lambdas[1] * p20 - 2 * lambdas[2] * p31);
  matrix.set(2, 1, -2 * lambdas[0] * p10 - 2 * lambdas[1] * p20 - 2 * lambdas[2] * p31);

  for (size_t i = 3; i < mesh->length() - 3; i++) {
    // fill row i and column i

    double p20  = 1.0 * 16 / pow4(mesh->get_time(i + 2) - mesh->get_time(i));
    double p0m2 = 1.0 * 16 / pow4(mesh->get_time(i - 2) - mesh->get_time(i));
    double p1m1 = 1.0 * 16 / pow4(mesh->get_time(i + 1) - mesh->get_time(i - 1));

    matrix.set(i, i, lambdas[i + 1] * p20 + 4 * lambdas[i] * p1m1 + lambdas[i - 1] * p0m2);

    matrix.set(i, i - 1, -2 * lambdas[i - 1] * p0m2 - 2 * lambdas[i] * p1m1);
    matrix.set(i - 1, i, -2 * lambdas[i - 1] * p0m2 - 2 * lambdas[i] * p1m1);

    matrix.set(i, i + 1, -2 * lambdas[i + 1] * p20 - 2 * lambdas[i] * p1m1);
    matrix.set(i + 1, i, -2 * lambdas[i + 1] * p20 - 2 * lambdas[i] * p1m1);

    matrix.set(i, i + 2, lambdas[i + 1] * p20);
    matrix.set(i + 2, i, lambdas[i + 1] * p20);

    matrix.set(i, i - 2, lambdas[i - 1] * p0m2);
    matrix.set(i - 2, i, lambdas[i - 1] * p0m2);
  }

  // (symmetric to the first entries)
  size_t N = mesh->length() - 1;  // makes it easier to read

  p20 = 1.0 * 16 / pow4(mesh->get_time(N - 2) - mesh->get_time(N - 0));
  p10 = 1.0 / pow4(mesh->get_time(N - 1) - mesh->get_time(N - 0));
  p31 = 1.0 * 16 / pow4(mesh->get_time(N - 3) - mesh->get_time(N - 1));
  p42 = 1.0 * 16 / pow4(mesh->get_time(N - 4) - mesh->get_time(N - 2));

  matrix.set(N - 0, N - 0, lambdas[N - 0] * p10 + lambdas[N - 1] * p20);
  matrix.set(N - 1, N - 1, 4 * lambdas[N - 0] * p10 + 4 * lambdas[N - 1] * p20 + lambdas[N - 2] * p31);
  matrix.set(N - 2, N - 2,
             lambdas[N - 0] * p10 + lambdas[N - 1] * p20 + 4 * lambdas[N - 2] * p31 + lambdas[N - 3] * p42);

  matrix.set(N - 0, N - 1, -2 * lambdas[N - 0] * p10 - 2 * lambdas[N - 1] * p20);
  matrix.set(N - 1, N - 0, -2 * lambdas[N - 0] * p10 - 2 * lambdas[N - 1] * p20);

  matrix.set(N - 0, N - 2, lambdas[N - 0] * p10 + lambdas[N - 1] * p20);
  matrix.set(N - 2, N - 0, lambdas[N - 0] * p10 + lambdas[N - 1] * p20);

  matrix.set(N - 1, N - 2, -2 * lambdas[N - 0] * p10 - 2 * lambdas[N - 1] * p20 - 2 * lambdas[N - 2] * p31);
  matrix.set(N - 2, N - 1, -2 * lambdas[N - 0] * p10 - 2 * lambdas[N - 1] * p20 - 2 * lambdas[N - 2] * p31);

  matrix *= beta_;

  // H1 part (+ trapezoidal rule)
  SparseMatrix<double> matrixH1(pattern);

  double sq20 = 1.0 / square(mesh->get_time(2) - mesh->get_time(0));
  double sq10 = 1.0 / square(mesh->get_time(1) - mesh->get_time(0));
  double sq31 = 1.0 / square(mesh->get_time(3) - mesh->get_time(1));

  matrixH1.set(0, 0, lambdas[1] * sq20 + lambdas[0] * sq10);
  matrixH1.set(1, 1, lambdas[2] * sq31 + lambdas[0] * sq10);
  matrixH1.set(0, 1, -lambdas[0] * sq10);
  matrixH1.set(1, 0, -lambdas[0] * sq10);

  for (size_t i = 2; i < mesh->length() - 2; i++) {
    // fill row i and column i

    double sq20  = 1.0 / square(mesh->get_time(i + 2) - mesh->get_time(i));
    double sq0m2 = 1.0 / square(mesh->get_time(i) - mesh->get_time(i - 2));

    matrixH1.set(i, i, lambdas[i + 1] * sq20 + lambdas[i - 1] * sq0m2);

    matrixH1.set(i, i - 2, -lambdas[i - 1] * sq0m2);
    matrixH1.set(i - 2, i, -lambdas[i - 1] * sq0m2);

    matrixH1.set(i, i + 2, -lambdas[i + 1] * sq20);
    matrixH1.set(i + 2, i, -lambdas[i + 1] * sq20);
  }

  // (symmetric to the first entries)
  sq20 = 1.0 / square(mesh->get_time(N - 2) - mesh->get_time(N));
  sq10 = 1.0 / square(mesh->get_time(N - 1) - mesh->get_time(N));
  sq31 = 1.0 / square(mesh->get_time(N - 3) - mesh->get_time(N - 1));

  matrixH1.set(N, N - 0, lambdas[N - 1] * sq20 + lambdas[N] * sq10);
  matrixH1.set(N - 1, N - 1, lambdas[N - 2] * sq31 + lambdas[N] * sq10);
  matrixH1.set(N, N - 1, -lambdas[N] * sq10);
  matrixH1.set(N - 1, N, -lambdas[N] * sq10);

  matrix.add(alpha_, matrixH1);

  // L2 part (+ trapezoidal rule)
  for (size_t i = 0; i < mesh->length(); i++)
    matrix.add(i, i, lambdas[i]);

  umfpack.factorize(matrix);
}

template <int dim>
void H2L2<dim>::dot_mult_mass_and_transform_inverse(DiscretizedFunction<dim>& u) {
  auto mesh = u.get_mesh();

  LogStream::Prefix p("h2l2_transform");
  Timer timer;
  timer.start();

  if (umfpack.n() != mesh->length()) factorize_matrix(mesh);

  // just to be sure
  for (size_t i = 0; i < mesh->length(); i++)
    Assert(u[i].size() == u[0].size(), ExcInternalError());

  // solve for every DoF
  Vector<double> tmp(mesh->length());

  for (size_t i = 0; i < u[0].size(); i++) {
    for (size_t j = 0; j < mesh->length(); j++)
      tmp[j] = u[j][i];

    umfpack.solve(tmp);

    for (size_t j = 0; j < mesh->length(); j++)
      u[j][i] = tmp[j];
  }

  deallog << "solved in " << Util::format_duration(timer.wall_time()) << std::endl;
}

template <int dim>
bool H2L2<dim>::hilbert() const {
  return true;
}

template <int dim>
std::string H2L2<dim>::name() const {
  return "H²([0,T], L²(Ω))";
}

template <int dim>
std::string H2L2<dim>::unique_id() const {
  return "H²([0,T], L²(Ω)) with α=" + std::to_string(alpha_) + ", β=" + std::to_string(beta_);
}

template class H2L2<1>;
template class H2L2<2>;
template class H2L2<3>;

} /* namespace norms */
} /* namespace wavepi */
