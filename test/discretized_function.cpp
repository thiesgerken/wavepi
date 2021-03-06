/*
 * discretized_function.cpp
 *
 *  Created on: 22.07.2017
 *      Author: thies
 */

#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <base/SpaceTimeMesh.h>
#include <base/Util.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <gtest/gtest.h>
#include <norms/H1H1.h>
#include <norms/H1L2.h>
#include <norms/H2L2.h>
#include <norms/H2L2PlusL2H1.h>
#include <norms/L2Coefficients.h>
#include <norms/L2L2.h>
#include <stddef.h>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using namespace dealii;
using namespace wavepi::base;
using namespace wavepi;

template <int dim>
class TestF : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if ((this->get_time() <= 1) && (p.norm() < 0.5))
      return std::sin(this->get_time() * 2 * numbers::PI);
    else
      return 0.0;
  }
};

template <int dim>
class TestF2 : public Function<dim> {
 public:
  TestF2() : Function<dim>() {}
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if ((this->get_time() <= 0.5) && (p.distance(actor_position) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
    else
      return 0.0;
  }

 private:
  static const Point<dim> actor_position;
};

template <>
const Point<1> TestF2<1>::actor_position = Point<1>(1.0);
template <>
const Point<2> TestF2<2>::actor_position = Point<2>(1.0, 0.5);
template <>
const Point<3> TestF2<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template <int dim>
class TestG : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    Point<dim> pc = Point<dim>::unit_vector(0);
    pc *= 0.5;

    if (std::abs(this->get_time() - 1.0) < 0.5 && (p.distance(pc) < 0.5))
      return std::sin(this->get_time() * 1 * numbers::PI);
    else
      return 0.0;
  }
};

template <int dim>
class TestH : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return p.norm() * this->get_time();
  }
};

template <int dim>
double rho(const Point<dim> &p, double t) {
  return p.norm() + t + 1.0;
}

template <int dim>
double c_squared(const Point<dim> &p, double t) {
  double tmp = p.norm() * t + 1.0;

  return tmp * tmp;
}

template <int dim>
class TestC : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return 1.0 / (rho(p, this->get_time()) * c_squared(p, this->get_time()));
  }
};

template <int dim>
class TestA : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return 1.0 / rho(p, this->get_time());
  }
};

template <int dim>
class TestNu : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return std::abs(p[0]) * this->get_time();
  }
};

template <int dim>
class TestQ : public Function<dim> {
 public:
  TestQ() : Function<dim>() {}
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return p.norm() < 0.5 ? std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
  }

  static const Point<dim> q_position;
};

template <>
const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template <>
const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template <>
const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

// tests whether norm and dot product are consistent
template <int dim>
void run_dot_norm_test(int fe_order, int quad_order, int refines, int n_steps,
                       std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm) {
  deallog << "== " << norm->name() << " ==" << std::endl;

  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  TestQ<dim> q_cont;
  DiscretizedFunction<dim> q(mesh, q_cont);
  q.set_norm(norm);

  double norm_q       = q.norm();
  double sqrt_dot_q_q = std::sqrt(q * q);
  double q_err        = std::abs(norm_q - sqrt_dot_q_q) / (std::abs(norm_q) + 1e-300);

  deallog << std::scientific << "???q??? = " << norm_q << ", ???(q, q) = " << sqrt_dot_q_q << ", rel. error = " << q_err
          << std::endl;

  TestG<dim> g_cont;
  DiscretizedFunction<dim> g(mesh, g_cont);
  g.set_norm(norm);

  double norm_g       = g.norm();
  double sqrt_dot_g_g = std::sqrt(g * g);
  double g_err        = std::abs(norm_g - sqrt_dot_g_g) / (std::abs(norm_g) + 1e-300);

  deallog << std::scientific << "???g??? = " << norm_q << ", ???(g, g) = " << sqrt_dot_g_g << ", rel. error = " << g_err
          << std::endl;

  TestF<dim> f_cont;
  DiscretizedFunction<dim> f(mesh, f_cont);
  f.set_norm(norm);

  double norm_f       = f.norm();
  double sqrt_dot_f_f = std::sqrt(f * f);
  double f_err        = std::abs(norm_f - sqrt_dot_f_f) / (std::abs(norm_f) + 1e-300);

  deallog << std::scientific << "???f??? = " << norm_q << ", ???(f, f) = " << sqrt_dot_f_f << ", rel. error = " << f_err
          << std::endl;

  double tol = 1e-14;

  EXPECT_LT(q_err, tol);
  EXPECT_LT(f_err, tol);
  EXPECT_LT(g_err, tol);

  deallog << std::endl;
}

// tests whether mass matrix operations are inverse to each other
template <int dim>
void run_dot_transform_inverse_test(int fe_order, int quad_order, int refines, int n_steps,
                                    std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm) {
  deallog << "== " << norm->name() << " ==" << std::endl;

  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  const double tol = 1e-2;
  const int N      = 10;

  Timer inverse_timer;
  Timer transform_timer;

  auto norm_vector = std::make_shared<norms::L2Coefficients<dim>>();
  auto norm_h1h1   = std::make_shared<norms::H1H1<dim>>(0.25, 0.25);  // for smoothing of noise

  for (int i = 0; i < N; i++) {
    DiscretizedFunction<dim> t = DiscretizedFunction<dim>::noise(mesh);

    // smooth a bit
    t.set_norm(norm_h1h1);
    t.dot_transform_inverse();
    t.mult_mass();
    t.mult_mass();

    // equip with correct norm
    t.set_norm(norm);

    {
      DiscretizedFunction<dim> x       = t;
      DiscretizedFunction<dim> trans_x = x;

      transform_timer.start();
      trans_x.dot_transform();
      transform_timer.stop();

      inverse_timer.start();
      trans_x.dot_transform_inverse();
      inverse_timer.stop();

      trans_x -= x;

      // error calculation in vector norm
      trans_x.set_norm(norm_vector);
      x.set_norm(norm_vector);
      double err_trans = trans_x.norm() / x.norm();

      deallog << std::scientific << "???x - T^{-1} T x??? / ???x???         = " << err_trans << std::endl;
      EXPECT_LT(err_trans, tol);
    }

    {
      DiscretizedFunction<dim> x       = t;
      DiscretizedFunction<dim> trans_x = x;

      inverse_timer.start();
      trans_x.dot_transform_inverse();
      inverse_timer.stop();

      transform_timer.start();
      trans_x.dot_transform();
      transform_timer.stop();

      trans_x -= x;

      // error calculation in vector norm
      trans_x.set_norm(norm_vector);
      x.set_norm(norm_vector);
      double err_trans = trans_x.norm() / x.norm();

      deallog << std::scientific << "???x - T T^{-1} x??? / ???x???         = " << err_trans << std::endl;
      EXPECT_LT(err_trans, tol);
    }

    {
      DiscretizedFunction<dim> x             = t;
      DiscretizedFunction<dim> solve_trans_x = x;

      transform_timer.start();
      solve_trans_x.dot_solve_mass_and_transform();
      transform_timer.stop();

      inverse_timer.start();
      solve_trans_x.dot_transform_inverse();
      inverse_timer.stop();

      x.solve_mass();
      solve_trans_x -= x;

      // error calculation in vector norm
      solve_trans_x.set_norm(norm_vector);
      x.set_norm(norm_vector);
      double err_solve_trans = solve_trans_x.norm() / x.norm();

      deallog << std::scientific << "???M^{-1} x - T^{-1} [T M^{-1}] x??? / ???M^{-1} x??? = " << err_solve_trans << std::endl;
      EXPECT_LT(err_solve_trans, tol);
    }

    {
      DiscretizedFunction<dim> x            = t;
      DiscretizedFunction<dim> mult_trans_x = x;

      inverse_timer.start();
      mult_trans_x.dot_mult_mass_and_transform_inverse();
      inverse_timer.stop();

      transform_timer.start();
      mult_trans_x.dot_transform();
      transform_timer.stop();

      x.mult_mass();
      mult_trans_x -= x;

      // error calculation in vector norm
      mult_trans_x.set_norm(norm_vector);
      x.set_norm(norm_vector);
      double err_mult_trans = mult_trans_x.norm() / x.norm();

      deallog << std::scientific << "???M x - T [T^{-1} M] x??? / ???M x??? = " << err_mult_trans << std::endl;
      EXPECT_LT(err_mult_trans, tol);
    }

    deallog << std::endl;
  }

  deallog << "avg time for dot_transform        : " << Util::format_duration(transform_timer.wall_time() / (3 * N))
          << std::endl;
  deallog << "avg time for dot_transform_inverse: " << Util::format_duration(inverse_timer.wall_time() / (3 * N))
          << std::endl;

  deallog << std::endl;
}

// tests whether dot_transform is consistent with dot
template <int dim>
void run_dot_transform_consistent_test(int fe_order, int quad_order, int refines, int n_steps,
                                       std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm) {
  deallog << "== " << norm->name() << " ==" << std::endl;

  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  for (int i = 0; i < 10; i++) {
    DiscretizedFunction<dim> x = DiscretizedFunction<dim>::noise(mesh);
    x.set_norm(norm);

    DiscretizedFunction<dim> y = DiscretizedFunction<dim>::noise(mesh);
    y.set_norm(norm);

    double dot_xy = x * y;

    y.dot_transform();
    x.set_norm(std::make_shared<norms::L2Coefficients<dim>>());
    y.set_norm(std::make_shared<norms::L2Coefficients<dim>>());

    double dot_xy_via_transform = x * y;
    double err                  = std::abs(dot_xy - dot_xy_via_transform) / (std::abs(dot_xy) + 1e-300);

    deallog << std::scientific << "(x, y) = " << dot_xy << ", x^t T y = " << dot_xy_via_transform
            << ", rel. error = " << err << std::endl;

    double tol = 1e-10;
    EXPECT_LT(err, tol);
  }

  deallog << std::endl;
}

template <int dim>
void run_derivative_transpose_test(int fe_order, int quad_order, int refines, int n_steps) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  const double tol = 1e-10;

  for (int i = 0; i < 10; i++) {
    DiscretizedFunction<dim> x = DiscretizedFunction<dim>::noise(mesh);
    x.set_norm(std::make_shared<norms::L2Coefficients<dim>>());

    DiscretizedFunction<dim> y = DiscretizedFunction<dim>::noise(mesh);
    y.set_norm(std::make_shared<norms::L2Coefficients<dim>>());

    auto Dy  = y.calculate_derivative();
    auto Dtx = x.calculate_derivative_transpose();

    double dot_x_Dy  = x * Dy;
    double dot_Dtx_y = Dtx * y;
    double err       = std::abs(dot_Dtx_y - dot_x_Dy) / (std::abs(dot_x_Dy) + 1e-300);

    deallog << std::scientific << "(x, Dy) = " << dot_x_Dy << ", (D^t x, y) = " << dot_Dtx_y << ", rel. error = " << err
            << std::endl;
    EXPECT_LT(err, tol);

    auto D2y  = y.calculate_second_derivative();
    auto D2tx = x.calculate_second_derivative_transpose();

    double dot_x_D2y  = x * D2y;
    double dot_D2tx_y = D2tx * y;
    double err2       = std::abs(dot_D2tx_y - dot_x_D2y) / (std::abs(dot_x_D2y) + 1e-300);

    deallog << std::scientific << "(x, D2y) = " << dot_x_D2y << ", (D2^t x, y) = " << dot_D2tx_y
            << ", rel. error = " << err2 << std::endl;
    EXPECT_LT(err2, tol);

    deallog << std::endl;
  }

  deallog << std::endl;
}

template <int dim>
void run_dot_norm_tests(int fe_order, int quad_order, int refines, int n_steps) {
  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::L2Coefficients<dim>>());

  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::L2L2<dim>>());

  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::H1L2<dim>>(0.5));

  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::H1H1<dim>>(0.5, 0.5));

  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::H2L2<dim>>(0.5, 0.25));

  run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps,
                         std::make_shared<norms::H2L2PlusL2H1<dim>>(0.5, 0.25, 0.5));
}

template <int dim>
void run_dot_transform_inverse_tests(int fe_order, int quad_order, int refines, int n_steps) {
  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps,
                                      std::make_shared<norms::L2Coefficients<dim>>());

  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::L2L2<dim>>());

  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::H1L2<dim>>(0.5));

  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps,
                                      std::make_shared<norms::H1H1<dim>>(0.5, 0.5));

  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps,
                                      std::make_shared<norms::H2L2<dim>>(0.5, 0.25));

  run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps,
                                      std::make_shared<norms::H2L2PlusL2H1<dim>>(0.5, 0.25, 0.5));
}

template <int dim>
void run_dot_transform_consistent_tests(int fe_order, int quad_order, int refines, int n_steps) {
  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::L2Coefficients<dim>>());

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps, std::make_shared<norms::L2L2<dim>>());

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::H1L2<dim>>(0.5));

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::H1H1<dim>>(0.5, 0.5));

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::H2L2<dim>>(0.5, 0.25));

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::H2L2<dim>>(0.5, 0.25));

  run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
                                         std::make_shared<norms::H2L2PlusL2H1<dim>>(0.5, 0.25, 0.5));
}

}  // namespace

TEST(DiscretizedFunction, DotNormConsistent1DFE1) {
  run_dot_norm_tests<1>(1, 3, 10, 128);
  run_dot_norm_tests<1>(1, 4, 9, 256);
}
TEST(DiscretizedFunction, DotNormConsistent1DFE2) {
  run_dot_norm_tests<1>(2, 4, 7, 128);
  run_dot_norm_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunction, DotNormConsistent2DFE1) {
  run_dot_norm_tests<2>(1, 3, 5, 128);
  run_dot_norm_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunction, DotNormConsistent2DFE2) {
  run_dot_norm_tests<2>(2, 4, 4, 128);
  run_dot_norm_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunction, DotNormConsistent3DFE1) {
  run_dot_norm_tests<3>(1, 3, 2, 32);
  run_dot_norm_tests<3>(1, 4, 1, 64);
}

TEST(DiscretizedFunction, DerivativeTranspose1DFE1) {
  run_derivative_transpose_test<1>(1, 3, 10, 128);
  run_derivative_transpose_test<1>(1, 4, 9, 256);
}
TEST(DiscretizedFunction, DerivativeTranspose1DFE2) {
  run_derivative_transpose_test<1>(2, 4, 7, 128);
  run_derivative_transpose_test<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunction, DerivativeTranspose2DFE1) {
  run_derivative_transpose_test<2>(1, 3, 5, 128);
  run_derivative_transpose_test<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunction, DerivativeTranspose2DFE2) {
  run_derivative_transpose_test<2>(2, 4, 4, 128);
  run_derivative_transpose_test<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunction, DerivativeTranspose3DFE1) {
  run_derivative_transpose_test<3>(1, 3, 2, 32);
  run_derivative_transpose_test<3>(1, 4, 1, 64);
}

TEST(DiscretizedFunction, DotTransformInverse1DFE1) {
  run_dot_transform_inverse_tests<1>(1, 3, 10, 128);
  run_dot_transform_inverse_tests<1>(1, 4, 9, 256);
}

TEST(DiscretizedFunction, DotTransformInverse1DFE2) {
  run_dot_transform_inverse_tests<1>(2, 4, 7, 128);
  run_dot_transform_inverse_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunction, DotTransformInverse2DFE1) {
  run_dot_transform_inverse_tests<2>(1, 3, 5, 128);
  run_dot_transform_inverse_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunction, DotTransformInverse2DFE2) {
  run_dot_transform_inverse_tests<2>(2, 4, 4, 128);
  run_dot_transform_inverse_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunction, DotTransformInverse3DFE1) {
  run_dot_transform_inverse_tests<3>(1, 3, 2, 32);
  run_dot_transform_inverse_tests<3>(1, 4, 3, 64);
}

TEST(DiscretizedFunction, DotTransformConsistent1DFE1) {
  run_dot_transform_consistent_tests<1>(1, 3, 10, 128);
  run_dot_transform_consistent_tests<1>(1, 4, 9, 256);
}

TEST(DiscretizedFunction, DotTransformConsistent1DFE2) {
  run_dot_transform_consistent_tests<1>(2, 4, 7, 128);
  run_dot_transform_consistent_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunction, DotTransformConsistent2DFE1) {
  run_dot_transform_consistent_tests<2>(1, 3, 5, 128);
  run_dot_transform_consistent_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunction, DotTransformConsistent2DFE2) {
  run_dot_transform_consistent_tests<2>(2, 4, 4, 128);
  run_dot_transform_consistent_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunction, DotTransformConsistent3DFE1) {
  run_dot_transform_consistent_tests<3>(1, 3, 2, 32);
  run_dot_transform_consistent_tests<3>(1, 4, 3, 64);
}
