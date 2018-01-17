/*
 * discretized_function.cpp
 *
 *  Created on: 22.07.2017
 *      Author: thies
 */

#include <gtest/gtest.h>

#include <deal.II/base/numbers.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <forward/ConstantMesh.h>
#include <forward/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <problems/L2QProblem.h>

#include <util/GridTools.h>

#include <bits/std_abs.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <functional>

namespace {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::problems;
using namespace wavepi::util;

template<int dim>
class TestF: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));
         if ((this->get_time() <= 1) && (p.norm() < 0.5))
            return std::sin(this->get_time() * 2 * numbers::PI);
         else
            return 0.0;
      }
};

template<int dim>
class TestF2: public Function<dim> {
   public:
      TestF2()
            : Function<dim>() {
      }
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

template<> const Point<1> TestF2<1>::actor_position = Point<1>(1.0);
template<> const Point<2> TestF2<2>::actor_position = Point<2>(1.0, 0.5);
template<> const Point<3> TestF2<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template<int dim>
class TestG: public Function<dim> {
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

template<int dim>
class TestH: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p.norm() * this->get_time();
      }
};

template<int dim>
double rho(const Point<dim> &p, double t) {
   return p.norm() + t + 1.0;
}

template<int dim>
double c_squared(const Point<dim> &p, double t) {
   double tmp = p.norm() * t + 1.0;

   return tmp * tmp;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / (rho(p, this->get_time()) * c_squared(p, this->get_time()));
      }
};

template<int dim>
class TestA: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / rho(p, this->get_time());
      }
};

template<int dim>
class TestNu: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return std::abs(p[0]) * this->get_time();
      }
};

template<int dim>
class TestQ: public Function<dim> {
   public:
      TestQ()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p.norm() < 0.5 ? std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
      }

      static const Point<dim> q_position;
};

template<> const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<> const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<> const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

// tests whether norm and dot product are consistent
template<int dim>
void run_dot_norm_test(int fe_order, int quad_order, int refines, int n_steps,
      typename DiscretizedFunction<dim>::Norm norm) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   wavepi::util::GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad,
         triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

   TestQ<dim> q_cont;
   DiscretizedFunction<dim> q(mesh, q_cont);
   q.set_norm(norm);

   double norm_q = q.norm();
   double sqrt_dot_q_q = std::sqrt(q * q);
   double q_err = std::abs(norm_q - sqrt_dot_q_q) / (std::abs(norm_q) + 1e-300);

   deallog << std::scientific << "‖q‖ = " << norm_q << ", √(q, q) = " << sqrt_dot_q_q << ", rel. error = "
         << q_err << std::endl;

   TestG<dim> g_cont;
   DiscretizedFunction<dim> g(mesh, g_cont);
   g.set_norm(norm);

   double norm_g = g.norm();
   double sqrt_dot_g_g = std::sqrt(g * g);
   double g_err = std::abs(norm_g - sqrt_dot_g_g) / (std::abs(norm_g) + 1e-300);

   deallog << std::scientific << "‖g‖ = " << norm_q << ", √(g, g) = " << sqrt_dot_g_g << ", rel. error = "
         << g_err << std::endl;

   TestF<dim> f_cont;
   DiscretizedFunction<dim> f(mesh, f_cont);
   f.set_norm(norm);

   double norm_f = f.norm();
   double sqrt_dot_f_f = std::sqrt(f * f);
   double f_err = std::abs(norm_f - sqrt_dot_f_f) / (std::abs(norm_f) + 1e-300);

   deallog << std::scientific << "‖f‖ = " << norm_q << ", √(f, f) = " << sqrt_dot_f_f << ", rel. error = "
         << f_err << std::endl;

   double tol = 1e-14;

   EXPECT_LT(q_err, tol);
   EXPECT_LT(f_err, tol);
   EXPECT_LT(g_err, tol);

   deallog << std::endl;
}

// tests whether mass matrix operations are inverse to each other
template<int dim>
void run_dot_transform_inverse_test(int fe_order, int quad_order, int refines, int n_steps,
      typename DiscretizedFunction<dim>::Norm norm) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   wavepi::util::GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad,
         triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

   TestQ<dim> q_cont;
   DiscretizedFunction<dim> q(mesh, q_cont);
   q.set_norm(norm);

   auto mstq(q);
   mstq.dot_transform();
   mstq.dot_transform_inverse();
   mstq -= q;
   double err_mstq = mstq.norm() / q.norm();

   auto mtq(q);
   mtq.dot_solve_mass_and_transform();
   mtq.dot_mult_mass_and_transform_inverse();
   mtq -= q;
   double err_mtq = mtq.norm() / q.norm();

   deallog << std::scientific << "‖q-dot_transform_inverse(dot_transform(q))‖/‖q‖ = " << err_mstq << std::endl;
   deallog << std::scientific << "‖q-dot_mult_mass_and_transform_inverse(dot_solve_mass_and_transform(q))‖/‖q‖ = " << err_mtq << std::endl;

   TestG<dim> g_cont;
   DiscretizedFunction<dim> g(mesh, g_cont);
   g.set_norm(norm);

   auto mstg(g);
   mstg.dot_transform();
   mstg.dot_transform_inverse();
   mstg -= g;
   double err_mstg = mstg.norm() / g.norm();

   auto mtg(g);
   mtg.dot_solve_mass_and_transform();
   mtg.dot_mult_mass_and_transform_inverse();
   mtg -= g;
   double err_mtg = mtg.norm() / g.norm();

   deallog << std::scientific << "‖g-st^-1(st(g))‖/‖g‖ = " << err_mstg << std::endl;
   deallog << std::scientific << "‖g- t^-1( t(g))‖/‖g‖ = " << err_mtg << std::endl;

   TestF<dim> f_cont;
   DiscretizedFunction<dim> f(mesh, f_cont);
   f.set_norm(norm);

   auto mstf(f);
   mstf.dot_transform();
   mstf.dot_transform_inverse();
   mstf -= f;
   double err_mstf = mstf.norm() / f.norm();

   auto mtf(f);
   mtf.dot_solve_mass_and_transform();
   mtf.dot_mult_mass_and_transform_inverse();
   mtf -= f;
   double err_mtf = mtf.norm() / f.norm();

   deallog << std::scientific << "‖f-st^-1(st(f))‖/‖f‖ = " << err_mstf << std::endl;
   deallog << std::scientific << "‖f- t^-1( t(f))‖/‖f‖ = " << err_mtf << std::endl;

   double tol = 1e-08;

   EXPECT_LT(err_mstq, tol);
   EXPECT_LT(err_mtq, tol);
   EXPECT_LT(err_mstg, tol);
   EXPECT_LT(err_mtg, tol);
   EXPECT_LT(err_mstf, tol);
   EXPECT_LT(err_mtf, tol);

   deallog << std::endl;
}

// tests whether dot_transform is consistent with dot
template<int dim>
void run_dot_transform_consistent_test(int fe_order, int quad_order, int refines, int n_steps,
      typename DiscretizedFunction<dim>::Norm norm) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   wavepi::util::GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad,
         triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;
   for (int i = 0; i < 10; i++) {
      DiscretizedFunction<dim> x = DiscretizedFunction<dim>::noise(mesh);
      x.set_norm(norm);

      DiscretizedFunction<dim> y = DiscretizedFunction<dim>::noise(mesh);
      y.set_norm(norm);

      double dot_xy = x*y;

      y.dot_transform();
      x.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);
      y.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);

      double dot_xy_via_transform = x*y;
      double err = std::abs(dot_xy - dot_xy_via_transform) / (std::abs(dot_xy) + 1e-300);

      deallog << std::scientific << "(x, y) = " << dot_xy << ", x^t T y = " << dot_xy_via_transform
            << ", rel. error = " << err << std::endl;

      double tol = 1e-10;
      EXPECT_LT(err, tol);
   }

   deallog << std::endl;
}

template<int dim>
void run_derivative_transpose_test(int fe_order, int quad_order, int refines, int n_steps) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   wavepi::util::GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad,
         triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

   for (int i = 0; i < 10; i++) {
      DiscretizedFunction<dim> x = DiscretizedFunction<dim>::noise(mesh);
      x.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);

      DiscretizedFunction<dim> y = DiscretizedFunction<dim>::noise(mesh);
      y.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);

      auto Dy = y.calculate_derivative();
      Dy.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);

      auto Dtx = x.calculate_derivative_transpose();
      Dtx.set_norm(DiscretizedFunction<dim>::Norm::Coefficients);

      double dot_x_Dy = x * Dy;
      double dot_Dtx_y = Dtx * y;
      double err = std::abs(dot_Dtx_y - dot_x_Dy) / (std::abs(dot_x_Dy) + 1e-300);

      deallog << std::scientific << "(x, Dy) = " << dot_x_Dy << ", (D^t x, y) = " << dot_Dtx_y
            << ", rel. error = " << err << std::endl;

      double tol = 1e-10;
      EXPECT_LT(err, tol);

      /* Test, whether M and D commute */
      //auto MDy = Dy;
      //MDy.mult_mass();
      //
      //auto DMy = y;
      //DMy.mult_mass();
      //DMy = DMy.calculate_derivative();
      //
      //deallog << "rerr(DMy,MDy) = " << MDy.relative_error(DMy) << std::endl;
   }

   deallog << std::endl;
}

}

template<int dim>
void run_dot_norm_tests(int fe_order, int quad_order, int refines, int n_steps) {
   deallog << "== norm=Vector ==" << std::endl;
   run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps,
         DiscretizedFunction<dim>::Norm::Coefficients);

   deallog << "== norm=L2L2 ==" << std::endl;
   run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::L2L2);

   deallog << "== norm=H1L2 ==" << std::endl;
   run_dot_norm_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::H1L2);
}

template<int dim>
void run_dot_transform_inverse_tests(int fe_order, int quad_order, int refines, int n_steps) {
   deallog << "== norm=Vector ==" << std::endl;
   run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps,
         DiscretizedFunction<dim>::Norm::Coefficients);

   deallog << "== norm=L2L2 ==" << std::endl;
   run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::L2L2);

   deallog << "== norm=H1L2 ==" << std::endl;
   run_dot_transform_inverse_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::H1L2);
}

template<int dim>
void run_dot_transform_consistent_tests(int fe_order, int quad_order, int refines, int n_steps) {
   deallog << "== norm=Vector ==" << std::endl;
   run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps,
         DiscretizedFunction<dim>::Norm::Coefficients);

   deallog << "== norm=L2L2 ==" << std::endl;
   run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::L2L2);

   deallog << "== norm=H1L2 ==" << std::endl;
   run_dot_transform_consistent_test<dim>(fe_order, quad_order, refines, n_steps, DiscretizedFunction<dim>::Norm::H1L2);
}

TEST(DiscretizedFunctionTest, DotNormConsistent1DFE1) {
   run_dot_norm_tests<1>(1, 3, 10, 128);
   run_dot_norm_tests<1>(1, 4, 9, 256);
}
TEST(DiscretizedFunctionTest, DotNormConsistent1DFE2) {
   run_dot_norm_tests<1>(2, 4, 7, 128);
   run_dot_norm_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunctionTest, DotNormConsistent2DFE1) {
   run_dot_norm_tests<2>(1, 3, 5, 128);
   run_dot_norm_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotNormConsistent2DFE2) {
   run_dot_norm_tests<2>(2, 4, 4, 128);
   run_dot_norm_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotNormConsistent3DFE1) {
   run_dot_norm_tests<3>(1, 3, 2, 32);
   run_dot_norm_tests<3>(1, 4, 1, 64);
}

TEST(DiscretizedFunctionTest, DerivativeTranspose1DFE1) {
   run_derivative_transpose_test<1>(1, 3, 10, 128);
   run_derivative_transpose_test<1>(1, 4, 9, 256);
}
TEST(DiscretizedFunctionTest, DerivativeTranspose1DFE2) {
   run_derivative_transpose_test<1>(2, 4, 7, 128);
   run_derivative_transpose_test<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunctionTest, DerivativeTranspose2DFE1) {
   run_derivative_transpose_test<2>(1, 3, 5, 128);
   run_derivative_transpose_test<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DerivativeTranspose2DFE2) {
   run_derivative_transpose_test<2>(2, 4, 4, 128);
   run_derivative_transpose_test<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DerivativeTranspose3DFE1) {
   run_derivative_transpose_test<3>(1, 3, 2, 32);
   run_derivative_transpose_test<3>(1, 4, 1, 64);
}

TEST(DiscretizedFunctionTest, DotTransformInverse1DFE1) {
   run_dot_transform_inverse_tests<1>(1, 3, 10, 128);
   run_dot_transform_inverse_tests<1>(1, 4, 9, 256);
}

TEST(DiscretizedFunctionTest, DotTransformInverse1DFE2) {
   run_dot_transform_inverse_tests<1>(2, 4, 7, 128);
   run_dot_transform_inverse_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunctionTest, DotTransformInverse2DFE1) {
   run_dot_transform_inverse_tests<2>(1, 3, 5, 128);
   run_dot_transform_inverse_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotTransformInverse2DFE2) {
   run_dot_transform_inverse_tests<2>(2, 4, 4, 128);
   run_dot_transform_inverse_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotTransformInverse3DFE1) {
   run_dot_transform_inverse_tests<3>(1, 3, 2, 32);
   run_dot_transform_inverse_tests<3>(1, 4, 3, 64);
}

TEST(DiscretizedFunctionTest, DotTransformConsistent1DFE1) {
   run_dot_transform_consistent_tests<1>(1, 3, 10, 128);
   run_dot_transform_consistent_tests<1>(1, 4, 9, 256);
}

TEST(DiscretizedFunctionTest, DotTransformConsistent1DFE2) {
   run_dot_transform_consistent_tests<1>(2, 4, 7, 128);
   run_dot_transform_consistent_tests<1>(2, 4, 7, 256);
}

TEST(DiscretizedFunctionTest, DotTransformConsistent2DFE1) {
   run_dot_transform_consistent_tests<2>(1, 3, 5, 128);
   run_dot_transform_consistent_tests<2>(1, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotTransformConsistent2DFE2) {
   run_dot_transform_consistent_tests<2>(2, 4, 4, 128);
   run_dot_transform_consistent_tests<2>(2, 4, 4, 256);
}

TEST(DiscretizedFunctionTest, DotTransformConsistent3DFE1) {
   run_dot_transform_consistent_tests<3>(1, 3, 2, 32);
   run_dot_transform_consistent_tests<3>(1, 4, 3, 64);
}

