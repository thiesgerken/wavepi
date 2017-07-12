/*
 * wave_equation.cpp
 *
 *  Created on: 11.07.2017
 *      Author: thies
 */

#include <gtest/gtest.h>

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

#include <bits/std_abs.h>
#include <stddef.h>
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

namespace {

using namespace dealii;
using namespace wavepi::forward;

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
class TestG: public Function<dim> {
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

         return p[0] * this->get_time();
      }
};

template<int dim>
class TestQ: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p.norm() < 0.3 ? this->get_time() : 0.0;
      }
};

template<int dim>
class DiscretizedFunctionDisguise: public Function<dim> {
   public:
      DiscretizedFunctionDisguise(std::shared_ptr<DiscretizedFunction<dim>> base)
            : base(base) {
      }

      double value(const Point<dim> &p, const unsigned int component = 0) const {
         return base->value(p, component);
      }

      void set_time(const double new_time) {
         Function<dim>::set_time(new_time);
         base->set_time(new_time);
      }
   private:
      std::shared_ptr<DiscretizedFunction<dim>> base;
};

// checks, whether the matrix assembly of discretized parameters works correct
// (by supplying DiscretizedFunctions and DiscretizedFunctionDisguises)
template<int dim>
void run_discretized_test(int fe_order, int quad_order, int refines) {
   std::ofstream logout("wavepi_test.log");
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();

   Timer timer;

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   double t_start = 0.0, t_end = 2.0, dt = 1.0 / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << std::endl << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler->n_dofs() << std::endl;
   deallog << "Number of time steps: " << times.size() << std::endl << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

   /* continuous */

   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));

   timer.restart();
   DiscretizedFunction<dim> sol_cont = wave_eq.run();
   timer.stop();
   deallog << "continuous params: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_cont.norm(), 0.0);

   /* discretized */

   TestC<dim> c;
   auto c_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, c);
   wave_eq.set_param_c(c_disc);

   TestA<dim> a;
   auto a_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, a);
   wave_eq.set_param_a(a_disc);

   TestQ<dim> q;
   auto q_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, q);
   wave_eq.set_param_q(q_disc);

   TestNu<dim> nu;
   auto nu_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, nu);
   wave_eq.set_param_nu(nu_disc);

   TestF<dim> f;
   auto f_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, f);
   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f_disc));

   timer.restart();
   DiscretizedFunction<dim> sol_disc = wave_eq.run();
   timer.stop();
   deallog << "all discretized: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc.norm(), 0.0);

   /* discretized, q disguised */

   auto c_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(c_disc);
   auto a_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(a_disc);
   auto q_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(q_disc);
   auto nu_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(nu_disc);
   auto f_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(f_disc);

   wave_eq.set_param_q(q_disguised);

   timer.restart();
   DiscretizedFunction<dim> sol_disc_except_q = wave_eq.run();
   timer.stop();
   deallog << "all discretized, q disguised: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc_except_q.norm(), 0.0);

   /* discretized, a disguised */

   wave_eq.set_param_a(a_disguised);
   wave_eq.set_param_q(q_disc);

   timer.restart();
   DiscretizedFunction<dim> sol_disc_except_a = wave_eq.run();
   timer.stop();
   deallog << "all discretized, a disguised: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc_except_a.norm(), 0.0);

   /* disguised */

   wave_eq.set_param_nu(nu_disguised);
   wave_eq.set_param_q(q_disguised);
   wave_eq.set_param_a(a_disguised);
   wave_eq.set_param_c(c_disguised);
   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f_disguised));

   timer.restart();
   DiscretizedFunction<dim> sol_disguised = wave_eq.run();
   timer.stop();
   deallog << "all discretized and disguised as continuous: " << std::fixed << timer.wall_time() << " s of wall time"
         << std::endl << std::endl;
   EXPECT_GT(sol_disguised.norm(), 0.0);

   /* results */

   DiscretizedFunction<dim> tmp(sol_cont);
   tmp -= sol_disc;
   double err_cont_vs_disc = tmp.norm() / sol_cont.norm();

   deallog << "rel. error between continuous and full discrete: " << std::scientific << err_cont_vs_disc << std::endl;
   EXPECT_LT(err_cont_vs_disc, 1.0);

   tmp.sadd(0.0, 1.0, sol_disc_except_q);
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_q = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (q disguised): " << std::scientific
         << err_disguised_vs_disc_except_q << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_q, 1e-7);

   tmp.sadd(0.0, 1.0, sol_disc_except_a);
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_a = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (a disguised): " << std::scientific
         << err_disguised_vs_disc_except_a << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_a, 1e-7);

   tmp.sadd(0.0, 1.0, sol_disc);
   tmp -= sol_disguised;
   double err_disguised_vs_disc = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and full discrete: " << std::scientific << err_disguised_vs_disc
         << std::endl << std::endl;
   EXPECT_LT(err_disguised_vs_disc, 1e-7);
}

// checks, whether the matrix assembly of discretized parameters works correct
// (by supplying DiscretizedFunctions and DiscretizedFunctionDisguises)
template<int dim>
void run_l2adjoint_test(int fe_order, int quad_order, int refines) {
   std::ofstream logout("wavepi_test.log");
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   double t_start = 0.0, t_end = 2.0, dt = 1.0 / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << std::endl << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler->n_dofs() << std::endl;
   deallog << "Number of time steps: " << times.size() << std::endl << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

   /* continuous */

   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   TestF<dim> f_cont;
   auto f = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, f_cont);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
   DiscretizedFunction<dim> sol_f = wave_eq.run();
   EXPECT_GT(sol_f.norm(), 0.0);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
   DiscretizedFunction<dim> adj_f = wave_eq.run(true);
   EXPECT_GT(adj_f.norm(), 0.0);

   TestG<dim> g_cont;
   auto g = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, g_cont);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g));
   DiscretizedFunction<dim> sol_g = wave_eq.run();
   EXPECT_GT(sol_g.norm(), 0.0);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g));
   DiscretizedFunction<dim> adj_g = wave_eq.run(true);
   EXPECT_GT(adj_g.norm(), 0.0);

   double dot_solf_f = sol_f * (*f);
   double dot_f_adjf = (*f) * adj_f;
   double ff_err = std::abs(dot_solf_f - dot_f_adjf) / std::abs(dot_solf_f);
   EXPECT_LT(ff_err, 1e-2);

   deallog << std::scientific << "(Lf, f) = " << dot_solf_f << ", (f, L*f) = " << dot_f_adjf << ", rel. error = "
         << ff_err << std::endl;

   double dot_solg_g = sol_g * (*g);
   double dot_g_adjg = (*g) * adj_g;
   double gg_err = std::abs(dot_solg_g - dot_g_adjg) / std::abs(dot_solg_g);
   EXPECT_LT(gg_err, 1e-2);

   deallog << std::scientific << "(Lg, g) = " << dot_solg_g << ", (g, L*g) = " << dot_g_adjg << ", rel. error = "
         << gg_err << std::endl;

   double dot_solg_f = sol_g * (*f);
   double dot_g_adjf = (*g) * adj_f;
   double gf_err = std::abs(dot_solg_f - dot_g_adjf) / std::abs(dot_solg_f);
   EXPECT_LT(gf_err, 1e-2);

   deallog << std::scientific << "(Lg, f) = " << dot_solg_f << ", (g, L*f) = " << dot_g_adjf << ", rel. error = "
         << gf_err << std::endl;

   double dot_solf_g = sol_f * (*g);
   double dot_f_adjg = (*f) * adj_g;
   double fg_err = std::abs(dot_solf_g - dot_f_adjg) / std::abs(dot_solf_g);
   EXPECT_LT(fg_err, 1e-2);

   deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg << ", rel. error = "
         << fg_err << std::endl << std::endl;
}

}

TEST(WaveEquationTest, DiscretizedParameters1DFE1) {
   run_discretized_test<1>(1, 3, 8);
}

TEST(WaveEquationTest, DiscretizedParameters1DFE2) {
   run_discretized_test<1>(2, 4, 8);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE1) {
   run_discretized_test<2>(1, 3, 4);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE2) {
   run_discretized_test<2>(2, 4, 4);
}

TEST(WaveEquationTest, DiscretizedParameters3DFE1) {
   run_discretized_test<3>(1, 3, 2);
}

TEST(WaveEquationTest, DiscretizedParameters3DFE2) {
   run_discretized_test<3>(2, 4, 1);
}

TEST(WaveEquationTest, L2Adjointness1DFE1) {
   run_l2adjoint_test<1>(1, 3, 8);
   run_l2adjoint_test<1>(1, 3, 10);
}

TEST(WaveEquationTest, L2Adjointness1DFE2) {
   run_l2adjoint_test<1>(2, 4, 8);
   run_l2adjoint_test<1>(1, 3, 10);
}

TEST(WaveEquationTest, L2Adjointness2DFE1) {
   run_l2adjoint_test<2>(1, 3, 4);
   run_l2adjoint_test<2>(1, 3, 6);
}

TEST(WaveEquationTest, L2Adjointness2DFE2) {
   run_l2adjoint_test<2>(2, 4, 4);
   run_l2adjoint_test<2>(2, 4, 6);
}

TEST(WaveEquationTest, L2Adjointness3DFE1) {
   run_l2adjoint_test<3>(1, 3, 2);
}

TEST(WaveEquationTest, L2Adjointness3DFE2) {
   run_l2adjoint_test<3>(2, 4, 2);
}
