/*
 * wave_equation.cpp
 *
 *  Created on: 11.07.2017
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
   std::ofstream logout("wavepi_test.log", std::ios_base::app);
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(0);
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

   double t_start = 0.0, t_end = 2.0, dt = t_end / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << "n_dofs: " << dof_handler->n_dofs();
   deallog << ", n_steps: " << times.size() << std::endl;

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
   deallog << "all discretized, q disguised: " << std::fixed << timer.wall_time() << " s of wall time"
         << std::endl;
   EXPECT_GT(sol_disc_except_q.norm(), 0.0);

   /* discretized, a disguised */

   wave_eq.set_param_a(a_disguised);
   wave_eq.set_param_q(q_disc);

   timer.restart();
   DiscretizedFunction<dim> sol_disc_except_a = wave_eq.run();
   timer.stop();
   deallog << "all discretized, a disguised: " << std::fixed << timer.wall_time() << " s of wall time"
         << std::endl;
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
   deallog << "all discretized and disguised as continuous: " << std::fixed << timer.wall_time()
         << " s of wall time" << std::endl << std::endl;
   EXPECT_GT(sol_disguised.norm(), 0.0);

   /* results */
   DiscretizedFunction<dim> tmp(sol_disc_except_q);
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_q = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (q disguised): " << std::scientific
         << err_disguised_vs_disc_except_q << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_q, 1e-7);

   tmp = sol_disc_except_a;
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_a = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (a disguised): " << std::scientific
         << err_disguised_vs_disc_except_a << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_a, 1e-7);

   tmp = sol_disc;
   tmp -= sol_disguised;
   double err_disguised_vs_disc = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and full discrete: " << std::scientific
         << err_disguised_vs_disc << std::endl << std::endl;
   EXPECT_LT(err_disguised_vs_disc, 1e-7);
}

template<int dim>
void run_l2adjoint_back_test(int fe_order, int quad_order, int refines, int n_steps) {
   std::ofstream logout("wavepi_test.log", std::ios_base::app);
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(0);
   deallog.precision(3);
   deallog.pop();

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << "n_dofs: " << dof_handler->n_dofs();
   deallog << ", n_steps: " << times.size() << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

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
   double ff_err = std::abs(dot_solf_f - dot_f_adjf) / (std::abs(dot_solf_f) + 1e-300);

   deallog << std::scientific << "(Lf, f) = " << dot_solf_f << ", (f, L*f) = " << dot_f_adjf
         << ", rel. error = " << ff_err << std::endl;

   double dot_solg_g = sol_g * (*g);
   double dot_g_adjg = (*g) * adj_g;
   double gg_err = std::abs(dot_solg_g - dot_g_adjg) / (std::abs(dot_solg_g) + 1e-300);

   deallog << std::scientific << "(Lg, g) = " << dot_solg_g << ", (g, L*g) = " << dot_g_adjg
         << ", rel. error = " << gg_err << std::endl;

   double dot_solg_f = sol_g * (*f);
   double dot_g_adjf = (*g) * adj_f;
   double gf_err = std::abs(dot_solg_f - dot_g_adjf) / (std::abs(dot_solg_f) + 1e-300);

   deallog << std::scientific << "(Lg, f) = " << dot_solg_f << ", (g, L*f) = " << dot_g_adjf
         << ", rel. error = " << gf_err << std::endl;

   double dot_solf_g = sol_f * (*g);
   double dot_f_adjg = (*f) * adj_g;
   double fg_err = std::abs(dot_solf_g - dot_f_adjg) / (std::abs(dot_solf_g) + 1e-300);

   deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
         << ", rel. error = " << fg_err << std::endl;

   EXPECT_LT(ff_err, 1e-2);
   EXPECT_LT(gg_err, 1e-2);
   EXPECT_LT(gf_err, 1e-2);
   EXPECT_LT(fg_err, 1e-2);
}

template<int dim>
void run_l2adjoint_test(int fe_order, int quad_order, int refines, int n_steps) {
   std::ofstream logout("wavepi_test.log", std::ios_base::app);
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(0);
   deallog.precision(3);
   deallog.pop();

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << "n_dofs: " << dof_handler->n_dofs();
   deallog << ", n_steps: " << times.size() << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);

   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);
   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   WaveEquationAdjoint<dim> wave_eq_adj(mesh, dof_handler, quad);
   wave_eq_adj.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq_adj.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq_adj.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq_adj.set_param_nu(std::make_shared<TestNu<dim>>());

   TestF<dim> f_cont;
   auto f = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, f_cont);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
   DiscretizedFunction<dim> sol_f = wave_eq.run();
   EXPECT_GT(sol_f.norm(), 0.0);

   wave_eq_adj.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
   DiscretizedFunction<dim> adj_f = wave_eq_adj.run();
   EXPECT_GT(adj_f.norm(), 0.0);

   TestG<dim> g_cont;
   auto g = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, g_cont);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g));
   DiscretizedFunction<dim> sol_g = wave_eq.run();
   EXPECT_GT(sol_g.norm(), 0.0);

   wave_eq_adj.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g));
   DiscretizedFunction<dim> adj_g = wave_eq_adj.run();
   EXPECT_GT(adj_g.norm(), 0.0);

   double dot_solf_f = sol_f * (*f);
   double dot_f_adjf = (*f) * adj_f;
   double ff_err = std::abs(dot_solf_f - dot_f_adjf) / (std::abs(dot_solf_f) + 1e-300);

   deallog << std::scientific << "(Lf, f) = " << dot_solf_f << ", (f, L*f) = " << dot_f_adjf
         << ", rel. error = " << ff_err << std::endl;

   double dot_solg_g = sol_g * (*g);
   double dot_g_adjg = (*g) * adj_g;
   double gg_err = std::abs(dot_solg_g - dot_g_adjg) / (std::abs(dot_solg_g) + 1e-300);

   deallog << std::scientific << "(Lg, g) = " << dot_solg_g << ", (g, L*g) = " << dot_g_adjg
         << ", rel. error = " << gg_err << std::endl;

   double dot_solg_f = sol_g * (*f);
   double dot_g_adjf = (*g) * adj_f;
   double gf_err = std::abs(dot_solg_f - dot_g_adjf) / (std::abs(dot_solg_f) + 1e-300);

   deallog << std::scientific << "(Lg, f) = " << dot_solg_f << ", (g, L*f) = " << dot_g_adjf
         << ", rel. error = " << gf_err << std::endl;

   double dot_solf_g = sol_f * (*g);
   double dot_f_adjg = (*f) * adj_g;
   double fg_err = std::abs(dot_solf_g - dot_f_adjg) / (std::abs(dot_solf_g) + 1e-300);

   deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
         << ", rel. error = " << fg_err << std::endl;

   EXPECT_LT(ff_err, 1e-2);
   EXPECT_LT(gg_err, 1e-2);
   EXPECT_LT(gf_err, 1e-2);
   EXPECT_LT(fg_err, 1e-2);
}

// product of sines in space to have dirichlet b.c. in [0,pi], times a sum of sine and cosine in time.
// its time derivative is the same function with C[1] = C[0]*norm(k), C[0] = -C[1]*norm(k)
template<int dim>
class SeparationAnsatz: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         double res = 1;

         for (size_t i = 0; i < dim; i++)
            res *= std::sin(k[i] * p[i]);

         res *= constants[0] * std::sin(std::sqrt(k.square()) * this->get_time())
               + constants[1] * std::cos(std::sqrt(k.square()) * this->get_time());
         return res;
      }

      SeparationAnsatz(Point<dim, int> k, Point<2> constants)
            : k(k), constants(constants) {
      }

   private:
      Point<dim, int> k;
      Point<2> constants;
};

template<int dim>
void run_reference_test(int fe_order, int quad_order, int refines, Point<dim, int> k, Point<2> constants,
      double t_end, int steps, bool expect = true) {
   std::ofstream logout("wavepi_test.log", std::ios_base::app);
   deallog.attach(logout);
   deallog.depth_console(0);
   deallog.depth_file(0);
   deallog.precision(3);
   deallog.pop();

   Triangulation<dim> triangulation;
   GridGenerator::hyper_cube(triangulation, 0, numbers::PI);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   double t_start = 0.0, dt = t_end / steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << "n_dofs: " << dof_handler->n_dofs();
   deallog << ", n_steps: " << times.size() << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

   Point<2> derivative_constants;
   derivative_constants[0] = -constants[1] * std::sqrt(k.square());
   derivative_constants[1] = constants[0] * std::sqrt(k.square());

   auto u = std::make_shared<SeparationAnsatz<dim>>(k, constants);
   auto v = std::make_shared<SeparationAnsatz<dim>>(k, derivative_constants);

   wave_eq.set_initial_values_u(u);
   wave_eq.set_initial_values_v(v);

   DiscretizedFunction<dim> solu = wave_eq.run();
   DiscretizedFunction<dim> solv = solu.derivative();
   solu.throw_away_derivative();

   DiscretizedFunction<dim> refu(mesh, dof_handler, *u);
   DiscretizedFunction<dim> refv(mesh, dof_handler, *v);

   DiscretizedFunction<dim> tmp(solu);
   tmp -= refu;
   double err_u = tmp.norm() / refu.norm();

   tmp = solv;
   tmp -= refv;
   double err_v = tmp.norm() / refv.norm();

   if (expect) {
      EXPECT_LT(err_u, 1e-1);
      EXPECT_LT(err_v, 1e-1);
   }

   deallog << std::scientific << "forward : rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl;

   solu = wave_eq.run(true);
   solv = solu.derivative();
   solu.throw_away_derivative();

   refu = DiscretizedFunction<dim>(mesh, dof_handler, *u);
   refv = DiscretizedFunction<dim>(mesh, dof_handler, *v);

   tmp = solu;
   tmp -= refu;
   err_u = tmp.norm() / refu.norm();

   tmp = solv;
   tmp -= refv;
   err_v = tmp.norm() / refv.norm();

   if (expect) {
      EXPECT_LT(err_u, 1e-1);
      EXPECT_LT(err_v, 1e-1);
   }

   deallog << std::scientific << "backward: rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl
         << std::endl;
}

}

TEST(WaveEquationTest, DiscretizedParameters1DFE1) {
   run_discretized_test<1>(1, 3, 8);
}

TEST(WaveEquationTest, DiscretizedParameters1DFE2) {
   run_discretized_test<1>(2, 4, 8);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE1) {
   run_discretized_test<2>(1, 3, 3);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE2) {
   run_discretized_test<2>(2, 4, 3);
}

TEST(WaveEquationTest, DiscretizedParameters3DFE1) {
   run_discretized_test<3>(1, 3, 1);
}

TEST(WaveEquationTest, L2AdjointnessBack1DFE1) {
   run_l2adjoint_back_test<1>(1, 3, 8, 64);
   run_l2adjoint_back_test<1>(1, 3, 8, 256);
}

TEST(WaveEquationTest, L2AdjointnessBack1DFE2) {
   run_l2adjoint_back_test<1>(2, 4, 7, 64);
   run_l2adjoint_back_test<1>(2, 4, 7, 256);
}

TEST(WaveEquationTest, L2AdjointnessBack2DFE1) {
   run_l2adjoint_back_test<2>(1, 3, 4, 16);
   run_l2adjoint_back_test<2>(1, 3, 4, 64);
   run_l2adjoint_back_test<2>(1, 3, 4, 256);
}

TEST(WaveEquationTest, L2AdjointnessBack2DFE2) {
   run_l2adjoint_back_test<2>(2, 4, 3, 64);
   run_l2adjoint_back_test<2>(2, 4, 3, 256);
}

TEST(WaveEquationTest, L2AdjointnessBack3DFE1) {
   run_l2adjoint_back_test<3>(1, 3, 2, 64);
   run_l2adjoint_back_test<3>(1, 3, 2, 256);
}

TEST(WaveEquationTest, L2Adjointness1DFE1) {
   run_l2adjoint_test<1>(1, 3, 8, 64);
   run_l2adjoint_test<1>(1, 3, 8, 256);
}

TEST(WaveEquationTest, L2Adjointness1DFE2) {
   run_l2adjoint_test<1>(2, 4, 7, 64);
   run_l2adjoint_test<1>(2, 4, 7, 256);
}

TEST(WaveEquationTest, L2Adjointness2DFE1) {
   run_l2adjoint_test<2>(1, 3, 4, 4);
   run_l2adjoint_test<2>(1, 3, 4, 16);
   run_l2adjoint_test<2>(1, 3, 4, 64);
   run_l2adjoint_test<2>(1, 3, 4, 256);
   run_l2adjoint_test<2>(1, 3, 4, 512);
   run_l2adjoint_test<2>(1, 3, 4, 1024);
}

TEST(WaveEquationTest, L2Adjointness2DFE2) {
   run_l2adjoint_test<2>(2, 4, 3, 64);
   run_l2adjoint_test<2>(2, 4, 3, 256);
}

TEST(WaveEquationTest, L2Adjointness3DFE1) {
   run_l2adjoint_test<3>(1, 3, 2, 64);
   run_l2adjoint_test<3>(1, 3, 2, 256);
}

TEST(WaveEquationTest, ReferenceTest1DFE1) {
   for (int steps = 128; steps <= 1024; steps *= 2)
      run_reference_test<1>(1, 3, 10, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 9; refine >= 1; refine--)
      run_reference_test<1>(1, 3, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 1024, false);
}

TEST(WaveEquationTest, ReferenceTest1DFE2) {
   for (int steps = 16; steps <= 128; steps *= 2)
      run_reference_test<1>(2, 4, 7, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 6; refine >= 1; refine--)
      run_reference_test<1>(2, 4, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 128, false);
}

TEST(WaveEquationTest, ReferenceTest2DFE1) {
   for (int steps = 16; steps <= 256; steps *= 2)
      run_reference_test<2>(1, 3, 6, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 5; refine >= 1; refine--)
      run_reference_test<2>(1, 3, refine, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, 256,
            false);
}

TEST(WaveEquationTest, ReferenceTest3DFE1) {
   for (int steps = 8; steps <= 32; steps *= 2)
      run_reference_test<3>(1, 3, 3, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, steps,
            steps >= 32);

   for (int refine = 2; refine >= 0; refine--)
      run_reference_test<3>(1, 3, refine, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, 32,
            false);
}
