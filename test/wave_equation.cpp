/*
 * discretized_params.cpp
 *
 *  Created on: 11.07.2017
 *      Author: thies
 */

#include "gtest/gtest.h"

#include <forward/WaveEquation.h>

namespace {

using namespace dealii;
using namespace wavepi::forward;

template<int dim>
class TestF: public Function<dim> {
   public:
      TestF()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<>
double TestF<1>::value(const Point<1> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 2) && (p.norm() < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<2>::value(const Point<2> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 2) && (p.norm() < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<3>::value(const Point<3> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 2) && (p.norm() < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<int dim>
double rho(const Point<dim> &p, double t);

template<>
double rho(const Point<1> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<1>(t - 3.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<>
double rho(const Point<2> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<2>(t - 3.0, t - 2.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<>
double rho(const Point<3> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<3>(t - 3.0, t - 2.0, 0.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<int dim>
double TestC<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   return 1.0 / (rho(p, this->get_time()) * 4.0);
}

template<int dim>
class TestA: public Function<dim> {
   public:
      TestA()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<int dim>
double TestA<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   return 1.0 / rho(p, this->get_time());
}

template<int dim>
class DiscretizedFunctionDisguise: public Function<dim> {
   public:
      DiscretizedFunctionDisguise(std::shared_ptr<DiscretizedFunction<dim>> base)
            : base(base) {
      }

      double value(const Point<dim> &p, const unsigned int component = 0) const {
         return base->value(p, component);
      }

      Tensor<1, dim, double> gradient(const Point<dim> &p, const unsigned int component) const {
         return base->gradient(p, component);
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

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(triangulation, -1, 1);
   triangulation.refine_global(refines);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

   DoFHandler<dim> dof_handler;
   dof_handler.initialize(triangulation, fe);

   double t_start = 0.0, t_end = 2.0, dt = 1.0 / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << std::endl << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;
   deallog << "Number of time steps: " << times.size() << std::endl << std::endl;

   WaveEquation<dim> wave_eq(&dof_handler, times, quad);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));

   /* continuous */

   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());

   timer.restart();
   DiscretizedFunction<dim> sol_cont = wave_eq.run();
   timer.stop();
   deallog << "continuous params: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;

   /* discretized */

   TestC<dim> q;
   auto c_disc = std::make_shared<DiscretizedFunction<dim>>(q, times, &dof_handler);
   wave_eq.set_param_c(c_disc);

   TestA<dim> a;
   auto a_disc = std::make_shared<DiscretizedFunction<dim>>(a, times, &dof_handler);
   wave_eq.set_param_a(a_disc);

   timer.restart();
   DiscretizedFunction<dim> sol_disc = wave_eq.run();
   timer.stop();
   deallog << "discretized params: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;

   /* disguised */

   auto c_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(c_disc);
   wave_eq.set_param_c(c_disguised);

   auto a_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(a_disc);
   wave_eq.set_param_a(a_disguised);

   timer.restart();
   DiscretizedFunction<dim> sol_disguised = wave_eq.run();
   timer.stop();
   deallog << "disguised discretized params: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl
         << std::endl;

   /* results */

   DiscretizedFunction<dim> tmp(sol_cont);
   tmp -= sol_disc;
   double err_cont_vs_disc = tmp.norm() / sol_cont.norm();

   deallog << "rel. error between continuous and discrete parameters: " << std::scientific << err_cont_vs_disc
         << std::endl;
   EXPECT_LT(err_cont_vs_disc, 1.0);

   tmp.sadd(0.0, 1.0, sol_disc);
   tmp -= sol_disguised;
   double err_disguised_vs_disc = tmp.norm() / sol_disc.norm();

   deallog << "rel. error between disguised discrete and discrete parameters: " << std::scientific
         << err_disguised_vs_disc << std::endl << std::endl;
   EXPECT_LT(err_disguised_vs_disc, 1e-7);
}
}

TEST(WaveEquationTest, DiscretizedParameters1D) {
   run_discretized_test<1>(1, 3, 8);
}

TEST(WaveEquationTest, DiscretizedParameters2D) {
   run_discretized_test<2>(1, 3, 4);
}

TEST(WaveEquationTest, DiscretizedParameters3D) {
   run_discretized_test<3>(1, 3, 2);
}

TEST(WaveEquationTest, L2Adjointness) {
   EXPECT_EQ(0, 0); // TODO
}
