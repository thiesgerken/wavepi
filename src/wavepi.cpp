/*
 * wavepi_inverse.cpp
 *
 *  Created on: 01.07.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <forward/ConstantMesh.h>
#include <forward/DiscretizedFunction.h>
#include <forward/L2ProductRightHandSide.h>
#include <forward/L2RightHandSide.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/ConjugateGradients.h>
#include <inversion/GradientDescent.h>
#include <inversion/Landweber.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/REGINN.h>
#include <inversion/ToleranceChoice.h>
#include <inversion/RiederToleranceChoice.h>
#include <inversion/ConstantToleranceChoice.h>

#include <problems/L2QProblem.h>
#include <problems/L2CProblem.h>
#include <problems/L2NuProblem.h>
#include <problems/L2AProblem.h>

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::problems;

template<int dim>
class TestNu: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p[0] * this->get_time();
      }
};

template<int dim>
class TestF: public Function<dim> {
   public:
      TestF()
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

template<> const Point<1> TestF<1>::actor_position = Point<1>(1.0);
template<> const Point<2> TestF<2>::actor_position = Point<2>(1.0, 0.5);
template<> const Point<3> TestF<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template<int dim>
double rho(const Point<dim> &p, double t);

template<>
double rho(const Point<1> &p, double t) {
   return p.distance(Point<1>(t - 3.0)) < 1.0 ? 1.5 : 1.0;
}

template<>
double rho(const Point<2> &p, double t) {
   return p.distance(Point<2>(t - 3.0, t - 2.0)) < 1.2 ? 1.5 : 1.0;
}

template<>
double rho(const Point<3> &p, double t) {
   return p.distance(Point<3>(t - 3.0, t - 2.0, 0.0)) < 1.2 ? 1.5 : 1.0;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / (rho(p, this->get_time()) * 1.0);
      }
};

template<int dim>
class TestA: public Function<dim> {
   public:
      TestA()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / rho(p, this->get_time());
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

         return p.distance(q_position) < 1.0 ? 10 * std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
      }

      static const Point<dim> q_position;
};

template<> const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<> const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<> const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

template<int dim>
void test() {
   std::ofstream logout("wavepi.log");
   deallog.attach(logout);
   deallog.depth_console(2);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();
   deallog.push("init");
   // deallog.log_execution_time(true);

   Triangulation<dim> triangulation;

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(triangulation, -5, 5);
   triangulation.refine_global(4);

   // QGauss<dim>(n) is exact in polynomials of degree <= 2n-1 (needed: fe_order*3)
   // -> fe_order*3 <= 2n-1  ==>  n >= (fe_order*3+1)/2
   const int fe_order = 1;
   const int quad_order = std::max((int) std::ceil((fe_order * 3 + 1.0) / 2.0), 3);
   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   deallog << "fe_order = " << fe_order << std::endl;
   deallog << "quad_order = " << quad_order << std::endl;

   deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler->n_dofs() << std::endl;

   double t_start = 0.0, t_end = 2.0, dt = 1.0 / 128.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   deallog << "Number of time steps: " << times.size() << std::endl;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);

   if (dim == 1)
      mesh->set_boundary_ids(std::vector<types::boundary_id> { 0, 1 });

   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   using Param = DiscretizedFunction<dim>;
   using Sol = DiscretizedFunction<dim>;

   std::shared_ptr<NonlinearProblem<Param, Sol>> problem;
   std::shared_ptr<Function<dim>> param_exact_cont;
   std::shared_ptr<Param> param_exact;
   Param initialGuess(mesh, dof_handler);

   /* Reconstruct TestQ */
//   param_exact_cont = std::make_shared<TestQ<dim>>();
//   param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
//   wave_eq.set_param_q(param_exact);
//   problem = std::make_shared<L2QProblem<dim>>(wave_eq);
//   initialGuess = 0;
   /* Reconstruct TestC */
//   param_exact_cont = std::make_shared<TestC<dim>>();
//   param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
//   wave_eq.set_param_c(param_exact);
//   problem = std::make_shared<L2CProblem<dim>>(wave_eq);
//   initialGuess = 1;
   /* Reconstruct TestNu */
//   param_exact_cont = std::make_shared<TestNu<dim>>();
//   param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
//   wave_eq.set_param_nu(param_exact);
//   problem = std::make_shared<L2NuProblem<dim>>(wave_eq);
//   initialGuess = 0;
   /* Reconstruct TestA */
   param_exact_cont = std::make_shared<TestA<dim>>();
   param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
   wave_eq.set_param_a(param_exact);
   problem = std::make_shared<L2AProblem<dim>>(wave_eq);
   initialGuess = 1;

   deallog.push("generate_data");

   auto data_exact = wave_eq.run();
   data_exact.throw_away_derivative();
   data_exact.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);

   double epsilon = 1e-3;
   auto data = DiscretizedFunction<dim>::noise(data_exact, epsilon * data_exact.norm());
   data.add(1.0, data_exact);

   deallog.pop();
   deallog.pop();

   auto linear_solver = std::make_shared<ConjugateGradients<Param, Sol>>();
   linear_solver->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol>>("k"));

   auto tol_choice = std::make_shared<RiederToleranceChoice>(0.7, 0.95, 0.9, 1.0);

   REGINN<Param, Sol> reginn(problem, linear_solver, tol_choice, initialGuess);
   reginn.add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol>>("i"));

   reginn.invert(data, 2 * epsilon * data_exact.norm(), param_exact);

   deallog.timestamp();
}

int main() {
   try {
      test<2>();
   } catch (std::exception &exc) {
      std::cerr << std::endl << std::endl;
      std::cerr << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!"
            << std::endl;
      std::cerr << "----------------------------------------------------" << std::endl;

      return 1;
   } catch (...) {
      std::cerr << std::endl << std::endl;
      std::cerr << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl;
      std::cerr << "----------------------------------------------------" << std::endl;
      return 1;
   }

   return 0;
}
