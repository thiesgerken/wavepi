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
#include <inversion/WaveProblem.h>

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
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / (rho(p, this->get_time()) * 4.0);
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

         return p[0] + this->get_time();
      }
};

template<int dim>
class QLinearizedProblem: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
   public:
      virtual ~QLinearizedProblem() {
      }

      QLinearizedProblem(const WaveEquation<dim> &weq, const DiscretizedFunction<dim>& q,
            const DiscretizedFunction<dim>& u)
            : weq(weq), weq_adj(weq) {
         this->q = std::make_shared<DiscretizedFunction<dim>>(q);
         this->u = std::make_shared<DiscretizedFunction<dim>>(u);

         this->rhs = std::make_shared<L2ProductRightHandSide<dim>>(this->u, this->u);
         this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);

         this->weq.set_initial_values_u(this->weq.zero);
         this->weq.set_initial_values_v(this->weq.zero);
         this->weq.set_boundary_values_u(this->weq.zero);
         this->weq.set_boundary_values_v(this->weq.zero);

         this->weq.set_param_q(this->q);
         this->weq_adj.set_param_q(this->q);
      }

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h) {
         rhs->set_func1(std::make_shared<DiscretizedFunction<dim>>(h));
         weq.set_right_hand_side(rhs);

         DiscretizedFunction<dim> res = weq.run();
         res.throw_away_derivative();

         return res;
      }

      // L2 adjoint
      virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g) {
         rhs_adj->set_base_rhs(std::make_shared<DiscretizedFunction<dim>>(g));
         weq_adj.set_right_hand_side(rhs_adj);

         // L*
         DiscretizedFunction<dim> res = weq_adj.run();
         // DiscretizedFunction<dim> res = weq.run(true);
         res.throw_away_derivative();

         // M*
         res *= -1.0;
         res.pointwise_multiplication(*u);
         // a bit awkward: res has the right nodal values, but it should be a quadratic polynomial by now.

         return res;
      }

      virtual DiscretizedFunction<dim> zero() {
         return DiscretizedFunction<dim>(q->get_mesh(), q->get_dof_handler());
      }

      bool progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state) {
         deallog << "k=" << state.iteration_number << ": rdisc="
               << state.current_discrepancy / state.norm_data;
         deallog << ", norm=" << state.norm_current_estimate;
         deallog << std::endl;
         return true;
      }
   private:
      WaveEquation<dim> weq;
      WaveEquationAdjoint<dim> weq_adj;

      std::shared_ptr<DiscretizedFunction<dim>> q;
      std::shared_ptr<DiscretizedFunction<dim>> u;

      std::shared_ptr<L2ProductRightHandSide<dim>> rhs;
      std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
};

template<int dim>
class QProblem: public WaveProblem<dim> {
   public:
      virtual ~QProblem() {
      }

      QProblem(WaveEquation<dim>& weq)
            : WaveProblem<dim>(weq) {
      }

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u) {
         return std::make_unique<QLinearizedProblem<dim>>(this->wave_equation, q, u);
      }

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& q) {
         this->wave_equation.set_param_q(std::make_shared<DiscretizedFunction<dim>>(q));

         DiscretizedFunction<dim> res = this->wave_equation.run();
         res.throw_away_derivative();

         return res;
      }
};

template<int dim>
void test() {
   std::ofstream logout("wavepi_inverse.log");
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

   double t_start = 0.0, t_end = 2.0, dt = 1.0 / 256.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());

   TestQ<dim> q;
   auto q_exact = std::make_shared<DiscretizedFunction<dim>>(mesh, dof_handler, q);
   wave_eq.set_param_q(q_exact);

   deallog.push("generate_data");

   auto data_exact = wave_eq.run();
   data_exact.throw_away_derivative();

   double epsilon = 1e-2;
   auto data = DiscretizedFunction<dim>::noise(data_exact, epsilon * data_exact.norm());
   data.add(1.0, data_exact);

   deallog.pop();
   deallog.pop();

   // currently using same grids for parameters and solution
   DiscretizedFunction<dim> initialGuess(mesh, dof_handler);

   //   NonlinearLandweber<DiscretizedFunction<dim>, DiscretizedFunction<dim>> lw(std::make_unique<QProblem<dim>>(wave_eq), initialGuess, 5e1);
   //   lw.invert(data, 1.5 * epsilon * data_exact.norm(), &q_exact);

   REGINN<DiscretizedFunction<dim>, DiscretizedFunction<dim>> reginn(std::make_unique<QProblem<dim>>(wave_eq),
         std::make_unique<ConjugateGradients<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>(),
         initialGuess);
   reginn.invert(data, 2 * epsilon * data_exact.norm(), q_exact);

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
