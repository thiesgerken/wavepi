/*
 * wavepi_direct.cpp
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

#include <stddef.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

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
   if ((this->get_time() <= 0.5) && (p.distance(Point<1>(1.0)) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<2>::value(const Point<2> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5) && (p.distance(Point<2>(1.0, 0.5)) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<3>::value(const Point<3> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5) && (p.distance(Point<3>(1.0, 0.5, 0.0)) < 0.4))
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
void test() {
   std::ofstream logout("wavepi_direct.log");
   deallog.attach(logout);
   deallog.depth_console(3);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();
   // deallog.log_execution_time(true);

   Triangulation<dim> triangulation;

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(triangulation, -5, 5);
   triangulation.refine_global(4);

   // QGauss<dim>(n) is exact in polynomials of degree <= 2n-1 (needed: fe_order*3)
   // -> fe_order*3 <= 2n-1  ==>  n >= (fe_order*3+1)/2
   const int fe_order = 1;
   const int quad_order = std::ceil((fe_order * 3 + 1.0) / 2.0);
   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);

   auto dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler->n_dofs() << std::endl;

   double t_start = 0.0, t_end = 6.0, dt = 1.0 / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);
   WaveEquationAdjoint<dim> wave_eq(mesh, dof_handler, quad);

   wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());

   Timer timer;
   timer.start();
   DiscretizedFunction<dim> sol = wave_eq.run();
   timer.stop();
   deallog << "solving took " << timer.wall_time() << " s of wall time" << std::endl;

   deallog << "L2L2-norm of solution: " << std::scientific << sol.norm() <<  std::endl;

   sol.set_norm(DiscretizedFunction<dim>::L2L2_Vector);
   deallog << "L2L2-norm of solution (lumped mass, different scaling): " << std::scientific << sol.norm() <<  std::endl;

   deallog.timestamp();
}

int main() {
   try {
      test<2>();
   } catch (std::exception &exc) {
      std::cerr << std::endl << std::endl;
      std::cerr << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl;
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
