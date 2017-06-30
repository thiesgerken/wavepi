#include <iostream>

#include "WaveEquation.h"

using namespace dealii;
using namespace wavepi;

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
   if ((this->get_time() <= 0.5) && (p.distance(Point<3>(1.0, 0.5,0.0)) < 0.4))
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
   std::ofstream logout("test.log");
   deallog.attach(logout);
   deallog.depth_console(2);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();
   // deallog.log_execution_time(true);

   Triangulation<dim> triangulation;

   GridGenerator::hyper_cube(triangulation, -5, 5);
   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   triangulation.refine_global(5);

   FE_Q<dim> fe(1);
   DoFHandler<dim> dof_handler;
   dof_handler.initialize(triangulation, fe);

   deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

   WaveEquation<dim> wave_eq(&dof_handler);

   TestF<dim> rhs;
   L2RightHandSide<dim> l2rhs(&rhs);
   wave_eq.set_right_hand_side(&l2rhs);

   wave_eq.set_time_end(6.0);
   wave_eq.set_theta(0.5);

   TestA<dim> a;
   wave_eq.set_param_a(&a);

   TestC<dim> c;
   wave_eq.set_param_c(&c);

   Timer timer;
   timer.start();
   DiscretizedFunction<dim> sol = wave_eq.run();
   timer.stop();
   deallog << "Continuous c, a: " << timer.wall_time() << " s of wall time" << std::endl;

//   DiscretizedFunction<dim> cdisc = DiscretizedFunction<dim>::discretize(&c, sol.get_times(),
//         sol.get_dof_handlers());
//   wave_eq.set_param_c(&cdisc);
//
//   timer.restart();
//   DiscretizedFunction<dim> sol2 = wave_eq.run();
//   timer.stop();
//   deallog << "Discrete c: " << timer.wall_time() << " s of wall time" << std::endl;
//
//   DiscretizedFunction<dim> adisc = DiscretizedFunction<dim>::discretize(&a, sol.get_times(),
//         sol.get_dof_handlers());
//   wave_eq.set_param_a(&adisc);
//   wave_eq.set_param_c(&c);
//
//   timer.restart();
//   DiscretizedFunction<dim> sol3 = wave_eq.run();
//   timer.stop();
//   deallog << "Discrete a: " << timer.wall_time() << " s of wall time" << std::endl;

//   sol.write_pvd("solution", "sol_u", "sol_v");
//   adisc.write_pvd("param_a", "param_a");

   deallog.timestamp();
}

int main() {
   try {
      test<2>();
   } catch (std::exception &exc) {
      std::cerr << std::endl << std::endl << "----------------------------------------------------"
            << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl
            << "Aborting!" << std::endl << "----------------------------------------------------"
            << std::endl;

      return 1;
   } catch (...) {
      std::cerr << std::endl << std::endl << "----------------------------------------------------"
            << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
            << "----------------------------------------------------" << std::endl;
      return 1;
   }

   return 0;
}
