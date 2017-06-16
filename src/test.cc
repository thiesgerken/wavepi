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

template<int dim>
double TestF<dim>::value(const Point<dim> &p, const unsigned int component) const {
   (void) component;
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5)
         && (p.distance(Point<2>(0.5, 0.5)) < 0.2 || p.distance(Point<2>(2.5, 2.5)) < 0.2))
      return 1000 * std::sin(this->get_time() * 4 * numbers::PI);
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
double TestC<dim>::value(const Point<dim> &p, const unsigned int component) const {
   (void) component;
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   double v = 1 + 2 * p[1] + p[0];
   return 1 / (v * v);
}

const int dim = 2;

template<int dim>
void test() {
   Triangulation<dim> triangulation;

   // GridGenerator::hyper_cube(triangulation, -1, 1);
   GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   triangulation.refine_global(4);

   FE_Q<dim> fe(1);
   DoFHandler<dim> dof_handler;
   dof_handler.initialize(triangulation, fe);

   std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl << std::endl;

   WaveEquation<dim> wave_eq(&dof_handler);

   TestF<dim> rhs;
   wave_eq.right_hand_side = &rhs;

   wave_eq.time_end = 5.0;
   wave_eq.theta = 0.5;

   TestC<dim> c;
   wave_eq.param_c = &c;

   DiscretizedFunction<dim> sol = wave_eq.run();
  sol.write_pvd("solution", "sol_u", "sol_v");
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
