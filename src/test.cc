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
   if ((this->get_time() <= 0.5) && (p.distance(Point<2>(0.5, 0.5)) < 0.2 || p.distance(Point<2>(2.5, 2.5)) < 0.2))
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

   double v = 0.1 + 2*p[1] + p[0];
   return 1 / (v*v);
}

int main() {
   try {

      WaveEquation<2> wave_eq;

      TestF<2> rhs;
      wave_eq.right_hand_side = &rhs;

      wave_eq.time_end = 5.0;
      wave_eq.theta = 0.5;

      TestC<2> c;
      wave_eq.param_c = &c;

      wave_eq.run();
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
