#include <iostream>

#include "WaveEquation.h"
  using namespace dealii;
  using namespace wavepi;

  template <int dim>
  class RHS : public Function<dim>
  {
  public:
        RHS () : Function<dim>() {}
     double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;
  };

template <int dim>
double RHS<dim>::value (const Point<dim> &p,
                                   const unsigned int component) const
{
 (void) component;
 Assert(component == 0, ExcIndexRange(component, 0, 1));
 if ((this->get_time() <= 0.5))
   return std::sin (this->get_time() * 4 * numbers::PI);
 else
   return 0;
}

int main() {
   try {

      WaveEquation<2> wave_eq;
      RHS<2> rhs;
      wave_eq.right_hand_side = &rhs;

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
