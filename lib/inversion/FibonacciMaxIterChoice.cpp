/*
 * FibonacciMaxIterChoice.cpp
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#include <inversion/FibonacciMaxIterChoice.h>

namespace wavepi {
namespace inversion {

FibonacciMaxIterChoice::FibonacciMaxIterChoice(int initial_max_iter)
      : initial_max_iter(initial_max_iter) {
}

FibonacciMaxIterChoice::FibonacciMaxIterChoice(ParameterHandler &prm) {
   get_parameters(prm);
}

void FibonacciMaxIterChoice::declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection("FibonacciMaxIterChoice");
   {
      prm.declare_entry("initial max iter", "1", Patterns::Integer(),
            "maximum iteration count for the first inner iteration");
   }
   prm.leave_subsection();
}

void FibonacciMaxIterChoice::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("FibonacciMaxIterChoice");
   {
      initial_max_iter = prm.get_integer("initial max iter");
   }
   prm.leave_subsection();
}

int FibonacciMaxIterChoice::calculate_max_iter() const {
   if (required_steps.size() == 0)
      return initial_max_iter;

   if (required_steps.size() == 1)
      return 2 * initial_max_iter;

   return required_steps[required_steps.size() - 1]
         + required_steps[required_steps.size() - 2];
}

} /* namespace inversion */
} /* namespace wavepi */
