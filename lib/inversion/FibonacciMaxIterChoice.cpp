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
      prm.declare_entry("initial max iter", "1", Patterns::Integer(), "maximum iteration count for the first inner iteration");
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
   if (previous_max_iters.size() == 0)
   return initial_max_iter;
   if (previous_max_iters.size() == 1)
     return 2*initial_max_iter;

   return previous_max_iters[previous_max_iters.size() - 1] + previous_max_iters[previous_max_iters.size() - 2];
}

} /* namespace inversion */
} /* namespace wavepi */
