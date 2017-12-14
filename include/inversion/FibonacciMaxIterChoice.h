/*
 * FibonacciMaxIterChoice.h
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#ifndef LIB_INVERSION_FIBONACCIMAXITERCHOICE_H_
#define LIB_INVERSION_FIBONACCIMAXITERCHOICE_H_

#include <deal.II/base/parameter_handler.h>
#include <inversion/MaxIterChoice.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

/**
 * Implements the max iter choice i^n_max = i^{n-1} + i^n, starting with given i^0_max (defaults to 1) and i^1_max = 2*i^0_max.
 * Idea is due to Winkler („A Model-Aware Inexact Newton Scheme for Electrical Impedance Tomography“, 2016)
 * I call it FibonacciMaxIterChoice because in the worst case the iteration counts are identical to the Fibonacci sequence.
 */
class FibonacciMaxIterChoice: public MaxIterChoice {
   public:
      FibonacciMaxIterChoice(int initial_max_iter);
      FibonacciMaxIterChoice(ParameterHandler &prm);

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

   protected:
      virtual int calculate_max_iter() const;

   private:
      int initial_max_iter = 1;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* LIB_INVERSION_CONSTANTMAXITERCHOICE_H_ */
