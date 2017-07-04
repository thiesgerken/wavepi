/*
 * LinearRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LINEARREGULARIZATION_H_
#define INVERSION_LINEARREGULARIZATION_H_

#include <inversion/Regularization.h>
#include <inversion/LinearProblem.h>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template<typename Param, typename Sol>
class LinearRegularization: Regularization<Param, Sol> {
   public:
      LinearRegularization(LinearProblem<Param, Sol> problem)
            : Regularization(problem), problem(problem) {
      }

      virtual ~LinearRegularization() {
      }

      // virtual Param invert(Sol data, double targetDiscrepancy) = 0;

   protected:
      LinearProblem problem;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LINEARREGULARIZATION_H_ */
