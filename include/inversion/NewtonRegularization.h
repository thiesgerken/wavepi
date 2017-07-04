/*
 * NewtonRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NEWTONREGULARIZATION_H_
#define INVERSION_NEWTONREGULARIZATION_H_

#include <inversion/Regularization.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template<typename Param, typename Sol>
class NewtonRegularization: public Regularization<Param, Sol> {
   public:
      NewtonRegularization(NonlinearProblem<Param, Sol>* problem)
            : Regularization<Param, Sol>(problem), problem(problem) {
      }

      virtual ~NewtonRegularization() {
      }

      // virtual Param invert(Sol data, double targetDiscrepancy) = 0;

   protected:
      NonlinearProblem<Param, Sol> *problem;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NEWTONREGULARIZATION_H_ */
