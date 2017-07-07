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

#include <memory>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template<typename Param, typename Sol>
class NewtonRegularization: public Regularization<Param, Sol> {
   public:
      NewtonRegularization(std::shared_ptr<NonlinearProblem<Param, Sol>> problem)
            : problem(problem) {
      }

      NewtonRegularization()
            : problem() {
      }

      virtual ~NewtonRegularization() {
      }

      const std::shared_ptr<NonlinearProblem<Param, Sol> >& get_problem() const {
         return problem;
      }

      void set_problem(const std::shared_ptr<NonlinearProblem<Param, Sol> >& problem) {
         this->problem = problem;
      }

      // virtual Param invert(Sol data, double targetDiscrepancy) = 0;

   protected:
      std::shared_ptr<NonlinearProblem<Param, Sol>> problem;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NEWTONREGULARIZATION_H_ */
