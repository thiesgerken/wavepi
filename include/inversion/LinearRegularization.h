/*
 * LinearRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LINEARREGULARIZATION_H_
#define INVERSION_LINEARREGULARIZATION_H_

#include <inversion/LinearProblem.h>
#include <inversion/Regularization.h>

#include <memory>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template<typename Param, typename Sol>
class LinearRegularization: public Regularization<Param, Sol> {
   public:
      
      virtual ~LinearRegularization() = default;

      const std::shared_ptr<LinearProblem<Param, Sol>>& get_problem() const {
         return problem;
      }

      void set_problem(const std::shared_ptr<LinearProblem<Param, Sol> >& problem) {
         this->problem = problem;
      }

   protected:
      std::shared_ptr<LinearProblem<Param, Sol>> problem;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LINEARREGULARIZATION_H_ */
