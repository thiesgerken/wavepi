/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LANDWEBER_H_
#define INVERSION_LANDWEBER_H_

#include <inversion/Regularization.h>
#include <inversion/NewtonRegularization.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

namespace wavepi {
namespace inversion {

// nonlinear Landweber iteration (it can be regarded as an inexact newton method,
// applying one linear landweber step to the linearized problem to 'solve' it)
template<typename Param, typename Sol>
class Landweber: public NewtonRegularization<Param, Sol> {
   public:
      Landweber(NonlinearProblem<Param, Sol> *problem, const Param& initial_guess, double omega)
            : NewtonRegularization<Param, Sol>(problem), omega(omega), initial_guess(initial_guess) {
      }

      virtual ~Landweber() {
      }

      virtual Param invert(const Sol& data, double targetDiscrepancy, const Param* exact_param) {
         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();

         for (int i = 0; discrepancy > targetDiscrepancy; i++) {
            std::unique_ptr<LinearProblem<Param, Sol>>
            lp = this->problem->derivative(estimate, data_current);

            Param adj = lp->adjoint(residual);

            // $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
            estimate.add(omega, adj);

            // calculate new residual and discrepancy for next step
            residual = Sol(data);
            data_current = this->problem->forward(estimate);
            residual -= data_current;
            discrepancy = residual.norm();

            this->problem->progress(estimate, residual, data, i, exact_param);
         }

         return estimate;
      }

   private:
      double omega;
      const Param initial_guess;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
