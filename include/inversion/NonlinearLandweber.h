/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NONLINEARLANDWEBER_H_
#define INVERSION_NONLINEARLANDWEBER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <inversion/NewtonRegularization.h>
#include <inversion/NonlinearProblem.h>

#include <memory>

namespace wavepi {
namespace inversion {

using namespace dealii;

// nonlinear Landweber iteration (it can be regarded as an inexact newton method,
// applying one linear landweber step to the linearized problem to 'solve' it)
template<typename Param, typename Sol>
class NonlinearLandweber: public NewtonRegularization<Param, Sol> {
   public:
      NonlinearLandweber(std::shared_ptr<NonlinearProblem<Param, Sol>> problem, const Param& initial_guess,
            double omega)
            : NewtonRegularization<Param, Sol>(problem), omega(omega), initial_guess(initial_guess) {
      }

      NonlinearLandweber(const Param& initial_guess, double omega)
            : omega(omega), initial_guess(initial_guess) {
      }

      virtual ~NonlinearLandweber() {
      }

      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("Landweber");
         Assert(this->problem, ExcInternalError());
         deallog.push("init");

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();

         deallog.pop();
         this->problem->progress(estimate, residual, data, 0, exact_param);

         for (int i = 1; discrepancy > target_discrepancy; i++) {
            std::unique_ptr<LinearProblem<Param, Sol>> lp = this->problem->derivative(estimate, data_current);

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
