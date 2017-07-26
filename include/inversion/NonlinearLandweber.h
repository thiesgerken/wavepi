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

      using Regularization<Param, Sol>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) {
         LogStream::Prefix p = LogStream::Prefix("Landweber");
         AssertThrow(this->problem, ExcInternalError());
         deallog.push("init");

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();
         double initial_discrepancy = discrepancy;
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         deallog.pop();
         InversionProgress<Param, Sol> status(0, estimate, estimate.norm(), residual, discrepancy, data,
               norm_data, exact_param, norm_exact);
         this->problem->progress(status);

         for (int i = 1;
               discrepancy > target_discrepancy
                     && (!this->abort_discrepancy_doubles || discrepancy < 2 * initial_discrepancy)
                     && i <= this->max_iterations; i++) {
            std::unique_ptr<LinearProblem<Param, Sol>> lp = this->problem->derivative(estimate, data_current);

            Param adj = lp->adjoint(residual);

            // $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
            estimate.add(omega, adj);

            // calculate new residual and discrepancy for next step
            residual = Sol(data);
            data_current = this->problem->forward(estimate);
            residual -= data_current;
            double discrepancy_last = discrepancy;
            discrepancy = residual.norm();

            status = InversionProgress<Param, Sol>(i, estimate, estimate.norm(), residual, discrepancy, data,
                  norm_data, exact_param, norm_exact);

            if (!this->problem->progress(status))
               break;

            if (discrepancy_last < discrepancy && this->abort_increasing_discrepancy)
               break;
         }

         if (status_out)
            *status_out = status;

         return estimate;
      }

   private:
      double omega;
      const Param initial_guess;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
