/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LANDWEBER_H_
#define INVERSION_LANDWEBER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <inversion/LinearProblem.h>
#include <inversion/LinearRegularization.h>

#include <memory>

namespace wavepi {
namespace inversion {
using namespace dealii;

// linear Landweber iteration
template<typename Param, typename Sol>
class Landweber: public LinearRegularization<Param, Sol> {
   public:
      Landweber(std::shared_ptr<LinearProblem<Param, Sol>> problem, const Param& initial_guess, double omega)
            : LinearRegularization<Param, Sol>(problem), omega(omega), initial_guess(initial_guess) {
      }

      Landweber(const Param& initial_guess, double omega)
            : omega(omega), initial_guess(initial_guess) {
      }

      virtual ~Landweber() {
      }

      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("Landweber");
         Assert(this->problem, ExcInternalError());

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();

         this->problem->progress(estimate, residual, data, 0, exact_param);

         for (int k = 1; discrepancy > target_discrepancy; k++) {
            Param adj = this->problem->adjoint(residual);

            // $`c_{k+1} = c_k + \omega A^* (g - A c_k)`$
            estimate.add(omega, adj);

            // calculate new residual and discrepancy for next step
            residual = Sol(data);
            data_current = this->problem->forward(estimate);
            residual -= data_current;
            discrepancy = residual.norm();

            this->problem->progress(estimate, residual, data, k, exact_param);
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
