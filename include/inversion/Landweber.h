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
      Landweber(double omega)
            : omega(omega) {
      }

      virtual ~Landweber() {
      }

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("Landweber");
         Assert(this->problem, ExcInternalError());

         Param estimate = this->problem->zero();
         Sol residual(data);

         double discrepancy = residual.norm();
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         deallog.pop();
         this->problem->progress(
               InversionProgress(0, estimate, estimate.norm(), residual, discrepancy, data, norm_data,
                     exact_param, norm_exact));

         for (int k = 1; discrepancy > target_discrepancy; k++) {
            Param adj = this->problem->adjoint(residual);

            // $`c_{k+1} = c_k + \omega A^* (g - A c_k)`$
            estimate.add(omega, adj);

            // calculate new residual and discrepancy for next step
            residual = Sol(data);
            Sol data_current = this->problem->forward(estimate);
            residual -= data_current;
            discrepancy = residual.norm();

            if (!this->problem->progress(
                  InversionProgress(k, estimate, estimate.norm(), residual, discrepancy, data, norm_data,
                        exact_param, norm_exact)))
               break;
         }

         return estimate;
      }

   private:
      double omega;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
