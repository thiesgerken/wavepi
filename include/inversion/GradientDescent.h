/*
 * GradientDescent.h
 *
 *  Created on: 07.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_GRADIENTDESCENT_H_
#define INCLUDE_INVERSION_GRADIENTDESCENT_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <inversion/LinearProblem.h>
#include <inversion/LinearRegularization.h>

#include <memory>

namespace wavepi {
namespace inversion {

using namespace dealii;

template<typename Param, typename Sol, typename Exact>
class GradientDescent: public LinearRegularization<Param, Sol, Exact> {
   public:

      virtual ~GradientDescent() = default;

//      GradientDescent() {
//         // should generate decreasing residuals
//         this->abort_discrepancy_doubles = true;
//         this->abort_increasing_discrepancy = true;
//      }

      using Regularization<Param, Sol, Exact>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
         LogStream::Prefix p = LogStream::Prefix("Gradient");
         AssertThrow(this->problem, ExcInternalError());

         Param estimate(this->problem->zero());
         Sol residual(data);

         double discrepancy = residual.norm();
         double norm_data = data.norm();

         InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
               target_discrepancy, &data, norm_data, exact_param, false);
         this->progress(status);

         for (int k = 1; discrepancy > target_discrepancy; k++) {
            Param step(this->problem->adjoint(residual));
            Sol Astep(this->problem->forward(step));

            double omega = square(step.norm() / Astep.norm());

            // deallog << "omega = " << omega << std::endl;
            estimate.add(omega, step);

            // calculate new residual and discrepancy for next step
            residual.add(-1.0 * omega, Astep);
            discrepancy = residual.norm();

            status = InversionProgress<Param, Sol, Exact>(k, &estimate, estimate.norm(), &residual,
                  discrepancy, target_discrepancy, &data, norm_data, exact_param, false);

            if (!this->progress(status))
               break;
         }

         status.finished = true;
         this->progress(status);

         if (status_out)
            *status_out = status;

         return estimate;
      }

   private:

      static inline double square(const double x) {
         return x * x;
      }
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_ */
