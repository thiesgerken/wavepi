/*
 * ConjugateGradients.h
 *
 *  Created on: 07.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_
#define INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <inversion/LinearProblem.h>
#include <inversion/LinearRegularization.h>

#include <memory>

namespace wavepi {
namespace inversion {

using namespace dealii;

// conjugate gradient method applied to normal equation (often called CGNR or CGLS)
// REGINN(CG) seems to diverge (especially if the time discretization is coarse)
// while REGINN(Gradient) and REGINN(Landweber) seem to work fine in those cases as well.
template<typename Param, typename Sol, typename Exact>
class ConjugateGradients: public LinearRegularization<Param, Sol, Exact> {
   public:

      virtual ~ConjugateGradients() = default;

      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
         LogStream::Prefix prefix("CGLS");
         AssertThrow(this->problem, ExcInternalError());

         Param estimate(this->problem->zero()); // f_k
         Sol residual(data); // r_k

         Param p(this->problem->adjoint(residual)); // p_{k+1}
         Param d(p); // d_k

         double norm_d = d.norm();
         double discrepancy = residual.norm();
         double norm_data = data.norm();

         InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
               target_discrepancy, &data, norm_data, exact_param, false);
         this->progress(status);

         for (int k = 1;
               discrepancy > target_discrepancy; k++) {
            Sol q(this->problem->forward(p)); // q_k

            double alpha = square(norm_d / q.norm()); // α_k

            {
               LogStream::Prefix pp("info");
               deallog << "α_k = " << alpha << std::endl;
            }

            if (alpha == 0.0)
               break;

            estimate.add(alpha, p);
            residual.add(-1.0 * alpha, q);
            discrepancy = residual.norm();

            status = InversionProgress<Param, Sol, Exact>(k, &estimate, estimate.norm(), &residual,
                  discrepancy, target_discrepancy, &data, norm_data, exact_param, false);

            if (!this->progress(status))
               break;

            // saves one evaluation of the adjoint if we are finished
            if (discrepancy <= target_discrepancy)
               break;

            d = this->problem->adjoint(residual);

            double norm_d_last = norm_d; // ‖d_{k-1}‖
            norm_d = d.norm();

            double beta = square(norm_d / norm_d_last); // β_k

            {
               LogStream::Prefix pp("info");
               deallog << "β_k = " << beta << std::endl;
            }

            p.sadd(beta, 1.0, d);
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
