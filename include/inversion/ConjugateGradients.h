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
template<typename Param, typename Sol>
class ConjugateGradients: public LinearRegularization<Param, Sol> {
   public:

      ConjugateGradients() {
         // cgls should generate decreasing residuals
         this->abort_discrepancy_doubles = true;
         this->abort_increasing_discrepancy = true;
      }

      virtual ~ConjugateGradients() {
      }

      using Regularization<Param, Sol>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) {
         LogStream::Prefix prefix("CGLS");
         AssertThrow(this->problem, ExcInternalError());

         Param estimate(this->problem->zero()); // f_k
         Sol residual(data); // r_k

         Param p(this->problem->adjoint(residual)); // p_{k+1}
         Param d(p); // d_k

         double norm_d = d.norm();
         double discrepancy = residual.norm();
         double initial_discrepancy = discrepancy;
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         InversionProgress<Param, Sol> status(0, &estimate, estimate.norm(), &residual, discrepancy, &data,
               norm_data, exact_param, norm_exact);
         this->progress(status);

         for (int k = 1;
               discrepancy > target_discrepancy
                     && (!this->abort_discrepancy_doubles || discrepancy < 2 * initial_discrepancy)
                     && k <= this->max_iterations; k++) {
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

            double discrepancy_last = discrepancy;
            discrepancy = residual.norm();

            status = InversionProgress<Param, Sol>(k, &estimate, estimate.norm(), &residual, discrepancy,
                  &data, norm_data, exact_param, norm_exact);

            if (!this->progress(status))
               break;

            if (discrepancy_last < discrepancy && this->abort_increasing_discrepancy)
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
