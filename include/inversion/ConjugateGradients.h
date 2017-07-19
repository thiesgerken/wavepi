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
         // cg should generate decreasing residuals
         this->abort_discrepancy_doubles = true;
         this->abort_increasing_discrepancy = true;
      }

      virtual ~ConjugateGradients() {
      }

      using Regularization<Param, Sol>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) {
         LogStream::Prefix p = LogStream::Prefix("CG");
         Assert(this->problem, ExcInternalError());


         Param estimate(this->problem->zero());
         Sol r_k(data);

         Param p_k1(this->problem->adjoint(r_k));
         Param d_k(p_k1);

         double norm_dk = d_k.norm();
         double discrepancy = r_k.norm();
         double initial_discrepancy = discrepancy;
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         InversionProgress<Param, Sol> status(0, &estimate, estimate.norm(), &r_k, discrepancy, &data,
               norm_data, exact_param, norm_exact);
         this->problem->progress(status);

         for (int k = 1;
               discrepancy > target_discrepancy
                     && (!this->abort_discrepancy_doubles || discrepancy < 2 * initial_discrepancy)
                     && k <= this->max_iterations; k++) {
            Sol q_k(this->problem->forward(p_k1));

            double alpha_k = square(norm_dk / q_k.norm());
            // deallog << "norm(d_k) = " << norm_dk << std::endl;
            // deallog << "norm(q_k) = " << q_k.norm() << std::endl;

            estimate.add(alpha_k, p_k1);
            r_k.add(-1.0 * alpha_k, q_k);

            double discrepancy_last = discrepancy;
            discrepancy = r_k.norm();

            status = InversionProgress<Param, Sol>(k, &estimate, estimate.norm(), &r_k, discrepancy, &data,
                  norm_data, exact_param, norm_exact);

            if (!this->problem->progress(status))
               break;

            if (discrepancy_last < discrepancy && this->abort_increasing_discrepancy)
               break;

            // saves one evaluation of the adjoint if we are finished
            if (discrepancy <= target_discrepancy)
               break;

            d_k = this->problem->adjoint(r_k);
            // deallog << "norm of d_k = " << d_k.norm() << std::endl;

            double norm_dkm1 = norm_dk;
            norm_dk = d_k.norm();

            double beta_k = square(norm_dk / norm_dkm1);
            // deallog << "beta_k = " << beta_k << std::endl;

            p_k1.sadd(beta_k, 1.0, d_k);
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
