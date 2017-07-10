/*
 * ConjugateGradients.h
 *
 *  Created on: 07.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_
#define INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_

#include <inversion/Regularization.h>
#include <inversion/LinearRegularization.h>
#include <inversion/LinearProblem.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <memory>

namespace wavepi {
namespace inversion {

using namespace dealii;

template<typename Param, typename Sol>
class ConjugateGradients: public LinearRegularization<Param, Sol> {
   public:

      ConjugateGradients(std::shared_ptr<LinearProblem<Param, Sol>> problem)
            : LinearRegularization<Param, Sol>(problem) {
      }

      ConjugateGradients() {
      }

      virtual ~ConjugateGradients() {
      }

      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("CG");
         Assert(this->problem, ExcInternalError());

         Sol r_k(data);

         Param p_k1 = this->problem->adjoint(r_k);
         Param d_k(p_k1);

         double norm_dk = d_k.norm();
         double discrepancy = r_k.norm();

         Param estimate(p_k1);
         estimate = 0.0;

         this->problem->progress(estimate, r_k, data, 0, exact_param);

         for (int k = 1; discrepancy > target_discrepancy; k++) {
            Sol q_k = this->problem->forward(p_k1);
            double alpha_k = square(norm_dk / q_k.norm());

            estimate.add(alpha_k, p_k1);

            r_k.add(-1.0 * alpha_k, q_k);

            discrepancy = r_k.norm();

            this->problem->progress(estimate, r_k, data, k, exact_param);

            // saves one evaluation of the adjoint if we are finished
            if (discrepancy < target_discrepancy)
               break;

            d_k = this->problem->adjoint(r_k);

            double norm_dkm1 = norm_dk;
            norm_dk = d_k.norm();

            double beta_k = square(norm_dk / norm_dkm1);
            p_k1.sadd(beta_k, 1.0, d_k);
         }

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
