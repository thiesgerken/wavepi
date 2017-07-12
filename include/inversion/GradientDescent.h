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

template<typename Param, typename Sol>
class GradientDescent: public LinearRegularization<Param, Sol> {
   public:

	GradientDescent(std::shared_ptr<LinearProblem<Param, Sol>> problem)
            : LinearRegularization<Param, Sol>(problem) {
      }

	GradientDescent() {
      }

      virtual ~GradientDescent() {
      }

     virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("Gradient");
         Assert(this->problem, ExcInternalError());

         Sol residual(data);
         double discrepancy = residual.norm();

         // TODO
 // Param estimate = this->problem->zero_param();
         Param estimate(data);
         estimate = 0.0;

         this->problem->progress(estimate, residual, data, 0, exact_param);

         for (int k = 1; discrepancy > target_discrepancy; k++) {
           Param step =  this->problem->adjoint(residual);
        		   Sol Astep = this->problem->forward(step);
        		      		   double omega = square(step.norm() / Astep.norm());

        		      		   estimate.add(omega, step);

            // calculate new residual and discrepancy for next step
            residual.add(-1.0 * omega, Astep);
            discrepancy = residual.norm();

            this->problem->progress(estimate, residual, data, k, exact_param);
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
