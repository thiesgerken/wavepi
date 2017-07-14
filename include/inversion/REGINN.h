/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_REGINN_H_
#define INVERSION_REGINN_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>

#include <inversion/LinearRegularization.h>
#include <inversion/NewtonRegularization.h>

#include <memory>

namespace wavepi {
namespace inversion {
using namespace dealii;

template<typename Param, typename Sol>
class REGINN: public NewtonRegularization<Param, Sol> {
   public:
      REGINN(std::shared_ptr<NonlinearProblem<Param, Sol>> problem,
            std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver, const Param& initial_guess)
            : NewtonRegularization<Param, Sol>(problem), initial_guess(initial_guess), linear_solver(
                  linear_solver) {
      }

      REGINN(std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver, const Param& initial_guess)
            : initial_guess(initial_guess), linear_solver(linear_solver) {
      }

      virtual ~REGINN() {
      }

      using Regularization<Param, Sol>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) {
         LogStream::Prefix p = LogStream::Prefix("REGINN");
         deallog.push("init");

         Assert(this->problem, ExcInternalError());
         Assert(linear_solver, ExcInternalError());

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();
         double initial_discrepancy = discrepancy;
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         deallog.pop();
         InversionProgress<Param, Sol> status(0, &estimate, estimate.norm(), &residual, discrepancy, &data,
               norm_data, exact_param, norm_exact);
         this->problem->progress(status);

         for (int i = 1;
               discrepancy > target_discrepancy
                     && (!this->abort_discrepancy_doubles || discrepancy < 2 * initial_discrepancy)
                     && i <= this->max_iterations; i++) {

            // TODO
            double theta_n = 0.7;
            double linear_target_discrepancy = discrepancy * theta_n;

            auto linear_status = std::make_shared<InversionProgress<Param, Sol>>(status);
            linear_solver->set_problem(this->problem->derivative(estimate, data_current));
            Param step = linear_solver->invert(residual, linear_target_discrepancy, nullptr,
                  linear_status);

            if (linear_status->current_discrepancy > linear_target_discrepancy) {
               deallog << "error: linear solver did not converge to desired discrepancy!" << std::endl;
               break;
            }

            estimate += step;

            // calculate new residual and discrepancy
            deallog.push("post_step");
            residual = Sol(data);
            data_current = this->problem->forward(estimate);
            residual -= data_current;
            double discrepancy_last = discrepancy;
            discrepancy = residual.norm();

            deallog.pop();
            status = InversionProgress<Param, Sol>(i, &estimate, estimate.norm(), &residual, discrepancy, &data,
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
      const Param initial_guess;
      std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INVERSION_REGINN_H_ */
