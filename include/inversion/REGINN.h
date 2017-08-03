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

#include <inversion/InverseProblem.h>
#include <inversion/LinearRegularization.h>
#include <inversion/NewtonRegularization.h>
#include <inversion/NonlinearProblem.h>
#include <inversion/Regularization.h>
#include <inversion/ToleranceChoice.h>

#include <iostream>
#include <memory>

namespace wavepi {
namespace inversion {
using namespace dealii;

template<typename Param, typename Sol>
class REGINN: public NewtonRegularization<Param, Sol> {
   public:
      REGINN(std::shared_ptr<NonlinearProblem<Param, Sol>> problem,
            std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver,
            std::shared_ptr<ToleranceChoice> tol_choice, const Param& initial_guess)
            : NewtonRegularization<Param, Sol>(problem), initial_guess(initial_guess), linear_solver(
                  linear_solver), tol_choice(tol_choice) {
      }

      virtual ~REGINN() {
      }

      using Regularization<Param, Sol>::invert;

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) {
         LogStream::Prefix p = LogStream::Prefix("REGINN");
         deallog.push("init");

         AssertThrow(problem, ExcInternalError());
         AssertThrow(linear_solver, ExcInternalError());
         AssertThrow(tol_choice, ExcInternalError());

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();
         double initial_discrepancy = discrepancy;
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         tol_choice->reset(target_discrepancy, discrepancy);

         deallog.pop();
         InversionProgress<Param, Sol> status(0, &estimate, estimate.norm(), &residual, discrepancy, &data,
               norm_data, exact_param, norm_exact);
         this->progress(status);

         for (int i = 1;
               discrepancy > target_discrepancy
                     && (!this->abort_discrepancy_doubles || discrepancy < 2 * initial_discrepancy)
                     && i <= this->max_iterations; i++) {

            double theta = tol_choice->get_tolerance();
            double linear_target_discrepancy = discrepancy * theta;
            deallog << "solving linear problem using rtol=" << theta << std::endl;

            auto linear_status = std::make_shared<InversionProgress<Param, Sol>>(status);
            linear_solver->set_problem(this->problem->derivative(estimate, data_current));
            Param step = linear_solver->invert(residual, linear_target_discrepancy, nullptr, linear_status);

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
            status = InversionProgress<Param, Sol>(i, &estimate, estimate.norm(), &residual, discrepancy,
                  &data, norm_data, exact_param, norm_exact);

            if (!this->progress(status))
               break;

            if (discrepancy_last < discrepancy && this->abort_increasing_discrepancy)
               break;

            tol_choice->add_iteration(discrepancy, linear_status->iteration_number);
         }

         if (status_out)
            *status_out = status;

         return estimate;
      }

   private:
      using NewtonRegularization<Param, Sol>::problem;

      const Param initial_guess;
      std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver;
      std::shared_ptr<ToleranceChoice> tol_choice;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INVERSION_REGINN_H_ */
