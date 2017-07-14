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

      virtual Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<const Param> exact_param) {
         LogStream::Prefix p = LogStream::Prefix("REGINN");
         deallog.push("init");

         Assert(this->problem, ExcInternalError());
         Assert(linear_solver, ExcInternalError());

         Param estimate(initial_guess);

         Sol residual(data);
         Sol data_current = this->problem->forward(estimate);
         residual -= data_current;

         double discrepancy = residual.norm();
         double norm_data = data.norm();
         double norm_exact = exact_param ? exact_param->norm() : -0.0;

         deallog.pop();
         this->problem->progress(
               InversionProgress(0, estimate, estimate.norm(), residual, discrepancy, data, norm_data,
                     exact_param, norm_exact));

         for (int i = 1; discrepancy > target_discrepancy; i++) {
            double theta_n = 0.7; // TODO

            linear_solver->set_problem(this->problem->derivative(estimate, data_current));
            Param step = linear_solver->invert(residual, discrepancy * theta_n, nullptr);
            estimate += step;

            // calculate new residual and discrepancy
            deallog.push("post_step");
            residual = Sol(data);
            data_current = this->problem->forward(estimate);
            residual -= data_current;
            discrepancy = residual.norm();

            deallog.pop();
            if (!this->problem->progress(
                  InversionProgress(i, estimate, estimate.norm(), residual, discrepancy, data, norm_data,
                        exact_param, norm_exact)))
               break;
         }

         return estimate;
      }

   private:
      const Param initial_guess;
      std::shared_ptr<LinearRegularization<Param, Sol>> linear_solver;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INVERSION_REGINN_H_ */
