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
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <inversion/ConjugateGradients.h>
#include <inversion/ConstantMaxIterChoice.h>
#include <inversion/ConstantToleranceChoice.h>
#include <inversion/FibonacciMaxIterChoice.h>
#include <inversion/GradientDescent.h>
#include <inversion/InversionProgress.h>
#include <inversion/Landweber.h>
#include <inversion/LinearRegularization.h>
#include <inversion/NewtonRegularization.h>
#include <inversion/NonlinearProblem.h>
#include <inversion/RiederToleranceChoice.h>

#include <iostream>
#include <memory>
#include <string>

namespace wavepi {
namespace inversion {
using namespace dealii;

template <typename Param, typename Sol, typename Exact>
class REGINN : public NewtonRegularization<Param, Sol, Exact> {
 public:
  virtual ~REGINN() = default;

  REGINN(std::shared_ptr<NonlinearProblem<Param, Sol>> problem, std::shared_ptr<Param> initial_guess,
         std::shared_ptr<LinearRegularization<Param, Sol, Exact>> linear_solver,
         std::shared_ptr<ToleranceChoice> tol_choice)
      : NewtonRegularization<Param, Sol, Exact>(problem),
        initial_guess(initial_guess),
        linear_solver(linear_solver),
        tol_choice(tol_choice) {}

  REGINN(std::shared_ptr<NonlinearProblem<Param, Sol>> problem, std::shared_ptr<Param> initial_guess,
         ParameterHandler &prm)
      : NewtonRegularization<Param, Sol, Exact>(problem), initial_guess(initial_guess) {
    get_parameters(prm);
  }

  static void declare_parameters(ParameterHandler &prm) {
    prm.enter_subsection("REGINN");
    {
      prm.declare_entry("linear solver", "ConjugateGradients",
                        Patterns::Selection("ConjugateGradients|GradientDescent|Landweber"),
                        "regularization method for the linear subproblems");
      prm.declare_entry("tolerance choice", "Rieder", Patterns::Selection("Rieder|Constant"),
                        "algorithm that chooses the target discrepancies for the linear subproblems");
      prm.declare_entry("maximum iteration choice", "Constant", Patterns::Selection("Fibonacci|Constant"),
                        "algorithm that chooses the maximum iteration counts for the linear subproblems");

      ToleranceChoice::declare_parameters(prm);
      RiederToleranceChoice::declare_parameters(prm);
      ConstantToleranceChoice::declare_parameters(prm);
      FibonacciMaxIterChoice::declare_parameters(prm);
      ConstantMaxIterChoice::declare_parameters(prm);
      Landweber<Param, Sol, Exact>::declare_parameters(prm);
      ConjugateGradients<Param, Sol, Exact>::declare_parameters(prm);
      InnerStatOutputProgressListener<Param, Sol, Exact>::declare_parameters(prm);
      WatchdogProgressListener<Param, Sol, Exact>::declare_parameters(prm, true, "linear watchdog", true);
    }
    prm.leave_subsection();
  }

  void get_parameters(ParameterHandler &prm) {
    prm.enter_subsection("REGINN");
    {
      std::string slinear_solver = prm.get("linear solver");

      if (slinear_solver == "ConjugateGradients")
        linear_solver = std::make_shared<ConjugateGradients<Param, Sol, Exact>>(prm);
      else if (slinear_solver == "GradientDescent")
        linear_solver = std::make_shared<GradientDescent<Param, Sol, Exact>>();
      else if (slinear_solver == "Landweber")
        linear_solver = std::make_shared<Landweber<Param, Sol, Exact>>(prm);
      else
        AssertThrow(false, ExcInternalError());

      inner_stats = std::make_shared<InnerStatOutputProgressListener<Param, Sol, Exact>>(prm);

      // only add stat output for the master node in case of MPI
      if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) linear_solver->add_listener(inner_stats);

      inner_watchdog = std::make_shared<WatchdogProgressListener<Param, Sol, Exact>>(prm, true, "linear watchdog");
      linear_solver->add_listener(inner_watchdog);

      linear_solver->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol, Exact>>("k"));
      linear_solver->add_listener(std::make_shared<CtrlCProgressListener<Param, Sol, Exact>>());

      std::string stol_choice = prm.get("tolerance choice");

      if (stol_choice == "Rieder")
        tol_choice = std::make_shared<RiederToleranceChoice>(prm);
      else if (stol_choice == "Constant")
        tol_choice = std::make_shared<ConstantToleranceChoice>(prm);
      else
        AssertThrow(false, ExcInternalError());

      std::string smax_iter_choice = prm.get("maximum iteration choice");

      if (smax_iter_choice == "Fibonacci")
        max_iter_choice = std::make_shared<FibonacciMaxIterChoice>(prm);
      else if (smax_iter_choice == "Constant")
        max_iter_choice = std::make_shared<ConstantMaxIterChoice>(prm);
      else
        AssertThrow(false, ExcInternalError());
    }
    prm.leave_subsection();
  }

  virtual Param invert(const Sol &data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
                       std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
    LogStream::Prefix p = LogStream::Prefix("REGINN");
    deallog.push("init");

    AssertThrow(problem, ExcInternalError());
    AssertThrow(linear_solver, ExcInternalError());
    AssertThrow(tol_choice, ExcInternalError());
    AssertThrow(max_iter_choice, ExcInternalError());

    Param estimate(*initial_guess);

    Sol residual(data);
    Sol data_current = this->problem->forward(estimate);
    residual -= data_current;

    double discrepancy = residual.norm();
    double norm_data   = data.norm();

    tol_choice->reset(target_discrepancy, discrepancy);

    deallog.pop();
    InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
                                                target_discrepancy, &data, norm_data, exact_param, false);
    this->progress(status);

    for (int i = 1; discrepancy > target_discrepancy; i++) {
      double theta        = tol_choice->get_tolerance();
      int linear_max_iter = max_iter_choice->get_max_iter();

      deallog << "Solving linear problem using rtol=" << theta << " and max_iter=" << linear_max_iter << std::endl;

      auto linear_status = std::make_shared<InversionProgress<Param, Sol, Exact>>(status);
      linear_solver->set_problem(this->problem->derivative(estimate));
      inner_stats->set_iteration(i);
      inner_watchdog->set_max_iter(linear_max_iter);

      double linear_target_discrepancy = discrepancy * theta;
      Param step                       = linear_solver->invert(residual, linear_target_discrepancy, linear_status);

      if (linear_status->current_discrepancy > linear_target_discrepancy) {
        deallog << "Linear solver did not achieve desired discrepancy";

        if (linear_status->iteration_number >= linear_max_iter && linear_max_iter > 0)
          deallog << " due to max_iter." << std::endl;
        else {
          deallog << "." << std::endl << "  Aborting REGINN!" << std::endl;
          break;
        }
      }

      estimate += step;
      double norm_estimate = estimate.norm();

      // post-processing
      deallog.push("post_processing");
      this->post_process(i, &estimate, norm_estimate);
      deallog.pop();

      // calculate new residual and discrepancy
      deallog.push("finish_step");
      residual     = data;
      data_current = this->problem->forward(estimate);
      residual -= data_current;
      discrepancy = residual.norm();
      deallog.pop();

      status = InversionProgress<Param, Sol, Exact>(i, &estimate, norm_estimate, &residual, discrepancy,
                                                    target_discrepancy, &data, norm_data, exact_param, false);

      if (!this->progress(status)) break;

      tol_choice->add_iteration(discrepancy, linear_status->iteration_number);
      max_iter_choice->add_iteration(discrepancy, linear_status->iteration_number);

      deallog << std::endl;
    }

    status.finished = true;
    this->progress(status);

    if (status_out) *status_out = status;

    return estimate;
  }

 private:
  using NewtonRegularization<Param, Sol, Exact>::problem;

  std::shared_ptr<Param> initial_guess;
  std::shared_ptr<LinearRegularization<Param, Sol, Exact>> linear_solver;
  std::shared_ptr<ToleranceChoice> tol_choice;
  std::shared_ptr<MaxIterChoice> max_iter_choice;
  std::shared_ptr<InnerStatOutputProgressListener<Param, Sol, Exact>> inner_stats;
  std::shared_ptr<WatchdogProgressListener<Param, Sol, Exact>> inner_watchdog;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INVERSION_REGINN_H_ */
