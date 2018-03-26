/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NONLINEARLANDWEBER_H_
#define INVERSION_NONLINEARLANDWEBER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <inversion/NewtonRegularization.h>
#include <inversion/NonlinearProblem.h>

#include <memory>

namespace wavepi {
namespace inversion {

using namespace dealii;

// nonlinear Landweber iteration (it can be regarded as an inexact newton method,
// applying one linear landweber step to the linearized problem to 'solve' it)
template <typename Param, typename Sol, typename Exact>
class NonlinearLandweber : public NewtonRegularization<Param, Sol, Exact> {
 public:
  virtual ~NonlinearLandweber() = default;

  static void declare_parameters(ParameterHandler &prm) {
    prm.enter_subsection("NonlinearLandweber");
    {
      prm.declare_entry("omega", "1", Patterns::Double(0), "relaxation factor ω");
      prm.declare_entry("p", "2", Patterns::Double(1), "Use duality mappings with index p");
    }
    prm.leave_subsection();
  }

  void get_parameters(ParameterHandler &prm) {
    prm.enter_subsection("NonlinearLandweber");
    {
      omega = prm.get_double("omega");
      p     = prm.get_double("p");
    }
    prm.leave_subsection();
  }

  NonlinearLandweber(std::shared_ptr<NonlinearProblem<Param, Sol>> problem, std::shared_ptr<Param> initial_guess,
                     double omega)
      : NewtonRegularization<Param, Sol, Exact>(problem), initial_guess(initial_guess), omega(omega), p(2.0) {}

  NonlinearLandweber(std::shared_ptr<NonlinearProblem<Param, Sol>> problem, std::shared_ptr<Param> initial_guess,
                     ParameterHandler &prm)
      : NewtonRegularization<Param, Sol, Exact>(problem), initial_guess(initial_guess) {
    get_parameters(prm);
  }

  virtual Param invert(const Sol &data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
                       std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
    LogStream::Prefix prefix = LogStream::Prefix("Landweber");
    AssertThrow(this->problem, ExcInternalError());
    deallog.push("init");

    // possible with LW, but currently not implemented.
    AssertThrow(data.hilbert(), ExcMessage("Landweber: Y is not a Hilbert space!"));

    Param estimate(*initial_guess);

    Sol residual(data);
    Sol data_current = this->problem->forward(estimate);
    residual -= data_current;

    double discrepancy = residual.norm();
    double norm_data   = data.norm();

    deallog.pop();
    InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
                                                target_discrepancy, &data, norm_data, exact_param, false);
    this->progress(status);

    // dual index to p to use in X
    double q = p / (p - 1);

    for (int i = 1; discrepancy > target_discrepancy; i++) {
      deallog.push("step");  // same log levels as for REGINN for forward/adjoint ops

      std::unique_ptr<LinearProblem<Param, Sol>> lp = this->problem->derivative(estimate);

      Param adj = lp->adjoint(residual);

      estimate.duality_mapping(p);

      // $`c_{k+1} = c_k + \omega (S' c_k)^* (g - S c_k)`$
      estimate.add(omega, adj);

      estimate.duality_mapping_dual(q);

      double norm_estimate = estimate.norm();
      deallog.pop();

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
    }

    status.finished = true;
    this->progress(status);

    if (status_out) *status_out = status;

    return estimate;
  }

 private:
  double omega;
  double p;
  std::shared_ptr<Param> initial_guess;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
