/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LANDWEBER_H_
#define INVERSION_LANDWEBER_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <inversion/LinearProblem.h>
#include <inversion/LinearRegularization.h>

#include <memory>

namespace wavepi {
namespace inversion {
using namespace dealii;

// linear Landweber iteration
template <typename Param, typename Sol, typename Exact>
class Landweber : public LinearRegularization<Param, Sol, Exact> {
 public:
  virtual ~Landweber() = default;

  Landweber(double omega) : omega(omega) {}

  Landweber(ParameterHandler &prm) { get_parameters(prm); }

  static void declare_parameters(ParameterHandler &prm) {
    prm.enter_subsection("Landweber");
    { prm.declare_entry("omega", "1", Patterns::Double(0), "relaxation factor ω"); }
    prm.leave_subsection();
  }

  void get_parameters(ParameterHandler &prm) {
    prm.enter_subsection("Landweber");
    { omega = prm.get_double("omega"); }
    prm.leave_subsection();
  }

  virtual Param invert(const Sol &data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
                       std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
    LogStream::Prefix p = LogStream::Prefix("Landweber");
    AssertThrow(this->problem, ExcInternalError());

    Param estimate = this->problem->zero();
    Sol residual(data);

    // possible with LW, but currently not implemented.
    AssertThrow(estimate.hilbert(), ExcMessage("linear Landweber: X is not a Hilbert space!"));
    AssertThrow(data.hilbert(), ExcMessage("linear Landweber: Y is not a Hilbert space!"));

    double discrepancy = residual.norm();
    double norm_data   = data.norm();

    InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
                                                target_discrepancy, &data, norm_data, exact_param, false);
    this->progress(status);

    for (int k = 1; discrepancy > target_discrepancy; k++) {
      Param adj = this->problem->adjoint(residual);

      // $`c_{k+1} = c_k + \omega A^* (g - A c_k)`$
      estimate.add(omega, adj);

      double norm_estimate = estimate.norm();

      // post-processing
      deallog.push("post_processing");
      this->post_process(k, &estimate, norm_estimate);
      deallog.pop();

      // calculate new residual and discrepancy
      deallog.push("finish_step");
      residual         = data;
      Sol data_current = this->problem->forward(estimate);
      residual -= data_current;
      discrepancy = residual.norm();
      deallog.pop();

      status = InversionProgress<Param, Sol, Exact>(k, &estimate, norm_estimate, &residual, discrepancy,
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
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
