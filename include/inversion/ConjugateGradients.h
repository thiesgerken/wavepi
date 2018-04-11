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
template <typename Param, typename Sol, typename Exact>
class ConjugateGradients : public LinearRegularization<Param, Sol, Exact> {
 public:
  virtual ~ConjugateGradients() = default;

  ConjugateGradients(bool use_safeguarding) : use_safeguarding(use_safeguarding) {}

  ConjugateGradients(ParameterHandler &prm) { get_parameters(prm); }

  static void declare_parameters(ParameterHandler &prm) {
    prm.enter_subsection("ConjugateGradients");
    {
      prm.declare_entry(
          "safeguarding", "true", Patterns::Bool(),
          "interpolate between the last two iterates such that the target discrepancy is exactly reached");
    }
    prm.leave_subsection();
  }

  void get_parameters(ParameterHandler &prm) {
    prm.enter_subsection("ConjugateGradients");
    { use_safeguarding = prm.get_bool("safeguarding"); }
    prm.leave_subsection();
  }

  virtual Param invert(const Sol &data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
                       std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
    LogStream::Prefix prefix("CGLS");
    AssertThrow(this->problem, ExcInternalError());

    Param estimate(this->problem->zero());  // f_k
    Sol residual(data);                     // r_k

    Param p(this->problem->adjoint(residual));  // p_{k+1}
    Param d(p);                                 // d_k
    Sol q(this->problem->forward(p));           // q_k

    AssertThrow(estimate.hilbert(), ExcMessage("CG: X is not a Hilbert space!"));
    // AssertThrow(data.hilbert(), ExcMessage("CG: Y is not a Hilbert space!"));

    // needs to be outside of the iteration
    double alpha = 0;

    double norm_d           = d.norm();
    double discrepancy      = residual.norm();
    double last_discrepancy = discrepancy;
    double norm_data        = data.norm();

    InversionProgress<Param, Sol, Exact> status(0, &estimate, estimate.norm(), &residual, discrepancy,
                                                target_discrepancy, &data, norm_data, exact_param, false);
    this->progress(status, this->problem->get_statistics());

    for (int k = 1; discrepancy > target_discrepancy; k++) {
      alpha = square(norm_d / q.norm());  // α_k

      {
        LogStream::Prefix pp("info");
        deallog << "α_k = " << alpha << std::endl;
      }

      if (alpha == 0.0) break;

      estimate.add(alpha, p);
      residual.add(-alpha, q);

      last_discrepancy = discrepancy;
      discrepancy      = residual.norm();

      status = InversionProgress<Param, Sol, Exact>(k, &estimate, estimate.norm(), &residual, discrepancy,
                                                    target_discrepancy, &data, norm_data, exact_param, false);

      if (!this->progress(status, this->problem->get_statistics())) break;

      // abort now to save one evaluation of the adjoint and forward operator if we are finished
      if (discrepancy <= target_discrepancy) break;

      d = this->problem->adjoint(residual);

      double norm_d_last = norm_d;  // ‖d_{k-1}‖
      norm_d             = d.norm();

      double beta = square(norm_d / norm_d_last);  // β_k

      {
        LogStream::Prefix pp("info");
        deallog << "β_k = " << beta << std::endl;
      }

      p.sadd(beta, 1.0, d);
      q = this->problem->forward(p);
    }

    // safe guarding if target is reached
    if (discrepancy <= target_discrepancy && use_safeguarding) {
      double lambda = compute_safeguarding_factor(
          last_discrepancy, discrepancy, alpha * q.dot(residual) + discrepancy * discrepancy, target_discrepancy);

      estimate.add(-lambda * alpha, p);
      residual.add(lambda * alpha, q);
      discrepancy = residual.norm();

      deallog << "Safeguarding: λ=" << lambda << " ⇒ rdisc=" << discrepancy / norm_data << std::endl;
    }

    status.finished = true;
    this->progress(status, this->problem->get_statistics());

    if (status_out) *status_out = status;

    return estimate;
  }

 private:
  bool use_safeguarding = true;

  /**
   * for iterates $x_k$ and $x_{k+1}$, compute $\lambda$ s.t. $\lambda x_k + (1-\lambda) x_{k+1}$ has discrepancy
   * `target`
   * ($\lambda \approx 0$ => use mostly $x_{k+1}$
   *
   * @param disc discrepancy for $x_k$, has to be > target
   * @param disc_new discrepancy for $x_{k+1}$, has to be ≤ target
   * @param res_dot scalar product between residuals, i.e. $(g-Ax_{k+1}, g-Ax_k)$
   * @param target desired discrepancy
   */
  static double compute_safeguarding_factor(double disc, double disc_new, double res_dot, double target) {
    AssertThrow(disc > target && disc_new <= target,
                ExcMessage("compute_safeguarding_factor: discrepancies not over- and under target"));

    double a = disc * disc + disc_new * disc_new - 2 * res_dot;

    double p = 2 * (res_dot - disc_new * disc_new) / a;
    double q = (disc_new * disc_new - target * target) / a;

    AssertThrow(p * p / 4 >= q, ExcMessage("compute_safeguarding_factor: cannot get real solutions"));

    double rad = std::sqrt(p * p / 4 - q);
    double x1  = -p / 2 + rad;
    double x2  = -p / 2 - rad;

    if (0 <= x1 && x1 <= 1) return x1;
    if (0 <= x2 && x2 <= 1) return x2;

    AssertThrow(false, ExcMessage("compute_safeguarding_factor: cannot get solutions between 0 and 1"));
    return 0;
  }

  static inline double square(const double x) { return x * x; }
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_CONJUGATEGRADIENTS_H_ */
