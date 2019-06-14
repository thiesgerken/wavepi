/*
 * RiederToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/RiederToleranceChoice.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace wavepi {
namespace inversion {

using namespace dealii;

RiederToleranceChoice::RiederToleranceChoice(double tol_max, double zeta, double beta,
                                             bool safeguarding)
    : tol_max(tol_max), zeta(zeta), beta(beta), safeguarding(safeguarding) {}

RiederToleranceChoice::RiederToleranceChoice(ParameterHandler &prm) { get_parameters(prm); }

void RiederToleranceChoice::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("RiederToleranceChoice");
  {
    prm.declare_entry("tol max", "0.999", Patterns::Double(0, 1), "rel. maximum tolerance");
    prm.declare_entry("zeta", "0.95", Patterns::Double(0, 1), "factor to decrease tolerance by");
    prm.declare_entry("beta", "1.0", Patterns::Double(0), "allowed speed of iteration numbers");
    prm.declare_entry("safeguarding", "true", Patterns::Bool(),
                      "safeguard the outer iteration from over-fulfillment of discrepancy principle");
  }
  prm.leave_subsection();
}

void RiederToleranceChoice::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("RiederToleranceChoice");
  {
    tol_max      = prm.get_double("tol max");
    zeta         = prm.get_double("zeta");
    beta         = prm.get_double("beta");
    safeguarding = prm.get_bool("safeguarding");
  }
  prm.leave_subsection();

  ToleranceChoice::get_parameters(prm);
}

double RiederToleranceChoice::calculate_tolerance() const {
  int k = previous_tolerances.size();

  double tol_tilde;

  // Original: 
  // if (k < 2)
  //   tol_tilde = tol_start;
  if (k == 0)
    tol_tilde = 1;
  else if (k==1) 
    tol_tilde = discrepancies[0] / initial_discrepancy;
  else if (required_steps[k - 1] >= required_steps[k - 2] && required_steps[k - 2] > 5)
    // last condition in else if is new: do not regard less than 5 inner iterations as an increase
    tol_tilde =
        1 - std::pow((double)required_steps[k - 2] / required_steps[k - 1], beta) * (1 - previous_tolerances[k - 1]);
  else
    tol_tilde = zeta * previous_tolerances[k - 1];

  double last_discrepancy = (k == 0) ? initial_discrepancy : discrepancies[k - 1];

  //  if (k < 2)
  //    return std::max(tol_max * target_discrepancy / last_discrepancy, tol_tilde);
  //  else
  //    return tol_max * std::max(target_discrepancy / last_discrepancy, tol_tilde);

  // a bit different, do not multiply with tol_max every time:
  if (safeguarding)
    return std::min(tol_max, std::max(target_discrepancy / last_discrepancy, tol_tilde));
  else
    return std::min(tol_max, tol_tilde);
}

}  // namespace inversion
} /* namespace wavepi */
