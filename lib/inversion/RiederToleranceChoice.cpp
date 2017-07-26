/*
 * RiederToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/RiederToleranceChoice.h>

namespace wavepi {
namespace inversion {

RiederToleranceChoice::RiederToleranceChoice(double tol_start, double tol_max, double zeta)
      : tol_start(tol_start), tol_max(tol_max), zeta(zeta) {
}

double RiederToleranceChoice::calculate_tolerance() const {
   int k = previous_tolerances.size();

   double tol_tilde;

   if (k < 2)
      tol_tilde = tol_start;
   else if (required_steps[k - 1] >= required_steps[k - 2])
      tol_tilde = 1 - required_steps[k - 2] / required_steps[k - 1] * (1 - previous_tolerances[k - 1]);
   else
      tol_tilde = zeta * previous_tolerances[k - 1];

   return tol_max * std::max(target_discrepancy / residuals[k - 1], tol_tilde);
}

double RiederToleranceChoice::get_tol_max() const {
   return tol_max;
}

void RiederToleranceChoice::set_tol_max(double tol_max) {
   this->tol_max = tol_max;
}

double RiederToleranceChoice::get_tol_start() const {
   return tol_start;
}

void RiederToleranceChoice::set_tol_start(double tol_start) {
   this->tol_start = tol_start;
}

double RiederToleranceChoice::get_zeta() const {
   return zeta;
}

void RiederToleranceChoice::set_zeta(double zeta) {
   this->zeta = zeta;
}

} /* namespace problems */
} /* namespace wavepi */
