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

RiederToleranceChoice::RiederToleranceChoice(double tol_start, double tol_max, double zeta, double beta)
      : tol_start(tol_start), tol_max(tol_max), zeta(zeta), beta(beta) {
}

double RiederToleranceChoice::calculate_tolerance() const {
   int k = previous_tolerances.size();

   double tol_tilde;

   if (k < 2)
      tol_tilde = tol_start;
   else if (required_steps[k - 1] >= required_steps[k - 2])
      tol_tilde = 1
            - std::pow((double) required_steps[k - 2] / required_steps[k - 1], beta)
                  * (1 - previous_tolerances[k - 1]);
   else
      tol_tilde = zeta * previous_tolerances[k - 1];

   double last_discrepancy = (k == 0) ? initial_discrepancy : discrepancies[k - 1];

   if (k < 2)
      return std::max(tol_max * target_discrepancy / last_discrepancy, tol_tilde);
   else
      return tol_max * std::max(target_discrepancy / last_discrepancy, tol_tilde);
}

} /* namespace problems */
} /* namespace wavepi */
