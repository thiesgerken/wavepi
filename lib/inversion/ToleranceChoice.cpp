/*
 * ToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/ToleranceChoice.h>

#include <deal.II/base/exceptions.h>

namespace wavepi {
namespace inversion {

using namespace dealii;

ToleranceChoice::~ToleranceChoice() {
}

void ToleranceChoice::reset(double target_discrepancy, double initial_discrepancy) {
   previous_tolerances.clear();
   required_steps.clear();
   discrepancies.clear();

   this->target_discrepancy = target_discrepancy;
   this->initial_discrepancy = initial_discrepancy;
}

double ToleranceChoice::get_tolerance() {
   AssertThrow(discrepancies.size() == previous_tolerances.size(), ExcInternalError());

   double tol = calculate_tolerance();

   previous_tolerances.push_back(tol);
   return tol;
}

void ToleranceChoice::add_iteration(double new_discrepancy, int steps) {
   AssertThrow(discrepancies.size() == previous_tolerances.size() - 1, ExcInternalError());

   discrepancies.push_back(new_discrepancy);
   required_steps.push_back(steps);
}

} /* namespace problems */
} /* namespace wavepi */
