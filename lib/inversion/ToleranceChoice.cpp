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

void ToleranceChoice::reset(double target_discrepancy) {
   previous_tolerances.clear();
   required_steps.clear();
   residuals.clear();

   this->target_discrepancy = target_discrepancy;
}

double ToleranceChoice::get_tolerance() {
   AssertThrow(residuals.size() == previous_tolerances.size(), ExcInternalError());

   double tol = calculate_tolerance();

   previous_tolerances.push_back(tol);
   return tol;
}

void ToleranceChoice::add_iteration(double new_residual, int steps) {
   AssertThrow(residuals.size() == previous_tolerances.size() - 1, ExcInternalError());

   residuals.push_back(new_residual);
   required_steps.push_back(steps);
}

double ToleranceChoice::get_target_discrepancy() const {
   return target_discrepancy;
}

void ToleranceChoice::set_target_discrepancy(double target_discrepancy) {
   this->target_discrepancy = target_discrepancy;
}

} /* namespace problems */
} /* namespace wavepi */
