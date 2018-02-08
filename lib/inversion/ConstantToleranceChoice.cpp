/*
 * ConstantToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/ConstantToleranceChoice.h>

namespace wavepi {
namespace inversion {

ConstantToleranceChoice::ConstantToleranceChoice(double tol) : tol(tol) {}

ConstantToleranceChoice::ConstantToleranceChoice(ParameterHandler &prm) { get_parameters(prm); }

void ConstantToleranceChoice::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ConstantToleranceChoice");
  { prm.declare_entry("tol", "0.7", Patterns::Double(0, 1), "rel. tolerance"); }
  prm.leave_subsection();
}

void ConstantToleranceChoice::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ConstantToleranceChoice");
  { tol = prm.get_double("tol"); }
  prm.leave_subsection();

  ToleranceChoice::get_parameters(prm);
}

double ConstantToleranceChoice::calculate_tolerance() const { return tol; }

}  // namespace inversion
} /* namespace wavepi */
