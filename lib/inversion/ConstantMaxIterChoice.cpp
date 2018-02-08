/*
 * ConstantMaxIterChoice.cpp
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#include <inversion/ConstantMaxIterChoice.h>

namespace wavepi {
namespace inversion {

ConstantMaxIterChoice::ConstantMaxIterChoice(int max_iter) : max_iter(max_iter) {}

ConstantMaxIterChoice::ConstantMaxIterChoice(ParameterHandler &prm) { get_parameters(prm); }

void ConstantMaxIterChoice::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ConstantMaxIterChoice");
  {
    prm.declare_entry("maximum iteration count", "0", Patterns::Integer(),
                      "constant maximum iteration threshold. Set to ≤ 0 to diable.");
  }
  prm.leave_subsection();
}

void ConstantMaxIterChoice::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ConstantMaxIterChoice");
  { max_iter = prm.get_integer("maximum iteration count"); }
  prm.leave_subsection();
}

int ConstantMaxIterChoice::calculate_max_iter() const { return max_iter; }

} /* namespace inversion */
} /* namespace wavepi */
