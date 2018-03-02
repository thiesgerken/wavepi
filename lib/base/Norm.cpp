/*
 * Norm.cpp
 *
 *  Created on: 02.03.2018
 *      Author: thies
 */

#include <base/Norm.h>

namespace wavepi {
namespace base {
using namespace dealii;

std::string to_string(const Norm &norm) {
  switch (norm) {
    case Norm::Invalid:
      return "Invalid";
    case Norm::Coefficients:
      return "ℝⁿ";
    case Norm::L2L2:
      return "L²([0,T], L²(Ω))";
    case Norm::H1L2:
      return "H¹([0,T], L²(Ω))";
    case Norm::H1H1:
      return "H¹([0,T], H¹(Ω))";
    case Norm::H2L2:
      return "H²([0,T], L²(Ω))";
    default:
      AssertThrow(false, ExcMessage("to_string called on unknown norm"));
  }

  return "";
}

}  // namespace base
}  // namespace wavepi
