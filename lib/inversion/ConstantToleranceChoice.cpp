/*
 * ConstantToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/ConstantToleranceChoice.h>

namespace wavepi {
namespace inversion {

ConstantToleranceChoice::ConstantToleranceChoice(double tol)
      : tol(tol) {
}

double ConstantToleranceChoice::get_tol() const {
   return tol;
}

void ConstantToleranceChoice::set_tol(double tol) {
   this->tol = tol;
}

double ConstantToleranceChoice::calculate_tolerance() const {
   return tol;
}

} /* namespace problems */
} /* namespace wavepi */
