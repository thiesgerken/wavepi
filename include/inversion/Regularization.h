/*
 * Regularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_REGULARIZATION_H_
#define INVERSION_REGULARIZATION_H_

#include <inversion/InverseProblem.h>

#include <iostream>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template<typename Param, typename Sol>
class Regularization {
   public:
      virtual ~Regularization() {
      }

      virtual Param invert(const Sol& data, double target_discrepancy, const Param* exact_param) = 0;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_REGULARIZATION_H_ */
