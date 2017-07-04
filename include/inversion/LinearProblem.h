/*
 * LinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LINEARPROBLEM_H_
#define INVERSION_LINEARPROBLEM_H_

#include "inversion/InverseProblem.h"

namespace wavepi {
namespace inversion {

template<typename Param, typename Sol>
class LinearProblem: public InverseProblem<Param, Sol> {
   public:
      virtual Param& adjoint(const Sol& g) const = 0;

      virtual ~LinearProblem() {
      }

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LINEARPROBLEM_H_ */
