/*
 * LinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_LINEARPROBLEM_H_
#define INVERSION_LINEARPROBLEM_H_

#include <inversion/InverseProblem.h>

namespace wavepi {
namespace inversion {

template<typename Param, typename Sol>
class LinearProblem: public InverseProblem<Param, Sol> {
   public:
      
      virtual ~LinearProblem() = default;

      virtual Param adjoint(const Sol& g) = 0;

      // linear methods often do not use an initial guess, but start with zero
      virtual Param zero() = 0;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LINEARPROBLEM_H_ */
