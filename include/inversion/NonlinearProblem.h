/*
 * NonlinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NONLINEARPROBLEM_H_
#define INVERSION_NONLINEARPROBLEM_H_

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>

#include <memory>

namespace wavepi {
namespace inversion {

template<typename Param, typename Sol>
class NonlinearProblem: public InverseProblem<Param, Sol> {
   public:

      virtual ~NonlinearProblem() = default;

      /**
       * returns the derivative (as linear operator) at p.
       */
      virtual std::unique_ptr<LinearProblem<Param, Sol>> derivative(const Param& p) = 0;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NONLINEARPROBLEM_H_ */
