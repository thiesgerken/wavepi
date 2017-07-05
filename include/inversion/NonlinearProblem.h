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
      virtual ~NonlinearProblem() {
      }

      virtual std::unique_ptr<LinearProblem<Param, Sol>> derivative(Param& h, Sol& u) = 0;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NONLINEARPROBLEM_H_ */
