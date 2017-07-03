/*
 * LinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_LINEARPROBLEM_H_
#define INCLUDE_LINEARPROBLEM_H_

#include "InverseProblem.h"

namespace wavepi {

   template<typename Param, typename Sol>
   class LinearProblem : public InverseProblem<Param, Sol> {
      public:
         virtual Param& adjoint(const Sol& g) const = 0 ;

         virtual ~LinearProblem() {}

   };

} /* namespace wavepi */


#endif /* INCLUDE_LINEARPROBLEM_H_ */
