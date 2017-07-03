/*
 * NonlinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_NONLINEARPROBLEM_H_
#define INCLUDE_NONLINEARPROBLEM_H_

#include <InverseProblem.h>
#include <LinearProblem.h>

namespace wavepi {

   template<typename Param, typename Sol>
   class NonlinearProblem : public InverseProblem<Param, Sol> {
      public:
         virtual ~NonlinearProblem() {}

         virtual LinearProblem<Param, Sol>& derivative(const Param& f) const = 0;

   };

} /* namespace wavepi */



#endif /* INCLUDE_NONLINEARPROBLEM_H_ */
