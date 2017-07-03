/*
 * LinearRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_LINEARREGULARIZATION_H_
#define INCLUDE_LINEARREGULARIZATION_H_

#include <Regularization.h>
#include <LinearProblem.h>

namespace wavepi {

   // Param and Sol need at least banach space structure
   template<typename Param, typename Sol>
   class LinearRegularization: Regularization<Param, Sol> {
      public:
         LinearRegularization(LinearProblem<Param, Sol> problem)
               : Regularization(problem), problem(problem) {
         }

         virtual ~LinearRegularization() {
         }

         // virtual Param invert(Sol data, double targetDiscrepancy) = 0;

      protected:
         LinearProblem problem;
   };

} /* namespace wavepi */

#endif /* INCLUDE_LINEARREGULARIZATION_H_ */
