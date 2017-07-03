/*
 * NewtonRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_NEWTONREGULARIZATION_H_
#define INCLUDE_NEWTONREGULARIZATION_H_

#include <Regularization.h>
#include <LinearProblem.h>
#include <NonlinearProblem.h>

namespace wavepi {

   // Param and Sol need at least banach space structure
   template<typename Param, typename Sol>
   class NewtonRegularization: public Regularization<Param, Sol> {
      public:
         NewtonRegularization(NonlinearProblem<Param, Sol>* problem)
               : Regularization<Param, Sol>(problem), problem(problem) {
         }

         virtual ~NewtonRegularization() {
         }

         // virtual Param invert(Sol data, double targetDiscrepancy) = 0;

      protected:
         NonlinearProblem<Param, Sol> *problem;
   };

} /* namespace wavepi */

#endif /* INCLUDE_NEWTONREGULARIZATION_H_ */
