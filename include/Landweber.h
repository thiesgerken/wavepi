/*
 * Landweber.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_LANDWEBER_H_
#define INCLUDE_LANDWEBER_H_

#include <Regularization.h>
#include <NewtonRegularization.h>
#include <LinearProblem.h>
#include <NonlinearProblem.h>

namespace wavepi {

   // nonlinear Landweber iteration (it can be regarded as an inexact newton method,
   // applying one linear landweber step to the linearized problem to 'solve' it)
   template<typename Param, typename Sol>
   class Landweber: public NewtonRegularization<Param, Sol> {
      public:
         Landweber(NonlinearProblem<Param, Sol> *problem)
               : NewtonRegularization<Param, Sol>(problem) {
         }

         virtual ~Landweber() {
         }

         virtual Param invert(Sol data, double targetDiscrepancy) {
            // TODO
         }
   };

} /* namespace wavepi */

#endif /* INCLUDE_LANDWEBER_H_ */
