/*
 * Regularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_REGULARIZATION_H_
#define INCLUDE_REGULARIZATION_H_

#include <InverseProblem.h>
#include <iostream>

namespace wavepi {

   // Param and Sol need at least banach space structure
   template<typename Param, typename Sol>
   class Regularization {
      public:
         Regularization(InverseProblem<Param, Sol> *problem)
               : problem(problem) {
         }

         virtual ~Regularization() {
         }

         virtual Param invert(Sol data, double targetDiscrepancy) = 0;

         const Param& test(const Param& f, double noiseLevel, double tau) {
            // TODO
            std::cout << "test" << noiseLevel << std::endl;
         }

      private:
         InverseProblem<Param, Sol> *problem;
   };

} /* namespace wavepi */

#endif /* INCLUDE_REGULARIZATION_H_ */
