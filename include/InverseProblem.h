/*
 * InverseProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSEPROBLEM_H_
#define INCLUDE_INVERSEPROBLEM_H_

namespace wavepi {

   template<typename Param, typename Sol>
   class InverseProblem {
      public:
         typedef Sol& (*ParamToSol)(const Param&);
         typedef Param& (*SolToParam)(const Sol&);

         virtual ~InverseProblem() {             }

         virtual Sol& generateNoise(const Param& like, double norm) const = 0;

         virtual Sol& forward(const Param& f) const = 0 ;

         // progress indicator that iterative methods can call
         virtual void progress(const Param& current_estimate, const Sol& current_residual, int iteration_number) {
         }

   };

} /* namespace wavepi */

#endif /* INCLUDE_INVERSEPROBLEM_H_ */
