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
using namespace dealii;

struct LinearProblemStats {
   public:
      int calls_forward;
      int calls_adjoint;

      double time_forward;
      double time_adjoint;

      int calls_measure_forward;
      int calls_measure_adjoint;

      double time_measure_forward;
      double time_measure_adjoint;

      double time_communication;
};

template<typename Param, typename Sol>
class LinearProblem: public InverseProblem<Param, Sol> {
   public:

      virtual ~LinearProblem() = default;

      virtual Param adjoint(const Sol& g) = 0;

      /**
       * Linear methods often do not use an initial guess, and want to start with zero.
       */
      virtual Param zero() = 0;

      /**
       * If supported, return statistics for the calls made to this class.
       * Otherwise, return `nullptr`, like the default implementation does.
       */
      virtual std::shared_ptr<LinearProblemStats> get_statistics() {
         return nullptr;
      }
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_LINEARPROBLEM_H_ */
