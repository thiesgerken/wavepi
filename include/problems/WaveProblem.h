/*
 * WaveProblem.h
 *
 *  Created on: 04.07.2017
 *      Author: thies
 */

#ifndef INVERSION_WAVEPROBLEM_H_
#define INVERSION_WAVEPROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/WaveEquation.h>

#include <inversion/NonlinearProblem.h>

#include <memory>

namespace wavepi {
namespace inversion {
using namespace wavepi::forward;

// Continuous Problem, no measurements
// abstract class, following functions still have to be implemented by the sub class:
// virtual DiscretizedFunction<dim>& forward(const DiscretizedFunction<dim>& param) {}
// virtual LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>& derivative(const DiscretizedFunction<dim>& f) const {   }
template<int dim>
class WaveProblem: public NonlinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
   public:
      virtual ~WaveProblem();

      // no adaptivity; weq contains the space-time grid that the data and the parameters live on
      WaveProblem(WaveEquation<dim> &weq);

      WaveEquation<dim>& get_wave_equation();

      DiscretizedFunction<dim> generateNoise(const DiscretizedFunction<dim>& like, double norm) const;

      bool progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state);

      // solvers for adjoint of L : L^2 -> L^2
      enum L2AdjointSolver {
         WaveEquationAdjoint, WaveEquationBackwards
      };

   protected:
      WaveEquation<dim> wave_equation;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEPROBLEM_H_ */
