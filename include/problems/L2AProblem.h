/*
 * L2CProblem.h
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2APROBLEM_H_
#define INCLUDE_PROBLEMS_L2APROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/DivRightHandSide.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>

#include <problems/WaveProblem.h>

#include <memory>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
class L2AProblem: public WaveProblem<dim> {
   public:
      virtual ~L2AProblem() {
      }

      L2AProblem(WaveEquation<dim>& weq);
      L2AProblem(WaveEquation<dim>& weq, typename WaveProblem<dim>::L2AdjointSolver adjoint_solver);

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& a, const DiscretizedFunction<dim>& u);

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& a);

   private:
      typename WaveProblem<dim>::L2AdjointSolver adjoint_solver;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:
            virtual ~Linearization();

            Linearization(const WaveEquation<dim> &weq,
                  typename WaveProblem<dim>::L2AdjointSolver adjoint_solver,
                  const DiscretizedFunction<dim>& a, const DiscretizedFunction<dim>& u);

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h);

            // L2 adjoint
            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g);

            virtual DiscretizedFunction<dim> zero();

            bool progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state);

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            typename WaveProblem<dim>::L2AdjointSolver adjoint_solver;

            std::shared_ptr<DiscretizedFunction<dim>> a;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<DivRightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2APROBLEM_H_ */
