/*
 * L2CProblem.h
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2CPROBLEM_H_
#define INCLUDE_PROBLEMS_L2CPROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <memory>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
class L2CProblem: public NonlinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
   public:

      virtual ~L2CProblem() = default;

      L2CProblem(WaveEquation<dim>& weq);
      L2CProblem(WaveEquation<dim>& weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver);

      // has to be called for the same parameter as the last forward problem!
      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& c);

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& c);

   private:
      WaveEquation<dim> wave_equation;
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

      // solution (with derivative!) and parameter from the last forward problem
      std::shared_ptr<DiscretizedFunction<dim>> c;
      std::shared_ptr<DiscretizedFunction<dim>> u;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:

            virtual ~Linearization() = default;

            Linearization(const WaveEquation<dim> &weq,
                  typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
                  std::shared_ptr<DiscretizedFunction<dim>> q, std::shared_ptr<DiscretizedFunction<dim>> u);

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h);

            // L2 adjoint
            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g);

            virtual DiscretizedFunction<dim> zero();

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

            std::shared_ptr<DiscretizedFunction<dim>> c;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<L2RightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2CPROBLEM_H_ */
