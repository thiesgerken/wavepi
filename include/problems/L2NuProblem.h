/*
 * L2CProblem.h
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2NUPROBLEM_H_
#define INCLUDE_PROBLEMS_L2NUPROBLEM_H_

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
class L2NuProblem: public NonlinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
   public:
      virtual ~L2NuProblem() {
      }

      L2NuProblem(WaveEquation<dim>& weq);
      L2NuProblem(WaveEquation<dim>& weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver);

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& nu, const DiscretizedFunction<dim>& u);

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& nu);

   private:
      WaveEquation<dim> wave_equation;
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

      // solution (with derivative!) and parameter from the last forward problem
      std::shared_ptr<DiscretizedFunction<dim>> nu;
      std::shared_ptr<DiscretizedFunction<dim>> u;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:
            virtual ~Linearization();

            Linearization(const WaveEquation<dim> &weq,
                  typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
                  std::shared_ptr<DiscretizedFunction<dim>> nu, std::shared_ptr<DiscretizedFunction<dim>> u);

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h);

            // L2 adjoint
            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g);

            virtual DiscretizedFunction<dim> zero();

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

            std::shared_ptr<DiscretizedFunction<dim>> nu;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<L2RightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2NUPROBLEM_H_ */
