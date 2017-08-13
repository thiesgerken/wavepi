/*
 * L2QProblem.h
 *
 *  Created on: 20.07.2017
 *      Author: thies
 */

#ifndef PROBLEMS_L2QPROBLEM_H_
#define PROBLEMS_L2QPROBLEM_H_

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
class L2QProblem: public NonlinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
   public:
      /**
       * Default destructor.
       */
      virtual ~L2QProblem() = default;

      L2QProblem(WaveEquation<dim>& weq);
      L2QProblem(WaveEquation<dim>& weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver);

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u);

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& q);

   private:
      WaveEquation<dim> wave_equation;
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:
            /**
             * Default destructor.
             */
            virtual ~Linearization() = default;

            Linearization(const WaveEquation<dim> &weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
                  const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u);

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h);

            // L2 adjoint
            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g);

            virtual DiscretizedFunction<dim> zero();

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

            std::shared_ptr<DiscretizedFunction<dim>> q;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<L2RightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* LIB_PROBLEMS_L2QPROBLEM_H_ */
