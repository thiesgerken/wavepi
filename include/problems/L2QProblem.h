/*
 * L2QProblem.h
 *
 *  Created on: 20.07.2017
 *      Author: thies
 */

#ifndef PROBLEMS_L2QPROBLEM_H_
#define PROBLEMS_L2QPROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/L2ProductRightHandSide.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>

#include <problems/WaveProblem.h>

#include <memory>

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

namespace wavepi {
namespace problems {

template<int dim>
class L2QProblem: public WaveProblem<dim> {
   public:
      virtual ~L2QProblem();

      L2QProblem(WaveEquation<dim>& weq);
      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u);

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& q);

   private:
      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:
            virtual ~Linearization();

            Linearization(const WaveEquation<dim> &weq, const DiscretizedFunction<dim>& q,
                  const DiscretizedFunction<dim>& u);

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h);

            // L2 adjoint
            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g);

            virtual DiscretizedFunction<dim> zero();

            bool progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state);

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            std::shared_ptr<DiscretizedFunction<dim>> q;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<L2ProductRightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* LIB_PROBLEMS_L2QPROBLEM_H_ */
