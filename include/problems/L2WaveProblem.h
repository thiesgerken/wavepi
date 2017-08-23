/*
 * L2WaveProblem.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_
#define INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_

#include <deal.II/base/function.h>

#include <forward/DiscretizedFunction.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>

#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <measurements/Tuple.h>
#include <measurements/Measure.h>

#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::measurements;

template<int dim, typename Measurement>
class L2WaveProblem: public NonlinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
   public:
      L2WaveProblem();
      virtual ~L2WaveProblem() = default;

      L2WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
            : wave_equation(weq), adjoint_solver(adjoint_solver), right_hand_sides(right_hand_sides), measures(measures) {
         AssertThrow(right_hand_sides.size() == measures.size() && measures.size(), ExcInternalError());
      }

      L2WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
            : L2WaveProblem<dim, Measurement>(weq, right_hand_sides, measures,
                  WaveEquationBase<dim>::WaveEquationAdjoint) {
      }

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>>> derivative(
            const DiscretizedFunction<dim>& param) {
         Assert(this->current_param->relative_error(param) < 1e-10, ExcInternalError());

         std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> derivs;

         for (size_t i = 0; i < right_hand_sides.size(); i++)
            derivs.push_back(derivative(i));

         return std::make_unique<L2WaveProblem<dim, Measurement>::Linearization>(derivs, measures);
      }

      virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& param) {
         LogStream::Prefix p("forward");

         // save a copy of the parameter
         this->current_param = std::make_shared<DiscretizedFunction<dim>>(param);

         Tuple<Measurement> result;

         for (size_t i = 0; i < right_hand_sides.size(); i++)
            result.push_back(measures[i]->evaluate(forward(i)));

         return result;
      }
   protected:
      WaveEquation<dim> wave_equation;
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;
      std::vector<std::shared_ptr<Function<dim>>> right_hand_sides;

      std::shared_ptr<DiscretizedFunction<dim>> current_param;

      /**
       * Get derivative for param `current_param`. It is guaranteed to be the parameter for which `forward` was last called.
       *
       * @param rhs_index index of right hand side to use
       */
      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            size_t rhs_index) = 0;

      /**
       * Evaluate forward operator for param `current_param`.
       *
       * @param rhs_index index of right hand side to use
       */
      virtual DiscretizedFunction<dim> forward(size_t rhs_index) = 0;

   private:

      std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
         public:
            virtual ~Linearization() = default;

            Linearization(
                  std::vector<
                        std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> sub_problems,
                  std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
                  : sub_problems(sub_problems), measures(measures) {
            }

            virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& h) {
               LogStream::Prefix p("lin_forward");

               Tuple<Measurement> result;

               for (size_t i = 0; i < measures.size(); i++)
                  result.push_back(measures[i]->evaluate(sub_problems[i]->forward(h)));

               return result;
            }

            virtual DiscretizedFunction<dim> adjoint(const Tuple<Measurement>& g) {
               LogStream::Prefix p("lin_adjoint");

               DiscretizedFunction<dim> result(zero());

               for (size_t i = 0; i < measures.size(); i++)
                  result += sub_problems[i]->adjoint(measures[i]->adjoint(g[i]));

               return result;
            }

            virtual DiscretizedFunction<dim> zero() {
               return sub_problems[0]->zero();
            }

         private:
            std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> sub_problems;
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_ */
