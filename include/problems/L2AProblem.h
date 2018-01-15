/*
 * L2AProblem.h
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2APROBLEM_H_
#define INCLUDE_PROBLEMS_L2APROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/DivRightHandSide.h>
#include <forward/DivRightHandSideAdjoint.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <problems/L2WaveProblem.h>

#include <memory>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim, typename Measurement>
class L2AProblem: public L2WaveProblem<dim, Measurement> {
   public:
      using L2WaveProblem<dim, Measurement>::derivative;
      using L2WaveProblem<dim, Measurement>::forward;

      virtual ~L2AProblem() = default;

      L2AProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
            : L2WaveProblem<dim, Measurement>(weq, right_hand_sides, measures, adjoint_solver), fields(
                  measures.size()) {
      }

      L2AProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
            : L2WaveProblem<dim, Measurement>(weq, right_hand_sides, measures), fields(measures.size()) {
      }

   protected:
      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            size_t i) {
         return std::make_unique<L2AProblem<dim, Measurement>::Linearization>(this->wave_equation,
               this->adjoint_solver, this->current_param, this->fields[i]);
      }

      virtual DiscretizedFunction<dim> forward(size_t i) {
         this->wave_equation.set_param_a(this->current_param);
         this->wave_equation.set_right_hand_side(
               std::make_shared<L2RightHandSide<dim>>(this->right_hand_sides[i]));
         this->wave_equation.set_run_direction(WaveEquation<dim>::Forward);

         DiscretizedFunction<dim> res = this->wave_equation.run();
         res.throw_away_derivative();

         // save a copy of res
         this->fields[i] = std::make_shared<DiscretizedFunction<dim>>(res);

         return res;
      }

   private:
      // solutions of the last forward problem
      std::vector<std::shared_ptr<DiscretizedFunction<dim>>> fields;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
         public:
            virtual ~Linearization() = default;

            Linearization(const WaveEquation<dim> &weq,
                  typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
                  const std::shared_ptr<DiscretizedFunction<dim>> a,
                  std::shared_ptr<DiscretizedFunction<dim>> u)
                  : weq(weq), weq_adj(weq), adjoint_solver(adjoint_solver) {
               this->a = a;
               this->u = u;

               this->rhs = std::make_shared<DivRightHandSide<dim>>(this->a, this->u);
               this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);
               this->m_adj = std::make_shared<DivRightHandSideAdjoint<dim>>(this->a, this->u);

               this->weq.set_right_hand_side(rhs);
               this->weq_adj.set_right_hand_side(rhs_adj);

               this->weq.set_initial_values_u(this->weq.zero);
               this->weq.set_initial_values_v(this->weq.zero);
               this->weq.set_boundary_values_u(this->weq.zero);
               this->weq.set_boundary_values_v(this->weq.zero);

               this->weq.set_param_a(this->a);
               this->weq_adj.set_param_a(this->a);
            }

            virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h) {
               rhs->set_a(std::make_shared<DiscretizedFunction<dim>>(h));
               weq.set_right_hand_side(rhs);
               weq.set_run_direction(WaveEquation<dim>::Forward);

               DiscretizedFunction<dim> res = weq.run();
               res.set_norm(DiscretizedFunction<dim>::Norm::L2L2);
               res.throw_away_derivative();

               return res;
            }

            virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& g) {
               // L*
               auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
               tmp->set_norm(DiscretizedFunction<dim>::Norm::L2L2);
               tmp->dot_solve_mass_and_transform();
               rhs_adj->set_base_rhs(tmp);

               DiscretizedFunction<dim> res(weq.get_mesh());

               if (adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards) {
                  AssertThrow((std::dynamic_pointer_cast<ZeroFunction<dim>, Function<dim>>(weq.get_param_nu()) != nullptr),
                  ExcMessage("Wrong adjoint because ν≠0!"));

                  weq.set_right_hand_side(rhs_adj);
                  weq.set_run_direction(WaveEquation<dim>::Backward);
                  res = weq.run();
                  res.throw_away_derivative();
               } else if (adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint)
               res = weq_adj.run();
               else
               Assert(false, ExcInternalError());

               res.set_norm(DiscretizedFunction<dim>::Norm::L2L2);
               // res.dot_mult_mass_and_transform_inverse();

               // M*
               // should be - nabla(res)*nabla(u) -> piecewise constant function -> fe spaces do not fit
               // res.dot_solve_mass_and_transform();
               m_adj->set_a(std::make_shared<DiscretizedFunction<dim>>(res));
               res = m_adj->run_adjoint(res.get_mesh());
               res.set_norm(DiscretizedFunction<dim>::Norm::L2L2);
               res.dot_transform_inverse();

               return res;
            }

            virtual DiscretizedFunction<dim> zero() {
               DiscretizedFunction<dim> res(a->get_mesh());
               res.set_norm(DiscretizedFunction<dim>::Norm::L2L2);

               return res;
            }

         private:
            WaveEquation<dim> weq;
            WaveEquationAdjoint<dim> weq_adj;

            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

            std::shared_ptr<DiscretizedFunction<dim>> a;
            std::shared_ptr<DiscretizedFunction<dim>> u;

            std::shared_ptr<DivRightHandSide<dim>> rhs;
            std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
            std::shared_ptr<DivRightHandSideAdjoint<dim>> m_adj;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2APROBLEM_H_ */
