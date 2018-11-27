/*
 * QProblem.h
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#ifndef PROBLEMS_QPROBLEM_H_
#define PROBLEMS_QPROBLEM_H_

#include <base/DiscretizedFunction.h>
#include <forward/DivRightHandSide.h>
#include <forward/DivRightHandSideAdjoint.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <problems/WaveProblem.h>

#include <memory>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim, typename Measurement>
class QProblem: public WaveProblem<dim, Measurement> {
public:
   using WaveProblem<dim, Measurement>::derivative;
   using WaveProblem<dim, Measurement>::forward;

   virtual ~QProblem() = default;

   QProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
         std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
         std::shared_ptr<Transformation<dim>> transform, std::shared_ptr<DiscretizedFunction<dim>> background_param)
         : WaveProblem<dim, Measurement>(weq, right_hand_sides, measures, transform, background_param),
               fields(measures.size()) {
   }

protected:
   virtual std::unique_ptr<LinearizedSubProblem<dim>> derivative(size_t i) {
      return std::make_unique<QProblem<dim, Measurement>::Linearization>(this->wave_equation, this->adjoint_solver,
            this->current_param, this->fields[i], this->norm_domain, this->norm_codomain);
   }

   virtual DiscretizedFunction<dim> forward(size_t i) {
      this->wave_equation.set_param_q(this->current_param);
      DiscretizedFunction<dim> res = this->wave_equation.run(
            std::make_shared<L2RightHandSide<dim>>(this->right_hand_sides[i]), WaveEquation<dim>::Forward);
      res.set_norm(this->norm_codomain);
      res.throw_away_derivative();

      // save a copy of res
      this->fields[i] = std::make_shared<DiscretizedFunction<dim>>(res);

      return res;
   }

private:
   // solutions of the last forward problem
   std::vector<std::shared_ptr<DiscretizedFunction<dim>>> fields;

   class Linearization: public LinearizedSubProblem<dim> {
   public:
      virtual ~Linearization() = default;

      Linearization(const WaveEquation<dim>& weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
            const std::shared_ptr<DiscretizedFunction<dim>> q, std::shared_ptr<DiscretizedFunction<dim>> u,
            std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain,
            std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain)
            : weq(weq), weq_adj(weq), norm_domain(norm_domain), norm_codomain(norm_codomain),
                  adjoint_solver(adjoint_solver) {
         this->q = q;
         this->u = u;

         this->rhs = std::make_shared<L2RightHandSide<dim>>(this->u);
         this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);

         this->weq.set_initial_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_initial_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_boundary_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_boundary_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));

         this->weq.set_param_q(this->q);
         this->weq_adj.set_param_q(this->q);
      }

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h) {
         auto Mh = std::make_shared<DiscretizedFunction<dim>>(h);
         *Mh *= -1.0;
         Mh->pointwise_multiplication(*u);

         rhs->set_base_rhs(Mh);

         DiscretizedFunction<dim> res = weq.run(rhs, WaveEquation<dim>::Forward);
         res.set_norm(this->norm_codomain);
         res.throw_away_derivative();

         return res;
      }

      virtual DiscretizedFunction<dim> adjoint_notransform(const DiscretizedFunction<dim>& g) {
         // L*
         auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
         tmp->set_norm(this->norm_codomain);
         tmp->dot_solve_mass_and_transform();
         rhs_adj->set_base_rhs(tmp);

         DiscretizedFunction<dim> res(weq.get_mesh());

         if (adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards) {
            AssertThrow((std::dynamic_pointer_cast<ConstantFunction<dim>, LightFunction<dim>>(weq.get_param_nu()) != nullptr),
            ExcMessage("Wrong adjoint because ν≠0!"));

            res = weq.run(rhs_adj, WaveEquation<dim>::Backward);
            res.throw_away_derivative();
         } else if (adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint)
         res = weq_adj.run(rhs_adj);
         else
         Assert(false, ExcInternalError());

         res.set_norm(this->norm_codomain);
         // res.dot_mult_mass_and_transform_inverse();
         res.mult_mass();  // instead of dot_mult_mass_and_transform_inverse+dot_transform

         // M*
         // res.dot_transform();
         res *= -1.0;
         res.pointwise_multiplication(*u);

         res.set_norm(this->norm_domain);
         // res.dot_transform_inverse();

         return res;
      }

      virtual DiscretizedFunction<dim> zero() {
         DiscretizedFunction<dim> res(q->get_mesh());
         res.set_norm(this->norm_domain);

         return res;
      }

   private:
      WaveEquation<dim> weq;
      WaveEquationAdjoint<dim> weq_adj;

      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain;
      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain;

      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

      std::shared_ptr<DiscretizedFunction<dim>> q;
      std::shared_ptr<DiscretizedFunction<dim>> u;

      std::shared_ptr<L2RightHandSide<dim>> rhs;
      std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
   };
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2QPROBLEM_H_ */
