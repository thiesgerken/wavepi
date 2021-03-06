/*
 * ConstantCProblem.h
 *
 *  Created on: 17.07.2019
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_CONSTANTCPROBLEM_H_
#define INCLUDE_PROBLEMS_CONSTANTCPROBLEM_H_

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
class ConstantCProblem: public WaveProblem<dim, Measurement> {
public:
   using WaveProblem<dim, Measurement>::derivative;
   using WaveProblem<dim, Measurement>::forward;

   virtual ~ConstantCProblem() = default;

   ConstantCProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
         std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
         std::shared_ptr<Transformation<dim>> transform, std::shared_ptr<DiscretizedFunction<dim>> background_param)
         : WaveProblem<dim, Measurement>(weq, right_hand_sides, measures, transform, background_param),
               fields(measures.size()) {
   }

protected:
   virtual std::unique_ptr<LinearizedSubProblem<dim>> derivative(size_t i) {
      AssertThrow(this->fields[i], ExcInternalError());

      return std::make_unique<ConstantCProblem<dim, Measurement>::Linearization>(this->wave_equation, this->adjoint_solver,
            this->current_param, this->fields[i], this->norm_domain, this->norm_codomain);
   }

   virtual DiscretizedFunction<dim> forward(size_t i) {
      this->wave_equation.set_param_c(this->current_param);
      DiscretizedFunction<dim> res = this->wave_equation.run(
            std::make_shared<L2RightHandSide<dim>>(this->right_hand_sides[i]), WaveEquation<dim>::Forward);
      res.set_norm(this->norm_codomain);

      // save a copy of res (with derivative)
      this->fields[i] = std::make_shared<DiscretizedFunction<dim>>(res);

      res.throw_away_derivative();
      return res;
   }

private:
   // solutions of the last forward problem
   std::vector<std::shared_ptr<DiscretizedFunction<dim>>> fields;

   class Linearization: public LinearizedSubProblem<dim> {
   public:
      virtual ~Linearization() = default;

      Linearization(const WaveEquation<dim>& weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
            const std::shared_ptr<DiscretizedFunction<dim>> c, std::shared_ptr<DiscretizedFunction<dim>> u,
            std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain,
            std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain)
            : weq(weq), weq_adj(weq), norm_domain(norm_domain), norm_codomain(norm_codomain),
                  adjoint_solver(adjoint_solver) {
         this->c = c;
         this->u = u;

         Assert(u->has_derivative(), ExcInternalError());

         this->rhs = std::make_shared<L2RightHandSide<dim>>(this->u);
         this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);

         this->weq.set_initial_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_initial_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_boundary_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
         this->weq.set_boundary_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));

         this->weq.set_param_c(this->c);
         this->weq_adj.set_param_c(this->c);

         auto rho_cont = weq.get_param_rho();
         rho_discretized = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(rho_cont);
         if (!rho_discretized) rho_discretized = std::make_shared<DiscretizedFunction<dim>>(c->get_mesh(), *rho_cont);
      }

      virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim>& h) {
         auto Mh = std::make_shared<DiscretizedFunction<dim>>(h);
         Mh->pointwise_multiplication(u->derivative());

         // multiply with 2 / c^3
         for (size_t i = 0; i < Mh->length(); i++) {
            Vector<double> &coeff_res = Mh->get_function_coefficients(i);
            const Vector<double> &coeff_c = c->get_function_coefficients(i);

            for (size_t j = 0; j < coeff_res.size(); j++)
               coeff_res[j] *= 2.0 / (coeff_c[j] * coeff_c[j] * coeff_c[j]);
         }

         *Mh = Mh->calculate_derivative();

         // multiply with 1 / rho (discretized)
         for (size_t i = 0; i < Mh->length(); i++) {
            Vector<double> &coeff_res = Mh->get_function_coefficients(i);
            const Vector<double> &coeff_rho = rho_discretized->get_function_coefficients(i);

            for (size_t j = 0; j < coeff_res.size(); j++)
               coeff_res[j] /= coeff_rho[j];
         }

         rhs->set_base_rhs(Mh);

         DiscretizedFunction<dim> res = weq.run(rhs, WaveEquation<dim>::Forward);
         res.set_norm(this->norm_codomain);
         res.throw_away_derivative();

         return res;
      }

      virtual DiscretizedFunction<dim> adjoint_notransform(const DiscretizedFunction<dim>& g) {
         /* L*  */
         auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
         tmp->set_norm(this->norm_codomain);
         tmp->dot_solve_mass_and_transform();
         rhs_adj->set_base_rhs(tmp);

         DiscretizedFunction<dim> res(weq.get_mesh());

         if (adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards) {
            deallog << "Attention: Using adjoint = Backward integration!" << std::endl;

            res = weq.run(rhs_adj, WaveEquation<dim>::Backward);
            res.throw_away_derivative();
         } else if (adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint)
            res = weq_adj.run(rhs_adj);
         else
            Assert(false, ExcInternalError());

         res.set_norm(this->norm_codomain);

         // res.dot_mult_mass_and_transform_inverse();
         res.mult_mass();  // instead of dot_mult_mass_and_transform_inverse+dot_transform

         // calc 2*u'/c^3 * (res/rho)' (last ' as calc_deriv_transpose)

         /* M*  */
         // res.dot_transform();
         res.throw_away_derivative();

         for (size_t i = 0; i < res.length(); i++) {
            Vector<double> &coeff_res = res[i];
            const Vector<double> &coeff_rho = rho_discretized->get_function_coefficients(i);

            for (size_t j = 0; j < coeff_res.size(); j++)
               coeff_res[j] /= coeff_rho[j];
         }

         res = res.calculate_derivative_transpose();

         for (size_t i = 0; i < res.length(); i++) {
            Vector<double> &coeff_res = res[i];
            const Vector<double> &coeff_c = c->get_function_coefficients(i);
            const Vector<double> &coeff_u1 = u->get_derivative_coefficients(i);

            for (size_t j = 0; j < coeff_res.size(); j++)
               coeff_res[j] *= 2.0 * coeff_u1[j] / (coeff_c[j] * coeff_c[j] * coeff_c[j]);
         }

         res.set_norm(this->norm_domain);
         // res.dot_transform_inverse();

        // averaging in time
        Vector<double> &coeff_zero = res[0];
        for (size_t i = 1; i < res.length(); i++) {
          const Vector<double> &coeff_res = res[i];

          for (size_t j = 0; j < coeff_res.size(); j++)
            coeff_zero[j] += coeff_res[j];
        }

        coeff_zero *= 1.0 / res.length();

        for (size_t i = 1; i < res.length(); i++)
          res.set_function_coefficients(i, res[0]);

        return res;
      }

      virtual DiscretizedFunction<dim> zero() {
         DiscretizedFunction<dim> res(c->get_mesh());
         res.set_norm(this->norm_domain);

         return res;
      }

   private:
      WaveEquation<dim> weq;
      WaveEquationAdjoint<dim> weq_adj;

      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain;
      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain;

      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

      std::shared_ptr<DiscretizedFunction<dim>> c;
      std::shared_ptr<DiscretizedFunction<dim>> u;

      std::shared_ptr<L2RightHandSide<dim>> rhs;
      std::shared_ptr<L2RightHandSide<dim>> rhs_adj;

      std::shared_ptr<DiscretizedFunction<dim>> rho_discretized;
   };
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2CPROBLEM_H_ */
