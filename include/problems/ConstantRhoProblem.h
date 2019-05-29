/*
 * ConstantRhoProblem.h
 *
 *  Created on: 13.05.2019
 *      Author: thies
 */

#ifndef PROBLEMS_CONSTANTRHOPROBLEM_H_
#define PROBLEMS_CONSTANTRHOPROBLEM_H_

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

template <int dim, typename Measurement>
class ConstantRhoProblem : public WaveProblem<dim, Measurement> {
 public:
  using WaveProblem<dim, Measurement>::derivative;
  using WaveProblem<dim, Measurement>::forward;

  virtual ~ConstantRhoProblem() = default;

  ConstantRhoProblem(WaveEquation<dim> &weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
                     std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
                     std::shared_ptr<Transformation<dim>> transform,
                     std::shared_ptr<DiscretizedFunction<dim>> background_param)
      : WaveProblem<dim, Measurement>(weq, right_hand_sides, measures, transform, background_param),
        fields(measures.size()) {}

 protected:
  virtual std::unique_ptr<LinearizedSubProblem<dim>> derivative(size_t i) {
    return std::make_unique<ConstantRhoProblem<dim, Measurement>::Linearization>(
        this->wave_equation, this->adjoint_solver, this->current_param, this->fields[i], this->norm_domain,
        this->norm_codomain);
  }

  virtual DiscretizedFunction<dim> forward(size_t i) {
    this->wave_equation.set_param_rho(this->current_param);
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

  class Linearization : public LinearizedSubProblem<dim> {
   public:
    virtual ~Linearization() = default;

    Linearization(const WaveEquation<dim> &weq, typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver,
                  const std::shared_ptr<DiscretizedFunction<dim>> rho, std::shared_ptr<DiscretizedFunction<dim>> u,
                  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain,
                  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain)
        : weq(weq),
          weq_adj(weq),
          norm_domain(norm_domain),
          norm_codomain(norm_codomain),
          adjoint_solver(adjoint_solver) {
      this->rho = rho;
      this->u   = u;

      this->rhs     = nullptr;
      this->rhs_adj = nullptr;
      this->m_adj   = std::make_shared<DivRightHandSideAdjoint<dim>>(this->rho, this->u);

      this->weq.set_initial_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
      this->weq.set_initial_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));
      this->weq.set_boundary_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1));
      this->weq.set_boundary_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1));

      this->weq.set_param_rho(this->rho);
      this->weq_adj.set_param_rho(this->rho);

      auto c_cont   = weq.get_param_c();
      c_discretized = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(c_cont);
      if (!c_discretized) c_discretized = std::make_shared<DiscretizedFunction<dim>>(rho->get_mesh(), *c_cont);
    }

    virtual DiscretizedFunction<dim> forward(const DiscretizedFunction<dim> &h) {
      // a = -h / rho^2
      auto a = std::make_shared<DiscretizedFunction<dim>>(h);

      // multiply with -1.0 / rho ^2
      for (size_t i = 0; i < a->length(); i++) {
        Vector<double> &coeff_a         = a->get_function_coefficients(i);
        const Vector<double> &coeff_rho = rho->get_function_coefficients(i);

        for (size_t j = 0; j < coeff_a.size(); j++)
          coeff_a[j] /= -1.0 * coeff_rho[j] * coeff_rho[j];
      }

      // b = h / rho^2 * (u' / c^2)'
      auto b = std::make_shared<DiscretizedFunction<dim>>(u->derivative());

      // multiply with 1 / c^2
      for (size_t i = 0; i < b->length(); i++) {
        Vector<double> &coeff_res     = b->get_function_coefficients(i);
        const Vector<double> &coeff_c = c_discretized->get_function_coefficients(i);

        for (size_t j = 0; j < coeff_res.size(); j++)
          coeff_res[j] /= coeff_c[j] * coeff_c[j];
      }

      *b = b->calculate_derivative();

      // multiply with h / rho (discretized)
      for (size_t i = 0; i < b->length(); i++) {
        Vector<double> &coeff_res       = b->get_function_coefficients(i);
        const Vector<double> &coeff_rho = rho->get_function_coefficients(i);
        const Vector<double> &coeff_h   = h[i];

        for (size_t j = 0; j < coeff_res.size(); j++)
          coeff_res[j] *= coeff_h[j] / (coeff_rho[j] * coeff_rho[j]);
      }

      this->rhs = std::make_shared<DivRightHandSide<dim>>(a, b, this->u);

      DiscretizedFunction<dim> res = weq.run(rhs, WaveEquation<dim>::Forward);
      res.set_norm(this->norm_codomain);
      res.throw_away_derivative();

      return res;
    }

    virtual DiscretizedFunction<dim> adjoint_notransform(const DiscretizedFunction<dim> &g) {
      // L* : Y -> Y
      auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
      tmp->set_norm(this->norm_codomain);
      tmp->dot_solve_mass_and_transform();
      rhs_adj = std::make_shared<L2RightHandSide<dim>>(tmp);

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
      // res.mult_mass();  // instead of dot_mult_mass_and_transform_inverse+dot_transform

      // M* : Y -> X
      // adj1 <- nabla(res)*nabla(u) / rho^2
      // res.dot_transform();
      m_adj->set_a(std::make_shared<DiscretizedFunction<dim>>(res));
      auto adj1 = m_adj->run_adjoint(res.get_mesh());  // something like `mass * (-nabla(res)*nabla(u))`
      adj1.set_norm(this->norm_domain);

      for (size_t i = 0; i < res.length(); i++) {
        Vector<double> &coeff_adj1      = adj1[i];
        const Vector<double> &coeff_rho = rho->get_function_coefficients(i);

        for (size_t j = 0; j < coeff_adj1.size(); j++)
          coeff_adj1[j] /= -1.0 * coeff_rho[j] * coeff_rho[j];
      }

      // adj2 <- res / rho^2 (u'/c^2)'

      // needed for adj2, but not for adj1 !!!
      res.mult_mass();  // instead of dot_mult_mass_and_transform_inverse+dot_transform

      /* M* */
      // res.dot_transform();
      DiscretizedFunction<dim> adj2 = u->derivative();
      adj2.set_norm(this->norm_domain);
      adj2.throw_away_derivative();

      for (size_t i = 0; i < adj2.length(); i++) {
        Vector<double> &coeff_adj2    = adj2[i];
        const Vector<double> &coeff_c = c_discretized->get_function_coefficients(i);

        for (size_t j = 0; j < coeff_adj2.size(); j++)
          coeff_adj2[j] /= coeff_c[j] * coeff_c[j];
      }

      adj2 = adj2.calculate_derivative();

      for (size_t i = 0; i < res.length(); i++) {
        Vector<double> &coeff_adj2      = adj2[i];
        const Vector<double> &coeff_res = res[i];
        const Vector<double> &coeff_rho = rho->get_function_coefficients(i);

        for (size_t j = 0; j < coeff_adj2.size(); j++)
          coeff_adj2[j] *= coeff_res[j] / (coeff_rho[j] * coeff_rho[j]);
      }

      // return the sum of both
      res = adj1;
      res += adj2;
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
      DiscretizedFunction<dim> res(rho->get_mesh());
      res.set_norm(this->norm_domain);

      return res;
    }

   private:
    WaveEquation<dim> weq;
    WaveEquationAdjoint<dim> weq_adj;

    std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain;
    std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain;

    typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;

    std::shared_ptr<DiscretizedFunction<dim>> rho;
    std::shared_ptr<DiscretizedFunction<dim>> u;

    std::shared_ptr<DivRightHandSide<dim>> rhs;
    std::shared_ptr<L2RightHandSide<dim>> rhs_adj;
    std::shared_ptr<DivRightHandSideAdjoint<dim>> m_adj;

    std::shared_ptr<DiscretizedFunction<dim>> c_discretized;
  };
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2RHOPROBLEM_H_ */
