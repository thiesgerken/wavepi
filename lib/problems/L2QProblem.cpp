/*
 * L2QProblem.cpp
 *
 *  Created on: 20.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <problems/L2QProblem.h>
#include <iostream>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
L2QProblem<dim>::L2QProblem(WaveEquation<dim>& weq)
      : wave_equation(weq), adjoint_solver(WaveEquationBase<dim>::WaveEquationAdjoint) {
}

template<int dim>
L2QProblem<dim>::L2QProblem(WaveEquation<dim>& weq,
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
      : wave_equation(weq), adjoint_solver(adjoint_solver) {
}

template<int dim>
std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> L2QProblem<dim>::derivative(
      const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u) {
   return std::make_unique<L2QProblem<dim>::Linearization>(this->wave_equation,
         adjoint_solver, q, u);
}

template<int dim>
DiscretizedFunction<dim> L2QProblem<dim>::forward(const DiscretizedFunction<dim>& q) {
   LogStream::Prefix p("eval_forward");

   this->wave_equation.set_param_q(std::make_shared<DiscretizedFunction<dim>>(q));
   this->wave_equation.set_run_direction(WaveEquation<dim>::Forward);

   DiscretizedFunction<dim> res = this->wave_equation.run();
   res.throw_away_derivative();

   return res;
}

template<int dim>
L2QProblem<dim>::Linearization::~Linearization() {
}

template<int dim>
L2QProblem<dim>::Linearization::Linearization(const WaveEquation<dim> &weq,
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver, const DiscretizedFunction<dim>& q,
      const DiscretizedFunction<dim>& u)
      : weq(weq), weq_adj(weq), adjoint_solver(adjoint_solver) {
   this->q = std::make_shared<DiscretizedFunction<dim>>(q);
   this->u = std::make_shared<DiscretizedFunction<dim>>(u);

   this->rhs = std::make_shared<L2RightHandSide<dim>>(this->u);
   this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);

   this->weq.set_right_hand_side(rhs);
   this->weq_adj.set_right_hand_side(rhs_adj);

   this->weq.set_initial_values_u(this->weq.zero);
   this->weq.set_initial_values_v(this->weq.zero);
   this->weq.set_boundary_values_u(this->weq.zero);
   this->weq.set_boundary_values_v(this->weq.zero);

   this->weq.set_param_q(this->q);
   this->weq_adj.set_param_q(this->q);
}

template<int dim>
DiscretizedFunction<dim> L2QProblem<dim>::Linearization::forward(const DiscretizedFunction<dim>& h) {
   LogStream::Prefix p("eval_linearization");

   auto Mh = std::make_shared<DiscretizedFunction<dim>>(h);
   *Mh *= -1.0;
   Mh->pointwise_multiplication(*u);

   rhs->set_base_rhs(Mh);
   weq.set_right_hand_side(rhs);
   weq.set_run_direction(WaveEquation<dim>::Forward);

   DiscretizedFunction<dim> res = weq.run();
   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   res.throw_away_derivative();

   return res;
}

template<int dim>
DiscretizedFunction<dim> L2QProblem<dim>::Linearization::adjoint(const DiscretizedFunction<dim>& g) {
   LogStream::Prefix p("eval_adjoint");

   // L*
   auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
   tmp->set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   tmp->mult_time_mass();
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

   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   res.solve_time_mass();

   // M*
   res.mult_space_time_mass();
   res *= -1.0;
   res.pointwise_multiplication(*u);
   res.solve_space_time_mass();

   return res;
}

template<int dim>
DiscretizedFunction<dim> L2QProblem<dim>::Linearization::zero() {
   DiscretizedFunction<dim> res(q->get_mesh());
   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);

   return res;
}

template class L2QProblem<1> ;
template class L2QProblem<2> ;
template class L2QProblem<3> ;

} /* namespace problems */
} /* namespace wavepi */
