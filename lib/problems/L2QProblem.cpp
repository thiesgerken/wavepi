/*
 * L2QProblem.cpp
 *
 *  Created on: 20.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <problems/L2QProblem.h>
#include <iostream>

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

namespace wavepi {
namespace problems {

template<int dim>
L2QProblem<dim>::~L2QProblem() {
}

template<int dim>
L2QProblem<dim>::L2QProblem(WaveEquation<dim>& weq)
      : WaveProblem<dim>(weq) {
}

template<int dim>
std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> L2QProblem<dim>::derivative(
      const DiscretizedFunction<dim>& q, const DiscretizedFunction<dim>& u) {
   return std::make_unique<L2QProblem<dim>::Linearization>(this->wave_equation, q, u);
}

template<int dim>
DiscretizedFunction<dim> L2QProblem<dim>::forward(const DiscretizedFunction<dim>& q) {
   LogStream::Prefix p("eval_forward");

   this->wave_equation.set_param_q(std::make_shared<DiscretizedFunction<dim>>(q));

   DiscretizedFunction<dim> res = this->wave_equation.run();
   res.throw_away_derivative();

   return res;
}

template<int dim>
L2QProblem<dim>::Linearization::~Linearization() {
}

template<int dim>
L2QProblem<dim>::Linearization::Linearization(const WaveEquation<dim> &weq, const DiscretizedFunction<dim>& q,
      const DiscretizedFunction<dim>& u)
      : weq(weq), weq_adj(weq) {
   this->q = std::make_shared<DiscretizedFunction<dim>>(q);
   this->u = std::make_shared<DiscretizedFunction<dim>>(u);

   this->rhs = std::make_shared<L2ProductRightHandSide<dim>>(this->u, this->u);
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

   rhs->set_func1(std::make_shared<DiscretizedFunction<dim>>(h));

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
   DiscretizedFunction<dim> res = weq_adj.run();
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
   DiscretizedFunction<dim> res(q->get_mesh(), q->get_dof_handler());
   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);

   return res;
}

template<int dim>
bool L2QProblem<dim>::Linearization::progress(
      InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state) {
   deallog << "k=" << state.iteration_number << ": rdisc=" << state.current_discrepancy / state.norm_data;
   deallog << ", norm=" << state.norm_current_estimate;
   deallog << std::endl;
   return true;
}

template class L2QProblem<1> ;
template class L2QProblem<2> ;
template class L2QProblem<3> ;

} /* namespace problems */
} /* namespace wavepi */
