/*
 * L2AProblem.cpp
 *
 *  Created on: 27.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <problems/L2AProblem.h>
#include <iostream>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
L2AProblem<dim>::L2AProblem(WaveEquation<dim>& weq)
      : WaveProblem<dim>(weq), adjoint_solver(WaveProblem<dim>::WaveEquationAdjoint) {
}

template<int dim>
L2AProblem<dim>::L2AProblem(WaveEquation<dim>& weq, typename WaveProblem<dim>::L2AdjointSolver adjoint_solver)
      : WaveProblem<dim>(weq), adjoint_solver(adjoint_solver) {
}

template<int dim>
std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> L2AProblem<dim>::derivative(
      const DiscretizedFunction<dim>& a, const DiscretizedFunction<dim>& u) {
   return std::make_unique<L2AProblem<dim>::Linearization>(this->wave_equation,
         WaveProblem<dim>::WaveEquationAdjoint, a, u);
}

template<int dim>
DiscretizedFunction<dim> L2AProblem<dim>::forward(const DiscretizedFunction<dim>& a) {
   LogStream::Prefix p("eval_forward");

   this->wave_equation.set_param_a(std::make_shared<DiscretizedFunction<dim>>(a));
   this->wave_equation.set_run_direction(WaveEquation<dim>::Forward);

   DiscretizedFunction<dim> res = this->wave_equation.run();
   res.throw_away_derivative();

   return res;
}

template<int dim>
L2AProblem<dim>::Linearization::~Linearization() {
}

template<int dim>
L2AProblem<dim>::Linearization::Linearization(const WaveEquation<dim> &weq,
      typename WaveProblem<dim>::L2AdjointSolver adjoint_solver, const DiscretizedFunction<dim>& a,
      const DiscretizedFunction<dim>& u)
      : weq(weq), weq_adj(weq), adjoint_solver(adjoint_solver) {
   this->a = std::make_shared<DiscretizedFunction<dim>>(a);
   this->u = std::make_shared<DiscretizedFunction<dim>>(u);

   this->rhs = std::make_shared<DivRightHandSide<dim>>(this->a, this->u);
   this->rhs_adj = std::make_shared<L2RightHandSide<dim>>(this->u);

   this->weq.set_right_hand_side(rhs);
   this->weq_adj.set_right_hand_side(rhs_adj);

   this->weq.set_initial_values_u(this->weq.zero);
   this->weq.set_initial_values_v(this->weq.zero);
   this->weq.set_boundary_values_u(this->weq.zero);
   this->weq.set_boundary_values_v(this->weq.zero);

   this->weq.set_param_a(this->a);
   this->weq_adj.set_param_a(this->a);
}

template<int dim>
DiscretizedFunction<dim> L2AProblem<dim>::Linearization::forward(const DiscretizedFunction<dim>& h) {
   LogStream::Prefix p("eval_linearization");

   rhs->set_a(std::make_shared<DiscretizedFunction<dim>>(h));
   weq.set_right_hand_side(rhs);
   weq.set_run_direction(WaveEquation<dim>::Forward);

   DiscretizedFunction<dim> res = weq.run();
   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   res.throw_away_derivative();

   return res;
}

template<int dim>
DiscretizedFunction<dim> L2AProblem<dim>::Linearization::adjoint(const DiscretizedFunction<dim>& g) {
   LogStream::Prefix p("eval_adjoint");

   // L*
   auto tmp = std::make_shared<DiscretizedFunction<dim>>(g);
   tmp->set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   tmp->mult_time_mass();
   rhs_adj->set_base_rhs(tmp);

   DiscretizedFunction<dim> res(weq.get_mesh(), weq.get_dof_handler());

   if (adjoint_solver == WaveProblem<dim>::WaveEquationBackwards) {
      AssertThrow((std::dynamic_pointer_cast<ZeroFunction<dim>, Function<dim>>(weq.get_param_nu()) != nullptr),
      ExcMessage("Wrong adjoint because ν≠0!"));

      weq.set_right_hand_side(rhs_adj);
      weq.set_run_direction(WaveEquation<dim>::Backward);
      res = weq.run();
      res.throw_away_derivative();
   } else if (adjoint_solver == WaveProblem<dim>::WaveEquationAdjoint)
      res = weq_adj.run();
   else
      Assert(false, ExcInternalError());

   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   res.solve_time_mass();

   // M*
   res.mult_space_time_mass();

   // TODO: make this work
   // should be - nabla(res)*nabla(u) -> piecewise constant function -> fe spaces do not fit
   AssertThrow(false, ExcNotImplemented());

   res.solve_space_time_mass();

   return res;
}

template<int dim>
DiscretizedFunction<dim> L2AProblem<dim>::Linearization::zero() {
   DiscretizedFunction<dim> res(a->get_mesh(), a->get_dof_handler());
   res.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);

   return res;
}

template<int dim>
bool L2AProblem<dim>::Linearization::progress(
      InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state) {
   deallog << "k=" << state.iteration_number << ": rdisc=" << state.current_discrepancy / state.norm_data;
   deallog << ", norm=" << state.norm_current_estimate;
   deallog << std::endl;
   return true;
}

template class L2AProblem<1> ;
template class L2AProblem<2> ;
template class L2AProblem<3> ;

} /* namespace problems */
} /* namespace wavepi */
