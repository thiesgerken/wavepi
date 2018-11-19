/*
 * WaveEquation.cc
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

/*
 * based on step23.cc from the deal.II tutorials
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <forward/WaveEquation.h>

#include <stddef.h>
#include <iostream>
#include <map>
#include <string>

#include <deal.II/base/data_out_base.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template<int dim>
WaveEquation<dim>::WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : AbstractEquation<dim>(mesh), initial_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1)),
            initial_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1)),
            boundary_values_u(std::make_shared<Functions::ZeroFunction<dim>>(1)),
            boundary_values_v(std::make_shared<Functions::ZeroFunction<dim>>(1)) {
}

template<int dim>
WaveEquation<dim>::WaveEquation(const WaveEquation<dim> &wave)
   : AbstractEquation<dim>(wave.get_mesh()), initial_values_u(wave.get_initial_values_u()), initial_values_v(wave.get_initial_values_v()), boundary_values_u(wave.get_boundary_values_u()), boundary_values_v(wave.get_boundary_values_v()) {

   this->set_param_c(wave.get_param_c());
   this->set_param_nu(wave.get_param_nu());
   this->set_param_a(wave.get_param_a());
   this->set_param_q(wave.get_param_q());

   this->set_theta(wave.get_theta());
   this->set_solver_tolerance(wave.get_solver_tolerance());
   this->set_solver_max_iter(wave.get_solver_max_iter());
}

template<int dim>
void WaveEquation<dim>::apply_boundary_conditions_u(double time) {
   boundary_values_u->set_time(time);
   boundary_values_u->set_time(time);

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_u, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs);
}

template<int dim>
void WaveEquation<dim>::apply_boundary_conditions_v(double time) {
   boundary_values_v->set_time(time);

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_v, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs);
}

template<int dim>
void WaveEquation<dim>::initial_values(double time) {
   initial_values_u->set_time(time);
   initial_values_v->set_time(time);

   /* projecting might make more sense, but VectorTools::project
    leads to a mutex error (deadlock) on my laptop (Core i5 6267U) */
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_u, old_solution_u);
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_v, old_solution_v);
   VectorTools::interpolate(*dof_handler, *initial_values_u, solution_u);
   constraints->distribute(solution_u);

   VectorTools::interpolate(*dof_handler, *initial_values_v, solution_v);
   constraints->distribute(solution_v);
}

template<int dim>
void WaveEquation<dim>::assemble_matrices(double time) {
   LogStream::Prefix p("assemble_matrices");

   param_a->set_time(time);
   param_nu->set_time(time);
   param_q->set_time(time);
   param_c->set_time(time);

   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_A, *this, mesh, *dof_handler, matrix_A);
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_B, *this, mesh, *dof_handler, matrix_B);
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_C, *this, mesh, *dof_handler, matrix_C);
   task_group.join_all();
}

template<int dim>
DiscretizedFunction<dim> WaveEquation<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side,
      typename AbstractEquation<dim>::Direction direction) {
   // bound checking for a and c (if possible)
   // (should not take long compared to the rest and can be very tricky to find out otherwise
   //    -> do it even in release mode)
   if (this->param_c_disc) {
      double cmin, cmax;
      this->param_c_disc->min_max_value(&cmin, &cmax);

      std::stringstream bound_str;
      bound_str << cmin << " <= c <= " << cmax;

      AssertThrow(cmax * cmin >= 0, ExcMessage("C is not coercive, " + bound_str.str()));

      if ((cmax > 0 && cmin < 1e-4) || (cmax > -1e-4 && cmin < 0))
         deallog << "Warning: coercivity of C is low, " << bound_str.str() << std::endl;

      if (cmax < 0 && cmin < 0) deallog << "Warning: C is negative definite, " << bound_str.str() << std::endl;
   }

   if (this->param_a_disc) {
      double amin, amax;
      this->param_a_disc->min_max_value(&amin, &amax);

      std::stringstream bound_str;
      bound_str << amin << " <= a <= " << amax;

      AssertThrow(amin >= 1e-4, ExcMessage("A is not coercive, a_min = " + bound_str.str()));

      if (amin < 1e-4) deallog << "Warning: coercivity of A is low, " << bound_str.str() << std::endl;
   }

   return AbstractEquation<dim>::run(right_hand_side, direction);
}

template class WaveEquation<1> ;
template class WaveEquation<2> ;
template class WaveEquation<3> ;

} /* namespace forward */
} /* namespace wavepi */
