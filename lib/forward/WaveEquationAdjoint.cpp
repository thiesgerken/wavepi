/*
 * WaveEquationAdjoint.cpp
 *
 *  Created on: 17.07.2017
 *      Author: thies
 */

/*
 * based on WaveEquation.cpp
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/identity_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquation.h>

#include <stddef.h>
#include <iostream>
#include <map>
#include <string>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : AbstractEquationAdjoint<dim>(mesh), zero(std::make_shared<Functions::ZeroFunction<dim>>(1)){
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquation<dim> &wave)
   : AbstractEquationAdjoint<dim>(wave.get_mesh()), zero(std::make_shared<Functions::ZeroFunction<dim>>(1)) {

   this->set_param_c(wave.get_param_c());
   this->set_param_nu(wave.get_param_nu());
   this->set_param_a(wave.get_param_a());
   this->set_param_q(wave.get_param_q());

   this->set_theta(wave.get_theta());
   this->set_solver_tolerance(wave.get_solver_tolerance());
   this->set_solver_max_iter(wave.get_solver_max_iter());
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_matrices(double time) {
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
void WaveEquationAdjoint<dim>::apply_boundary_conditions_u(double time __attribute ((unused))) {
   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *this->zero, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs_u);
}

template<int dim>
void WaveEquationAdjoint<dim>::apply_boundary_conditions_v(double time __attribute ((unused))) {
   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *this->zero, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs_v);
}

template class WaveEquationAdjoint<1> ;
template class WaveEquationAdjoint<2> ;
template class WaveEquationAdjoint<3> ;

} /* namespace forward */
} /* namespace wavepi */
