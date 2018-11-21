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
      : AbstractEquationAdjoint<dim>(mesh), zero(std::make_shared<Functions::ZeroFunction<dim>>(1)) {
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquation<dim> &wave)
      : AbstractEquationAdjoint<dim>(wave.get_mesh()), zero(std::make_shared<Functions::ZeroFunction<dim>>(1)) {

   this->set_param_c(wave.get_param_c());
   this->set_param_nu(wave.get_param_nu());
   this->set_param_rho(wave.get_param_rho());
   this->set_param_q(wave.get_param_q());
   this->set_rho_time_dependent(wave.is_rho_time_dependent());

   this->set_theta(wave.get_theta());
   this->set_solver_tolerance(wave.get_solver_tolerance());
   this->set_solver_max_iter(wave.get_solver_max_iter());
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_matrices(double time) {
   LogStream::Prefix p("assemble_matrices");

   param_rho->set_time(time);
   param_nu->set_time(time);
   param_q->set_time(time);
   param_c->set_time(time);

   this->fill_matrices(mesh, *dof_handler, matrix_A, matrix_B, matrix_C);
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
