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
      : AbstractEquation<dim>(wave.get_mesh()), initial_values_u(wave.get_initial_values_u()),
            initial_values_v(wave.get_initial_values_v()), boundary_values_u(wave.get_boundary_values_u()),
            boundary_values_v(wave.get_boundary_values_v()) {

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
void WaveEquation<dim>::assemble_matrices(size_t time_idx) {
   LogStream::Prefix p("assemble_matrices");

   this->fill_matrices(mesh, time_idx, *dof_handler, matrix_A, matrix_B, matrix_C);
}

template<int dim>
DiscretizedFunction<dim> WaveEquation<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side,
      typename AbstractEquation<dim>::Direction direction) {
   {
      LogStream::Prefix p("WaveEq");
      LogStream::Prefix pp("BoundChecking");

      // bound checking for ρ and c (if possible)
      // (should not take long compared to the rest and can be very tricky to find out otherwise
      //    -> do it even in release mode)
      if (this->param_c_disc) {
         double c_min, c_max;
         this->param_c_disc->min_max_value(&c_min, &c_max);

         std::stringstream bound_str;
         bound_str << c_min << " ≤ c ≤ " << c_max;

         AssertThrow(c_min > 0, ExcMessage("C is not positive, " + bound_str.str()));
         AssertThrow(1.0 / (c_max * c_max) >= 1e-4, ExcMessage("C is not coercive, " + bound_str.str()));

         deallog << bound_str.str() << std::endl;
      }

      if (this->param_rho_disc) {
         double rho_min, rho_max;
         this->param_rho_disc->min_max_value(&rho_min, &rho_max);

         std::stringstream bound_str;
         bound_str << rho_min << " ≤ ρ ≤ " << rho_max;

         AssertThrow(rho_min > 0, ExcMessage("A and D are not positive, " + bound_str.str()));
         AssertThrow(1.0 / rho_max >= 1e-4, ExcMessage("A and D are not coercive, " + bound_str.str()));

         deallog << bound_str.str() << std::endl;
      }
   }

   return AbstractEquation<dim>::run(right_hand_side, direction);
}

template class WaveEquation<1> ;
template class WaveEquation<2> ;
template class WaveEquation<3> ;

} /* namespace forward */
} /* namespace wavepi */
