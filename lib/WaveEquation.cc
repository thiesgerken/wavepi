/*
 * WaveEquation.cc
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

/*
 * based on step23.cc from the deal.II tutorials
 */

#include "WaveEquation.h"

using namespace dealii;

namespace wavepi {

   template class WaveEquation<1> ;
   template class WaveEquation<2> ;
   template class WaveEquation<3> ;

   template<int dim>
   WaveEquation<dim>::WaveEquation(DoFHandler<dim> *dof_hndl)
         : initial_values_u(&zero), initial_values_v(&zero), boundary_values_u(&zero), boundary_values_v(
               &zero), right_hand_side(&zero), param_c(&one), param_nu(&zero), param_a(&one), param_q(
               &zero), theta(0.5), time_end(1), time_step(1. / 64), backwards(false), dof_handler(
               dof_hndl) {
   }

   template<int dim>
   void WaveEquation<dim>::init_system() {
      DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(*dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      // std::ofstream out("sparsity_pattern.svg");
      // sparsity_pattern.print_svg(out);

      matrix_A.reinit(sparsity_pattern);
      matrix_B.reinit(sparsity_pattern);
      matrix_C.reinit(sparsity_pattern);
      matrix_A_old.reinit(sparsity_pattern);
      matrix_B_old.reinit(sparsity_pattern);
      matrix_C_old.reinit(sparsity_pattern);

      system_matrix.reinit(sparsity_pattern);

      solution_u.reinit(dof_handler->n_dofs());
      solution_v.reinit(dof_handler->n_dofs());
      solution_u_old.reinit(dof_handler->n_dofs());
      solution_v_old.reinit(dof_handler->n_dofs());

      rhs.reinit(dof_handler->n_dofs());
      rhs_old.reinit(dof_handler->n_dofs());

      system_rhs.reinit(dof_handler->n_dofs());

      constraints.close();

      /* projecting might make more sense, but VectorTools::project
       leads to a mutex error (deadlock) on my laptop (Core i5 6267U) */
      //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3),*initial_values_u, old_solution_u);
      //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3),*initial_values_v, old_solution_v);
      VectorTools::interpolate(*dof_handler, *initial_values_u, solution_u);
      VectorTools::interpolate(*dof_handler, *initial_values_v, solution_v);
   }

   template<int dim>
   void WaveEquation<dim>::setup_step(double time) {
      // matrices, solution and right hand side of current time step -> matrices, solution and rhs of last time step
      matrix_A_old.copy_from(matrix_A);
      matrix_B_old.copy_from(matrix_B);
      matrix_C_old.copy_from(matrix_C);
      rhs_old = rhs;

      solution_u_old = solution_u;
      solution_v_old = solution_v;

      // setup matrices and right hand side for current time step
      param_a->set_time(time);
      param_nu->set_time(time);
      param_q->set_time(time);
      param_c->set_time(time);
      right_hand_side->set_time(time);
      boundary_values_u->set_time(time);
      boundary_values_v->set_time(time);

      // Note: system_matrix is used as temporary storage
      MatrixCreator::create_mass_matrix(*dof_handler, QGauss<dim>(3), system_matrix, param_q);
      MatrixCreator::create_laplace_matrix(*dof_handler, QGauss<dim>(3), matrix_A, param_a);
      matrix_A.add(1.0, system_matrix);

      MatrixCreator::create_mass_matrix(*dof_handler, QGauss<dim>(3), matrix_B, param_nu);
      MatrixCreator::create_mass_matrix(*dof_handler, QGauss<dim>(3), matrix_C, param_c);

      VectorTools::create_right_hand_side(*dof_handler, QGauss<dim>(3), *right_hand_side, rhs);
   }

   template<int dim>
   void WaveEquation<dim>::assemble_u() {
      Vector<double> tmp(solution_u.size());
      system_rhs = 0.0;

      matrix_C_old.vmult(tmp, solution_v_old);
      system_rhs.add(theta * time_step, tmp);

      system_rhs.add(theta * theta * time_step * time_step, rhs);

      system_rhs.add(theta * (1.0 - theta) * time_step * time_step, rhs_old);

      matrix_B_old.vmult(tmp, solution_v_old);
      system_rhs.add(-1.0 * theta * (1.0 - theta) * time_step * time_step, tmp);

      matrix_A_old.vmult(tmp, solution_u_old);
      system_rhs.add(-1.0 * theta * (1.0 - theta) * time_step * time_step, tmp);

      Vector<double> tmp2(solution_u.size());
      tmp2.add(1.0, solution_u_old);
      tmp2.add((1.0 - theta) * time_step, solution_v_old);

      matrix_C.vmult(tmp, tmp2);
      system_rhs.add(1.0, tmp);

      matrix_B.vmult(tmp, tmp2);
      system_rhs.add(theta * time_step, tmp);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_u,
            boundary_values);

      system_matrix.copy_from(matrix_C);
      system_matrix.add(theta * time_step, matrix_B);
      system_matrix.add(theta * theta * time_step * time_step, matrix_A);

      MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs);
   }

   template<int dim>
   void WaveEquation<dim>::assemble_v() {
      Vector<double> tmp(solution_u.size());
      system_rhs = 0.0;

      matrix_C_old.vmult(tmp, solution_v_old);
      system_rhs.add(1.0, tmp);

      system_rhs.add(theta * time_step, rhs);

      matrix_A.vmult(tmp, solution_u);
      system_rhs.add(-1.0 * theta * time_step, tmp);

      system_rhs.add((1.0 - theta) * time_step, rhs_old);

      matrix_B_old.vmult(tmp, solution_v_old);
      system_rhs.add(-1.0 * (1.0 - theta) * time_step, tmp);

      matrix_A_old.vmult(tmp, solution_u_old);
      system_rhs.add(-1.0 * (1.0 - theta) * time_step, tmp);

      std::map<types::global_dof_index, double> boundary_values;
      VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_v,
            boundary_values);

      system_matrix = 0.0;
      system_matrix.add(1.0, matrix_C);
      system_matrix.add(theta * time_step, matrix_B);

      MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs);
   }

   template<int dim>
   void WaveEquation<dim>::solve_u() {
      LogStream::Prefix p("solve_u");

      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<> cg(solver_control);

      cg.solve(system_matrix, solution_u, system_rhs, PreconditionIdentity());

      deallog << std::scientific;
      deallog << "Steps: " << solver_control.last_step();
      deallog << " ‖res‖ = " << solver_control.last_value();
      deallog << " ‖rhs‖ = " << system_rhs.l2_norm();
      deallog << " ‖sol‖ = " << solution_u.l2_norm();
      deallog << std::fixed << std::endl;
   }

   template<int dim>
   void WaveEquation<dim>::solve_v() {
      LogStream::Prefix p("solve_v");

      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<> cg(solver_control);

      cg.solve(system_matrix, solution_v, system_rhs, PreconditionIdentity());

      deallog << std::scientific;
      deallog << "Steps: " << solver_control.last_step();
      deallog << " ‖res‖ = " << solver_control.last_value();
      deallog << " ‖rhs‖ = " << system_rhs.l2_norm();
      deallog << " ‖sol‖ = " << solution_v.l2_norm();
      deallog << std::fixed << std::endl;
   }

   template<int dim>
   DiscretizedFunction<dim> WaveEquation<dim>::run() {
      LogStream::Prefix p("WaveEq");

      // initialize everything and project/interpolate initial values
      init_system();

      DiscretizedFunction<dim> u(true, (int) std::ceil(time_end / time_step));
      double time = backwards ? time_end : 0.0;

      // create matrices for first time step
      setup_step(time);

      // add initial values to output data
      u.push_back(dof_handler, time, solution_u, solution_v);
      deallog << std::endl;

      for (int i = 1; (!backwards && time < time_end) || (backwards && time > 0.0); i++) {
         time += backwards ? -time_step : time_step;

         deallog << "Time step " << i << " at t=" << time << std::endl;
         setup_step(time);

         // solve for $u^{n+1}$
         assemble_u();
         solve_u();

         // solve for $v^{n+1}$
         assemble_v();
         solve_v();

         u.push_back(dof_handler, time, solution_u, solution_v);
         deallog << std::endl;
      }

      return u;
   }
}
