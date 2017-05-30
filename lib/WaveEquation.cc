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
   WaveEquation<dim>::WaveEquation()
         : initial_values_u(&zero), initial_values_v(&zero), boundary_values_u(&zero), boundary_values_v(
               &zero), right_hand_side(&zero), param_c(&one), param_nu(&zero), param_a(&one), param_q(
               &zero), theta(0.5), time_end(1), time_step(1. / 64), fe(1), dof_handler(
               triangulation), timestep_number(1), time(time_step) {
   }

   template<int dim>
   void WaveEquation<dim>::init_system() {
      time = 0.0;
      timestep_number = 0;

      // GridGenerator::hyper_cube(triangulation, -1, 1);
      GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 2, 2 }));
      triangulation.refine_global(4);

      std::cout << "Number of active cells: " << triangulation.n_active_cells() << std::endl;

      dof_handler.distribute_dofs(fe);

      std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl
            << std::endl;

      DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      std::ofstream out("sparsity_pattern.svg");
      sparsity_pattern.print_svg(out);

      matrix_A.reinit(sparsity_pattern);
      matrix_B.reinit(sparsity_pattern);
      matrix_C.reinit(sparsity_pattern);
      matrix_A_old.reinit(sparsity_pattern);
      matrix_B_old.reinit(sparsity_pattern);
      matrix_C_old.reinit(sparsity_pattern);

      matrix_u.reinit(sparsity_pattern);
      matrix_v.reinit(sparsity_pattern);

      solution_u.reinit(dof_handler.n_dofs());
      solution_v.reinit(dof_handler.n_dofs());
      solution_u_old.reinit(dof_handler.n_dofs());
      solution_v_old.reinit(dof_handler.n_dofs());

      rhs.reinit(dof_handler.n_dofs());
      rhs_old.reinit(dof_handler.n_dofs());

      system_rhs.reinit(dof_handler.n_dofs());

      constraints.close();

      /* projecting might make more sense, but VectorTools::project
       leads to a mutex error (deadlock) on my laptop (Core i5 6267U) */
      //   VectorTools::project(dof_handler, constraints, QGauss<dim>(3),*initial_values_u, old_solution_u);
      //   VectorTools::project(dof_handler, constraints, QGauss<dim>(3),*initial_values_v, old_solution_v);
      VectorTools::interpolate(dof_handler, *initial_values_u, solution_u);
      VectorTools::interpolate(dof_handler, *initial_values_v, solution_v);

      // TODO: setup rhs, matrix_A,B,C for time = 0
      //      MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
      //      MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(3), laplace_matrix);
   }

   template<int dim>
   void WaveEquation<dim>::setup_step() {
      // TODO: copy matrices to old matrices, generate new ones

      matrix_A_old.copy_from(matrix_A);
      matrix_B_old.copy_from(matrix_B);
      matrix_C_old.copy_from(matrix_C);
      rhs_old = rhs;

      solution_u_old = solution_u;
      solution_v_old = solution_v;

      // TODO: setup rhs, matrix_A,B,C for current time
      //      MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
      //      MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(3), laplace_matrix);
   }

   template<int dim>
   void WaveEquation<dim>::solve_u() {
      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<> cg(solver_control);

      cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

      std::cout << "   u-equation: " << solver_control.last_step() << " CG iterations."
            << std::endl;
   }

   template<int dim>
   void WaveEquation<dim>::solve_v() {
      SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
      SolverCG<> cg(solver_control);

      cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

      std::cout << "   v-equation: " << solver_control.last_step() << " CG iterations."
            << std::endl;
   }

   template<int dim>
   void WaveEquation<dim>::output_results() const {
      DataOut<dim> data_out;

      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution_u, "U");
      data_out.add_data_vector(solution_v, "V");

      data_out.build_patches();

      const std::string filename = "solution-" + Utilities::int_to_string(timestep_number, 3)
            + ".vtu";
      std::ofstream output(filename.c_str());
      data_out.write_vtu(output);

      static std::vector<std::pair<double, std::string> > times_and_names;
      times_and_names.push_back(std::pair<double, std::string>(time, filename));
      std::ofstream pvd_output("solution.pvd");
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
   }

   template<int dim>
   void WaveEquation<dim>::run() {
      // setup: u, v contain the initial values, u_old and v_old are not defined.
      init_system();

      // add initial values to output data
      output_results();

      Vector<double> tmp(solution_u.size());
      Vector<double> forcing_terms(solution_u.size());

      for (timestep_number = 1, time = time_step; time <= time_end;
            time += time_step, timestep_number++) {
         std::cout << "Time step " << timestep_number << " at t=" << time << std::endl;
         setup_step();

         // solve for $U^n$
//         mass_matrix.vmult(system_rhs, old_solution_u);
//
//         mass_matrix.vmult(tmp, old_solution_v);
//         system_rhs.add(time_step, tmp);
//
//         laplace_matrix.vmult(tmp, old_solution_u);
//         system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);
//
//         right_hand_side->set_time(time);
//         VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2), *right_hand_side, tmp);
//         forcing_terms = tmp;
//         forcing_terms *= theta * time_step;
//
//         right_hand_side->set_time(time - time_step);
//         VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2), *right_hand_side, tmp);
//
//         forcing_terms.add((1 - theta) * time_step, tmp);
//
//         system_rhs.add(theta * time_step, forcing_terms);
//
//         {
//            boundary_values_u->set_time(time);
//
//            std::map<types::global_dof_index, double> boundary_values;
//            VectorTools::interpolate_boundary_values(dof_handler, 0, *boundary_values_u,
//                  boundary_values);
//
//            matrix_u.copy_from(mass_matrix);
//            matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
//            MatrixTools::apply_boundary_values(boundary_values, matrix_u, solution_u, system_rhs);
//         }
//         solve_u();
//
//         // solve for $V^n$
//         laplace_matrix.vmult(system_rhs, solution_u);
//         system_rhs *= -theta * time_step;
//
//         mass_matrix.vmult(tmp, old_solution_v);
//         system_rhs += tmp;
//
//         laplace_matrix.vmult(tmp, old_solution_u);
//         system_rhs.add(-time_step * (1 - theta), tmp);
//
//         system_rhs += forcing_terms;
//
//         {
//            boundary_values_v->set_time(time);
//
//            std::map<types::global_dof_index, double> boundary_values;
//            VectorTools::interpolate_boundary_values(dof_handler, 0, *boundary_values_v,
//                  boundary_values);
//            matrix_v.copy_from(mass_matrix);
//            MatrixTools::apply_boundary_values(boundary_values, matrix_v, solution_v, system_rhs);
//         }
//         solve_v();

         output_results();
      }
   }
}
