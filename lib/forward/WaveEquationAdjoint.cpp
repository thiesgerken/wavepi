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

#include <stddef.h>
#include <iostream>
#include <map>
#include <string>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
WaveEquationAdjoint<dim>::~WaveEquationAdjoint() {
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : WaveEquationBase<dim>(mesh) {
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquationAdjoint<dim>& weq)
      : WaveEquationBase<dim>(weq.get_mesh()) {
   this->set_theta(weq.get_theta());

   this->set_param_c(weq.get_param_c());
   this->set_param_q(weq.get_param_q());
   this->set_param_a(weq.get_param_a());
   this->set_param_nu(weq.get_param_nu());

   this->set_right_hand_side(weq.get_right_hand_side());
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquationBase<dim>& weq)
      : WaveEquationBase<dim>(weq.get_mesh()) {
   this->set_theta(weq.get_theta());

   this->set_param_c(weq.get_param_c());
   this->set_param_q(weq.get_param_q());
   this->set_param_a(weq.get_param_a());
   this->set_param_nu(weq.get_param_nu());

   this->set_right_hand_side(weq.get_right_hand_side());
}

template<int dim>
WaveEquationAdjoint<dim>& WaveEquationAdjoint<dim>::operator=(const WaveEquationAdjoint<dim>& weq) {
   this->set_mesh(weq.get_mesh());
   this->set_theta(weq.get_theta());

   this->set_param_c(weq.get_param_c());
   this->set_param_q(weq.get_param_q());
   this->set_param_a(weq.get_param_a());
   this->set_param_nu(weq.get_param_nu());

   this->set_right_hand_side(weq.get_right_hand_side());
   return *this;
}

template<int dim>
void WaveEquationAdjoint<dim>::next_mesh(size_t source_idx, size_t target_idx) {
   if (source_idx == mesh->get_times().size()) {
      dof_handler = mesh->get_dof_handler(target_idx);

      system_rhs_u.reinit(dof_handler->n_dofs());
      system_rhs_v.reinit(dof_handler->n_dofs());

      tmp_u.reinit(dof_handler->n_dofs());
      tmp_v.reinit(dof_handler->n_dofs());
   } else
      dof_handler = mesh->transfer(source_idx, target_idx, { &system_rhs_u, &system_rhs_v, &tmp_u, &tmp_v });

   DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
   DoFTools::make_sparsity_pattern(*dof_handler, dsp);
   sparsity_pattern.copy_from(dsp);

   // std::ofstream out("sparsity_pattern.svg");
   // sparsity_pattern.print_svg(out);

   matrix_A.reinit(sparsity_pattern);
   matrix_B.reinit(sparsity_pattern);
   matrix_C.reinit(sparsity_pattern);

   system_matrix.reinit(sparsity_pattern);

   rhs.reinit(dof_handler->n_dofs());

   solution_u.reinit(dof_handler->n_dofs());
   solution_v.reinit(dof_handler->n_dofs());
}

template<int dim>
void WaveEquationAdjoint<dim>::next_step(size_t time_idx) {
   LogStream::Prefix p("next_step");

   double time = mesh->get_time(time_idx);

   param_a->set_time(time);
   param_nu->set_time(time);
   param_q->set_time(time);
   param_c->set_time(time);

   right_hand_side->set_time(time);

   if (time_idx != mesh->get_times().size() - 1) {
      matrix_A_old.reinit(sparsity_pattern);
      matrix_B_old.reinit(sparsity_pattern);
      matrix_C_old.reinit(sparsity_pattern);

      // matrices, solution and right hand side of current time step -> matrices, solution and rhs of last time step
      matrix_A_old.copy_from(matrix_A);
      matrix_B_old.copy_from(matrix_B);
      matrix_C_old.copy_from(matrix_C);
      rhs_old = rhs;

      solution_u_old = solution_u;
      solution_v_old = solution_v;
   }
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_matrices() {
   LogStream::Prefix p("assemble_matrices");

   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_A, *this, *dof_handler, matrix_A);
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_B, *this, *dof_handler, matrix_B);
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_C, *this, *dof_handler, matrix_C);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side, *dof_handler, mesh->get_quadrature(),
         rhs);
   task_group.join_all();
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_u_pre(size_t i) {
   if (i == mesh->get_times().size() - 1) {
      /* i == N
       *
       * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
       *
       * g_N = ((M_N^2)^t)_11 u_N + ((M_N^2)^t)_12 v_N
       *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N + (1-θ) A^N v_N
       *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N
       */
   } else if (i == 0) {
      /*
       * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
       *
       * u_0 = g_0 + [-θ(1-θ) A^i + k_{i+1}(k_{i+1} C^{i+1} + θ B^{i+1})] u_1
       *              - (1-θ) A^i v_1
       */

      double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

      Vector<double> tmp = solution_u_old;
      tmp *= 1 / (time_step_last * time_step_last);
      matrix_C_old.vmult(system_rhs_u, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs_u, tmp);

      tmp_u.equ(-1.0 * theta * (1 - theta), solution_u_old);
      tmp_u.add(-1.0 * (1 - theta), solution_v_old);
   } else {
      /*
       * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
       *
       * ((M_i^2)^t)_11 u_i = g_i - (M_{i+1}^1)^t_11 u_{i+1} - (M_{i+1}^1)^t_12 v_{i+1} - ((M_i^2)^t)_12 v_i
       * ╰──────┬─────╯     = g_i + [-θ(1-θ) A^i + k_{i+1}(k_{i+1} C^{i+1} + θ B^{i+1})] u_{i+1}
       *        │             - (1-θ) A^i v_{i+1} - θ A^i v_i
       *        │
       *        ╰‒‒‒  =  [k_i^2 C^i + θ k_i B^i + θ^2 A^i]
       */

      double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

      Vector<double> tmp = solution_u_old;
      tmp *= 1 / (time_step_last * time_step_last);
      matrix_C_old.vmult(system_rhs_u, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs_u, tmp);

      tmp_u.equ(-1.0 * theta * (1 - theta), solution_u_old);
      tmp_u.add(-1.0 * (1 - theta), solution_v_old);
   }
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_u(size_t i) {
   if (i == mesh->get_times().size() - 1) {
      /* i == N
       *
       * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
       *
       * g_N = ((M_N^2)^t)_11 u_N + ((M_N^2)^t)_12 v_N
       *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N + (1-θ) A^N v_N
       *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N
       */

      double time_step = mesh->get_time(i) - mesh->get_time(i - 1);

      system_rhs_u = rhs;

      system_matrix.add(1.0 / (time_step * time_step), matrix_C);
      system_matrix.add(theta / time_step, matrix_B);
      system_matrix.add(theta * theta, matrix_A);
   } else if (i == 0) {
      /*
       * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
       *
       * u_0 = g_0 + [-θ(1-θ) A^i + k_{i+1}(k_{i+1} C^{i+1} + θ B^{i+1})] u_1
       *              - (1-θ) A^i v_1
       */

      matrix_A.vmult_add(system_rhs_u, tmp_u);

      system_rhs_u.add(1.0, rhs);

      system_matrix = IdentityMatrix(solution_u.size());
   } else {
      /*
       * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
       *
       * ((M_i^2)^t)_11 u_i = g_i - (M_{i+1}^1)^t_11 u_{i+1} - (M_{i+1}^1)^t_12 v_{i+1} - ((M_i^2)^t)_12 v_i
       * ╰──────┬─────╯     = g_i + [-θ(1-θ) A^i + k_{i+1}(k_{i+1} C^{i+1} + θ B^{i+1})] u_{i+1}
       *        │             - (1-θ) A^i v_{i+1} - θ A^i v_i
       *        │
       *        ╰‒‒‒  =  [k_i^2 C^i + θ k_i B^i + θ^2 A^i]
       */

      double time_step = mesh->get_time(i) - mesh->get_time(i - 1);

      tmp_u.add(-theta, solution_v);
      matrix_A.vmult_add(system_rhs_u, tmp_u);

      system_rhs_u.add(1.0, rhs);

      system_matrix.add(1.0 / (time_step * time_step), matrix_C);
      system_matrix.add(theta / time_step, matrix_B);
      system_matrix.add(theta * theta, matrix_A);
   }

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *this->zero, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs_u);
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_v_pre(size_t i) {
   if (i == mesh->get_times().size() - 1) {
      /*
       * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
       *
       * 0 = ((M_N^2)^t)_21 u_N + ((M_N^2)^t)_22 v_N
       *     \------------/       \------------/
       *           = 0                 /= 0
       *
       *     => v_N = 0
       */
   } else if (i == 0) {
      /*
       * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
       *
       * v_0 = - ((M_{i+1}^1)^t)_21 u_1 - ((M_{i+1}^1)^t)_22 v_1
       *     = [θ(k_{i+1} C^i - (1-θ) B^i) + (1-θ) (k_{i+1} C^{i+1} + θ B^{i+1})] u_1
       *       + [k_{i+1} C^i - (1-θ) B^i)] v_1
       */

      double time_step_last = mesh->get_time(1) - mesh->get_time(0);

      Vector<double> tmp = solution_u_old;
      tmp *= (1 - theta) / time_step_last;
      matrix_C_old.vmult(system_rhs_v, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs_v, tmp);

      tmp_v.equ(theta, solution_u_old);
      tmp_v.add(1.0, solution_v_old);
   } else {
      /*
       * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
       *
       * ((M_i^2)^t)_22 v_i = - (M_{i+1}^1)^t_21 u_{i+1} - (M_{i+1}^1)^t_22 v_{i+1} - ((M_i^2)^t)_21 u_i
       * ╰──────┬─────╯     = [θ(k_{i+1} C^i - (1-θ) B^i) + (1-θ) (k_{i+1} C^{i+1} + θ B^{i+1})] u_{i+1}
       *        │             + [k_{i+1} C^i - (1-θ) B^i)] v_{i+1}
       *        │
       *        ╰‒‒‒  =  [k_i C^i + θ B^i]
       */

      double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

      Vector<double> tmp = solution_u_old;
      tmp *= (1 - theta) / time_step_last;
      matrix_C_old.vmult(system_rhs_v, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs_v, tmp);

      tmp_v.equ(theta, solution_u_old);
      tmp_v.add(1.0, solution_v_old);
   }
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_v(size_t i) {
   if (i == mesh->get_times().size() - 1) {
      /*
       * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
       *
       * 0 = ((M_N^2)^t)_21 u_N + ((M_N^2)^t)_22 v_N
       *     \------------/       \------------/
       *           = 0                 /= 0
       *
       *     => v_N = 0
       */

      system_rhs_v = 0.0;

      system_matrix = IdentityMatrix(solution_u.size());
   } else if (i == 0) {
      /*
       * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
       *
       * v_0 = - ((M_{i+1}^1)^t)_21 u_1 - ((M_{i+1}^1)^t)_22 v_1
       *     = [θ(k_{i+1} C^i - (1-θ) B^i) + (1-θ) (k_{i+1} C^{i+1} + θ B^{i+1})] u_1
       *       + [k_{i+1} C^i - (1-θ) B^i)] v_1
       */

      double time_step_last = mesh->get_time(1) - mesh->get_time(0);

      tmp_v *= 1 / time_step_last;
      matrix_C.vmult_add(system_rhs_v, tmp_v);

      tmp_v *= -time_step_last * (1 - theta);
      matrix_B.vmult_add(system_rhs_v, tmp_v);

      system_matrix = IdentityMatrix(solution_u.size());
   } else {
      /*
       * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
       *
       * ((M_i^2)^t)_22 v_i = - (M_{i+1}^1)^t_21 u_{i+1} - (M_{i+1}^1)^t_22 v_{i+1} - ((M_i^2)^t)_21 u_i
       * ╰──────┬─────╯     = [θ(k_{i+1} C^i - (1-θ) B^i) + (1-θ) (k_{i+1} C^{i+1} + θ B^{i+1})] u_{i+1}
       *        │             + [k_{i+1} C^i - (1-θ) B^i)] v_{i+1}
       *        │
       *        ╰‒‒‒  =  [k_i C^i + θ B^i]
       */

      double time_step = mesh->get_time(i) - mesh->get_time(i - 1);
      double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

      tmp_v *= 1 / time_step_last;
      matrix_C.vmult_add(system_rhs_v, tmp_v);

      tmp_v *= -time_step_last * (1 - theta);
      matrix_B.vmult_add(system_rhs_v, tmp_v);

      system_matrix.add(1.0 / time_step, matrix_C);
      system_matrix.add(theta, matrix_B);
   }

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *this->zero, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs_v);
}

template<int dim>
void WaveEquationAdjoint<dim>::solve_u() {
   LogStream::Prefix p("solve_u");

   double norm_rhs = system_rhs_u.l2_norm();

   SolverControl solver_control(2000, this->tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // Fewer (~half) iterations using preconditioner, but at least in 2D this is still not worth the effort
   // PreconditionSSOR<SparseMatrix<double> > precondition;
   // precondition.initialize (system_matrix, PreconditionSSOR<SparseMatrix<double> >::AdditionalData(.6));
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_u, system_rhs_u, precondition);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
void WaveEquationAdjoint<dim>::solve_v() {
   LogStream::Prefix p("solve_v");

   double norm_rhs = system_rhs_v.l2_norm();

   SolverControl solver_control(2000, this->tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // See the comment in solve_u about preconditioning
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_v, system_rhs_v, precondition);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));

   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
DiscretizedFunction<dim> WaveEquationAdjoint<dim>::run() {
   LogStream::Prefix p("WaveEqAdj");
   Assert(mesh->get_times().size() >= 2, ExcInternalError());
   Assert(mesh->get_times().size() < 10000, ExcNotImplemented());

   Timer timer, assembly_timer;
   timer.start();

   // this is going to be the result
   DiscretizedFunction<dim> u(mesh, true);

   for (size_t j = 0; j < mesh->get_times().size(); j++) {
      size_t i = mesh->get_times().size() - 1 - j;

      LogStream::Prefix pp("step-" + Utilities::int_to_string(j, 4));
      double time = mesh->get_time(i);

      // u -> u_old, same for v and matrices
      next_step(i);

      // assembling that needs to take place on the old grid
      assemble_v_pre(i);
      assemble_u_pre(i);

      // set dof_handler to mesh for this time step,
      // interpolate to new mesh (if j != 0)
      next_mesh(i + 1, i);

      // assemble new matrices
      assembly_timer.start();
      assemble_matrices();
      assembly_timer.stop();

      // finish assembling of rhs_u
      // and solve for $v^i$
      assemble_v(i);
      solve_v();

      // finish assembling of rhs_u
      // and solve for $u^i$
      assemble_u(i);
      solve_u();

      u.set(i, solution_u, solution_v);

      std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
      deallog << "t=" << time << std::scientific << ", ";
      deallog << "‖u‖=" << solution_u.l2_norm() << ", ‖v‖=" << solution_v.l2_norm() << std::endl;
      deallog.flags(f);
   }

   timer.stop();
   std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
   deallog << "solved adjoint pde in " << timer.wall_time() << "s (setup " << assembly_timer.wall_time() << "s)" << std::endl;
   deallog.flags(f);

   return apply_R_transpose(u);
}

// also applies Mass matrix afterwards
template<int dim>
DiscretizedFunction<dim> WaveEquationAdjoint<dim>::apply_R_transpose(const DiscretizedFunction<dim>& u) {
   DiscretizedFunction<dim> res(mesh, false);
   dof_handler = mesh->get_dof_handler(mesh->get_times().size() - 1);

   for (size_t j = 0; j < mesh->get_times().size(); j++) {
      size_t i = mesh->get_times().size() - 1 - j;

      Vector<double> tmp(u.get_function_coefficient(i).size());

      if (i != mesh->get_times().size() - 1) {
         tmp.equ(theta * (1 - theta), u.get_function_coefficient(i + 1));
         tmp.add(1 - theta, u.get_derivative_coefficient(i + 1));

         dof_handler = mesh->transfer(i + 1, i, { &tmp });
      }

      if (i != 0) {
         tmp.add(theta * theta, u.get_function_coefficient(i));
         tmp.add(theta, u.get_derivative_coefficient(i));
      }

      res.set(i, tmp);
   }

   return res;
}

template class WaveEquationAdjoint<1> ;
template class WaveEquationAdjoint<2> ;
template class WaveEquationAdjoint<3> ;

} /* namespace forward */
} /* namespace wavepi */
