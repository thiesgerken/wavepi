/*
 * AbstractEquationAdjoint.cpp
 *
 *  Created on: 16.11.2018
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/patterns.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/types.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <iostream>
#include <map>
#include <string>

#include <base/Norm.h>
#include <forward/AbstractEquationAdjoint.h>

namespace wavepi {
namespace forward {

template<int dim>
void AbstractEquationAdjoint<dim>::next_mesh(size_t source_idx, size_t target_idx) {
   if (source_idx == mesh->length()) {
      dof_handler = mesh->get_dof_handler(target_idx);

      system_rhs_u.reinit(dof_handler->n_dofs());
      system_rhs_v.reinit(dof_handler->n_dofs());

      tmp_u.reinit(dof_handler->n_dofs());
      tmp_v.reinit(dof_handler->n_dofs());
   } else
      dof_handler = mesh->transfer(source_idx, target_idx, { &system_rhs_u, &system_rhs_v, &tmp_u, &tmp_v });

   sparsity_pattern = mesh->get_sparsity_pattern(target_idx);
   constraints = mesh->get_constraint_matrix(target_idx);

   matrix_A.reinit(*sparsity_pattern);
   matrix_B.reinit(*sparsity_pattern);
   matrix_C.reinit(*sparsity_pattern);

   system_matrix.reinit(*sparsity_pattern);

   rhs.reinit(dof_handler->n_dofs());

   solution_u.reinit(dof_handler->n_dofs());
   solution_v.reinit(dof_handler->n_dofs());
}

template<int dim>
void AbstractEquationAdjoint<dim>::cleanup() {
   matrix_A.clear();
   matrix_B.clear();
   matrix_C.clear();

   matrix_A_old.clear();
   matrix_B_old.clear();
   matrix_C_old.clear();

   solution_u.reinit(0);
   solution_v.reinit(0);

   solution_u_old.reinit(0);
   solution_v_old.reinit(0);

   system_rhs_u.reinit(0);
   system_rhs_v.reinit(0);

   tmp_u.reinit(0);
   tmp_v.reinit(0);

   rhs.reinit(0);
}

template<int dim>
void AbstractEquationAdjoint<dim>::next_step(size_t time_idx) {
   LogStream::Prefix p("next_step");

   double time = mesh->get_time(time_idx);
   right_hand_side->set_time(time);

   if (time_idx != mesh->length() - 1) {
      matrix_A_old.reinit(*sparsity_pattern);
      matrix_B_old.reinit(*sparsity_pattern);
      matrix_C_old.reinit(*sparsity_pattern);

      // matrices, solution and right hand side of current time step -> matrices, solution and rhs of last time step
      matrix_A_old.copy_from(matrix_A);
      matrix_B_old.copy_from(matrix_B);
      matrix_C_old.copy_from(matrix_C);

      solution_u_old = solution_u;
      solution_v_old = solution_v;
   }
}

template<int dim>
void AbstractEquationAdjoint<dim>::assemble(double time) {
   right_hand_side->set_time(time);

   // this helps only a bit because each of the operations is already parallelized
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&AbstractEquationAdjoint<dim>::assemble_matrices, *this, time);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side, *dof_handler,
         mesh->get_quadrature(), rhs);
   task_group.join_all();
}

template<int dim>
void AbstractEquationAdjoint<dim>::assemble_u_pre(size_t i) {
   if (i == mesh->length() - 1) {
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
void AbstractEquationAdjoint<dim>::assemble_u(size_t i) {
   if (i == mesh->length() - 1) {
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

      system_matrix = 0.0;  // important because it still holds the matrix for v !!
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

      system_rhs_u += rhs;

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

      system_rhs_u += rhs;

      system_matrix = 0.0;  // important because it still holds the matrix for v !!
      system_matrix.add(1.0 / (time_step * time_step), matrix_C);
      system_matrix.add(theta / time_step, matrix_B);
      system_matrix.add(theta * theta, matrix_A);
   }

   // needed, because hanging node constraints are not already built into the sparsity pattern
   constraints->condense(system_matrix, system_rhs_u);

   apply_boundary_conditions_u(mesh->get_time(i));
}

template<int dim>
void AbstractEquationAdjoint<dim>::assemble_v_pre(size_t i) {
   if (i == mesh->length() - 1) {
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
      tmp_v += solution_v_old;
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
      tmp_v += solution_v_old;
   }
}

template<int dim>
void AbstractEquationAdjoint<dim>::assemble_v(size_t i) {
   if (i == mesh->length() - 1) {
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

      // system_matrix = 0.0; // not needed because matrix was reinited due to possible mesh change
      system_matrix.add(1.0 / time_step, matrix_C);
      system_matrix.add(theta, matrix_B);
   }

   // needed, because hanging node constraints are not already built into the sparsity pattern
   constraints->condense(system_matrix, system_rhs_v);

   apply_boundary_conditions_v(mesh->get_time(i));
}

template<int dim>
void AbstractEquationAdjoint<dim>::solve_u() {
   LogStream::Prefix p("solve_u");

   double norm_rhs = system_rhs_u.l2_norm();

   SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // Fewer (~half) iterations using preconditioner, but at least in 2D this is still not worth the effort
   // PreconditionSSOR<SparseMatrix<double> > precondition;
   // precondition.initialize (system_matrix, PreconditionSSOR<SparseMatrix<double> >::AdditionalData(.6));
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_u, system_rhs_u, precondition);
   constraints->distribute(solution_u);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
void AbstractEquationAdjoint<dim>::solve_v() {
   LogStream::Prefix p("solve_v");

   double norm_rhs = system_rhs_v.l2_norm();

   SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // See the comment in solve_u about preconditioning
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_v, system_rhs_v, precondition);
   constraints->distribute(solution_v);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));

   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
DiscretizedFunction<dim> AbstractEquationAdjoint<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side) {
   LogStream::Prefix p("AbstractEqAdj");
   Assert(mesh->length() >= 2, ExcInternalError());
   Assert(mesh->length() < 10000, ExcNotImplemented());

   Timer timer, assembly_timer;
   timer.start();

   // this is going to be the result
   DiscretizedFunction<dim> u(mesh, std::make_shared<InvalidNorm<DiscretizedFunction<dim>>>(), true);

   // save handle to rhs function so that `assemble` and `next_step` can use it
   this->right_hand_side = right_hand_side;

   for (size_t j = 0; j < mesh->length(); j++) {
      size_t i = mesh->length() - 1 - j;

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
      assemble(time);
      assembly_timer.stop();

      // finish assembling of rhs_v
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
   deallog << "solved adjoint pde in " << timer.wall_time() << "s (setup " << assembly_timer.wall_time() << "s)"
         << std::endl;
   deallog.flags(f);

   cleanup();
   return apply_R_transpose(u);
}

// also applies Mass matrix afterwards
template<int dim>
DiscretizedFunction<dim> AbstractEquationAdjoint<dim>::apply_R_transpose(const DiscretizedFunction<dim>& u) {
   DiscretizedFunction<dim> res(mesh);
   Vector<double> tmp;

   for (size_t j = 0; j < mesh->length(); j++) {
      size_t i = mesh->length() - 1 - j;

      if (i != mesh->length() - 1) {
         tmp.reinit(u[i + 1].size());

         tmp.equ(theta * (1 - theta), u[i + 1]);
         tmp.add(1 - theta, u.get_derivative_coefficients(i + 1));

         mesh->transfer(i + 1, i, { &tmp });
      } else
         tmp.reinit(u[i].size());

      if (i != 0) {
         tmp.add(theta * theta, u[i]);
         tmp.add(theta, u.get_derivative_coefficients(i));
      }

      res[i] = tmp;
   }

   return res;
}

template class AbstractEquationAdjoint<1> ;
template class AbstractEquationAdjoint<2> ;
template class AbstractEquationAdjoint<3> ;

} /* namespace forward */
} /* namespace wavepi */
