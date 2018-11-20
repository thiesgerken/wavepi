/*
 * AbstractEquation.cpp
 *
 *  Created on: 14.11.2018
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
#include <forward/AbstractEquation.h>

namespace wavepi {
namespace forward {

template<int dim>
void AbstractEquation<dim>::declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection("WaveEquation");
   {
      prm.declare_entry("theta", "0.5", Patterns::Double(0, 1),
            "parameter θ in the time discretization (θ=1 → backward Euler, θ=0 → forward Euler, θ=0.5 → Crank-Nicolson");
      prm.declare_entry("tol", "1e-8", Patterns::Double(0, 1), "relative tolerance for the solution of linear systems");
      prm.declare_entry("max iter", "10000", Patterns::Integer(0),
            "maximum iteration threshold for the solution of linear systems");
   }
   prm.leave_subsection();
}

template<int dim>
void AbstractEquation<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("WaveEquation");
   {
      theta = prm.get_double("theta");
      this->solver_tolerance = prm.get_double("tol");
      this->solver_max_iter = prm.get_double("max iter");
   }
   prm.leave_subsection();
}

template<int dim>
void AbstractEquation<dim>::init_system(size_t first_idx) {
   dof_handler = mesh->get_dof_handler(first_idx);
   sparsity_pattern = mesh->get_sparsity_pattern(first_idx);
   constraints = mesh->get_constraint_matrix(first_idx);

   matrix_A.reinit(*sparsity_pattern);
   matrix_B.reinit(*sparsity_pattern);
   matrix_C.reinit(*sparsity_pattern);

   rhs.reinit(dof_handler->n_dofs());

   solution_u.reinit(dof_handler->n_dofs());
   solution_v.reinit(dof_handler->n_dofs());
   system_rhs.reinit(dof_handler->n_dofs());
   system_rhs.reinit(dof_handler->n_dofs());
   system_tmp1.reinit(dof_handler->n_dofs());
   system_tmp2.reinit(dof_handler->n_dofs());

   double time = mesh->get_time(first_idx);

   right_hand_side->set_time(time);
   initial_values(time);
}

template<int dim>
void AbstractEquation<dim>::cleanup() {
   matrix_A.clear();
   matrix_B.clear();
   matrix_C.clear();

   solution_u.reinit(0);
   solution_v.reinit(0);

   solution_u_old.reinit(0);
   solution_v_old.reinit(0);

   system_rhs.reinit(0);
   system_rhs.reinit(0);

   system_tmp1.reinit(0);
   system_tmp2.reinit(0);

   rhs.reinit(0);
   rhs_old.reinit(0);
}

template<int dim>
void AbstractEquation<dim>::next_mesh(size_t source_idx, size_t target_idx) {
   // TODO: system_tmp2 is a right hand side vector, maybe have to multiply with M^-1 first (and with M after?)
   dof_handler = mesh->transfer(source_idx, target_idx, { &system_tmp1, &system_tmp2 });
   sparsity_pattern = mesh->get_sparsity_pattern(target_idx);
   constraints = mesh->get_constraint_matrix(target_idx);

   matrix_A.reinit(*sparsity_pattern);
   matrix_B.reinit(*sparsity_pattern);
   matrix_C.reinit(*sparsity_pattern);

   system_matrix.reinit(*sparsity_pattern);
   system_rhs.reinit(dof_handler->n_dofs());

   rhs.reinit(dof_handler->n_dofs());

   solution_u.reinit(dof_handler->n_dofs());
   solution_v.reinit(dof_handler->n_dofs());
}

template<int dim>
void AbstractEquation<dim>::assemble(double time) {
   right_hand_side->set_time(time);

   // this helps only a bit because each of the operations is already parallelized
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&AbstractEquation<dim>::assemble_matrices, *this, time);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side, *dof_handler,
         mesh->get_quadrature(), rhs);
   task_group.join_all();
}

template<int dim>
void AbstractEquation<dim>::assemble_pre(double time_step) {
   Vector<double> tmp(solution_u_old.size());

   // grid has not been changed yet,
   // matrix_* contain the matrices of the *last* time step.

   matrix_C.vmult(tmp, solution_v_old);
   system_tmp2.equ(1.0 / time_step, tmp);

   matrix_B.vmult(tmp, solution_v_old);
   system_tmp2.add(-1.0 * (1.0 - theta), tmp);

   matrix_A.vmult(tmp, solution_u_old);
   system_tmp2.add(-1.0 * (1.0 - theta), tmp);

   system_tmp2.add((1.0 - theta), rhs_old);

   // system_tmp2 contains
   // Y^n = (1-theta) * (F^n - B^n V^n - A^n U^n) + 1.0 / dt * C^n V^n
   // TODO: C^n V^n -> C^{n,n-1}, first summand: multiply with (D^n)^-1 D^{n-1} M^{-1} if necessary
   // -> TODO: introduce abstract functions for this?
   // -> TODO: assemble in waveEquation should do these as well, even though they are only needed in the next step
   // -> TODO: waveeq: D^n-1 missing in C^n, add another param. (the real ones!)

   system_tmp1 = solution_u_old;
   system_tmp1 *= 1.0 / time_step;
   system_tmp1.add((1.0 - theta), solution_v_old);

   // system_tmp1 contains
   // X^n = 1/dt U^n + (1-theta) V^n
}

// everything until this point of assembling for u depends on the old mesh and the old matrices
// -> interpolate system_rhs and tmp_u to the new grid and calculate new matrices on new grid

template<int dim>
void AbstractEquation<dim>::assemble_u(double time, double time_step) {
   Vector<double> tmp(solution_u.size());

   system_rhs.equ(theta, system_tmp2);
   system_rhs.add(theta * theta, rhs);

   matrix_C.vmult(tmp, system_tmp1);
   system_rhs.add(1.0 / time_step, tmp);

   matrix_B.vmult(tmp, system_tmp1);
   system_rhs.add(theta, tmp);

   // system_rhs contains
   // theta * \bar Y^n + theta^2 F^{n+1} + (1/dt C^{n+1} + theta * B^{n+1}) \bar X^n

   system_matrix.copy_from(matrix_C);
   system_matrix *= 1.0 / (time_step * time_step);
   system_matrix.add(theta / time_step, matrix_B);
   system_matrix.add(theta * theta, matrix_A);

   // system_matrix contains
   // theta^2 * A^{n+1} + theta/dt * B^{n+1} + 1/dt^2 C^{n+1}

   // needed, because hanging node constraints are not already built into the sparsity pattern
   constraints->condense(system_matrix, system_rhs);

   apply_boundary_conditions_v(time);
}

template<int dim>
void AbstractEquation<dim>::assemble_v(double time, double time_step) {
   Vector<double> tmp(solution_u.size());

   system_rhs.equ(1.0, system_tmp2);
   system_rhs.add(theta, rhs);

   matrix_A.vmult(tmp, solution_u);
   system_rhs.add(-1.0 * theta, tmp);

   // system_rhs contains
   // \bar Y^n + theta * F^{n+1} - theta * A^{n+1} U^{n+1}

   system_matrix.copy_from(matrix_C);
   system_matrix *= 1.0 / time_step;

   system_matrix.add(theta, matrix_B);

   // system_matrix contains
   // theta * B^{n+1} + 1/dt C^{n+1}

   // needed, because hanging node constraints are not already built into the sparsity pattern
   constraints->condense(system_matrix, system_rhs);

   apply_boundary_conditions_v(time);
}

template<int dim>
void AbstractEquation<dim>::solve_u() {
   LogStream::Prefix p("solve_u");

   double norm_rhs = system_rhs.l2_norm();

   SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // Fewer (~half) iterations using preconditioner, but at least in 2D this is still not worth the effort
   // PreconditionSSOR<SparseMatrix<double> > precondition;
   // precondition.initialize (system_matrix, PreconditionSSOR<SparseMatrix<double> >::AdditionalData(.6));
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_u, system_rhs, precondition);
   constraints->distribute(solution_u);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
void AbstractEquation<dim>::solve_v() {
   LogStream::Prefix p("solve_v");

   double norm_rhs = system_rhs.l2_norm();

   SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
   SolverCG<> cg(solver_control);

   // See the comment in solve_u about preconditioning
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_v, system_rhs, precondition);
   constraints->distribute(solution_v);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));

   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

   deallog.flags(f);
}

template<int dim>
DiscretizedFunction<dim> AbstractEquation<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side,
      Direction direction) {
   LogStream::Prefix p("AbstractEq");
   Assert(mesh->length() >= 2, ExcInternalError());
   Assert(mesh->length() < 10000, ExcNotImplemented());

   Timer timer, assembly_timer;
   timer.start();

   // this is going to be the result
   DiscretizedFunction<dim> u(mesh, std::make_shared<InvalidNorm<DiscretizedFunction<dim>>>(), true);

   // save handle to rhs function so that `assemble` can use it
   this->right_hand_side = right_hand_side;

   int first_idx = direction == Backward ? mesh->length() - 1 : 0;

   // set dof_handler to first grid,
   // initialize everything and project/interpolate initial values
   init_system(first_idx);

   // create matrices and rhs for first time step
   assemble(mesh->get_time(first_idx));

   // add initial values to output data
   u.set(first_idx, solution_u, solution_v);

   for (size_t i = 1; i < mesh->length(); i++) {
      LogStream::Prefix pp("step-" + Utilities::int_to_string(i, 4));

      int time_idx = direction == Backward ? mesh->length() - 1 - i : i;
      int last_time_idx = direction == Backward ? mesh->length() - i : i - 1;

      double time = mesh->get_time(time_idx);
      double last_time = mesh->get_time(last_time_idx);
      double dt = time - last_time;

      // u -> u_old, same for v and rhs
      rhs_old = rhs;
      solution_u_old = solution_u;
      solution_v_old = solution_v;

      // vector assembling that needs to take place on the old grid
      assemble_pre(dt);

      // set dof_handler to mesh for this time step,
      // interpolate to new mesh
      next_mesh(last_time_idx, time_idx);

      // assemble new matrices and rhs
      assembly_timer.start();
      assemble(time);
      assembly_timer.stop();

      // finish assembling of rhs_u
      // and solve for $u^i$
      assemble_u(time, dt);
      solve_u();

      // finish assembling of rhs_u
      // and solve for $v^i$
      assemble_v(time, dt);
      solve_v();

      u.set(time_idx, solution_u, solution_v);

      std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
      deallog << "t=" << time << std::scientific << ", ";
      deallog << "‖u‖=" << solution_u.l2_norm() << ", ‖v‖=" << solution_v.l2_norm() << std::endl;
      deallog.flags(f);
   }

   timer.stop();
   std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
   deallog << "solved PDE in " << timer.wall_time() << "s (matrix assembly " << assembly_timer.wall_time() << "s)"
         << std::endl;
   deallog.flags(f);

   cleanup();
   return u;
}

template class AbstractEquation<1> ;
template class AbstractEquation<2> ;
template class AbstractEquation<3> ;

} /* namespace forward */
} /* namespace wavepi */
