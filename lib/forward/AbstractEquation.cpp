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

template <int dim>
void AbstractEquation<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("WaveEquation");
  {
    prm.declare_entry(
        "theta", "0.5", Patterns::Double(0, 1),
        "parameter θ in the time discretization (θ=1 → backward Euler, θ=0 → forward Euler, θ=0.5 → Crank-Nicolson");
    prm.declare_entry("tol", "1e-8", Patterns::Double(0, 1), "relative tolerance for the solution of linear systems");
    prm.declare_entry("max iter", "10000", Patterns::Integer(0),
                      "maximum iteration threshold for the solution of linear systems");
  }
  prm.leave_subsection();
}

template <int dim>
void AbstractEquation<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("WaveEquation");
  {
    theta                  = prm.get_double("theta");
    this->solver_tolerance = prm.get_double("tol");
    this->solver_max_iter  = prm.get_double("max iter");
  }
  prm.leave_subsection();
}

template <int dim>
void AbstractEquation<dim>::init_system(size_t first_idx) {
  dof_handler      = mesh->get_dof_handler(first_idx);
  sparsity_pattern = mesh->get_sparsity_pattern(first_idx);
  constraints      = mesh->get_constraint_matrix(first_idx);

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

template <int dim>
void AbstractEquation<dim>::cleanup() {
  matrix_A.clear();
  matrix_B.clear();
  matrix_C.clear();

  solution_u.reinit(0);
  solution_v.reinit(0);

  system_rhs.reinit(0);
  system_rhs.reinit(0);

  system_tmp1.reinit(0);
  system_tmp2.reinit(0);

  rhs.reinit(0);
}

template <int dim>
void AbstractEquation<dim>::next_mesh(size_t source_idx, size_t target_idx) {
  // TODO: system_tmp2 is a right hand side vector, maybe have to multiply with M^-1 first (and with M after?)
  dof_handler      = mesh->transfer(source_idx, target_idx, {&system_tmp1, &system_tmp2});
  sparsity_pattern = mesh->get_sparsity_pattern(target_idx);
  constraints      = mesh->get_constraint_matrix(target_idx);

  matrix_A.reinit(*sparsity_pattern);
  matrix_B.reinit(*sparsity_pattern);
  matrix_C.reinit(*sparsity_pattern);

  system_matrix.reinit(*sparsity_pattern);
  system_rhs.reinit(dof_handler->n_dofs());

  rhs.reinit(dof_handler->n_dofs());

  solution_u.reinit(dof_handler->n_dofs());
  solution_v.reinit(dof_handler->n_dofs());
}

template <int dim>
void AbstractEquation<dim>::assemble(size_t i) {
  double time = mesh->get_time(i);
  right_hand_side->set_time(time);

  // this helps only a bit because each of the operations is already parallelized
  Threads::TaskGroup<void> task_group;
  task_group += Threads::new_task(&AbstractEquation<dim>::assemble_matrices, *this, i);
  task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side, *dof_handler,
                                  mesh->get_quadrature(), rhs);
  task_group.join_all();
}

template <int dim>
void AbstractEquation<dim>::assemble_pre(const SparseMatrix<double> &mass_matrix, double time_step) {
  Vector<double> tmp(solution_u.size());

  // grid has not been changed yet,
  // matrix_* contain the matrices of the *last* time step, rhs = rhs_old

  system_tmp2 = rhs;

  matrix_B.vmult(tmp, solution_v);
  system_tmp2.add(-1.0, tmp);

  matrix_A.vmult(tmp, solution_u);
  system_tmp2.add(-1.0, tmp);

  tmp.equ(1.0 - theta, system_tmp2);

  // tmp contains
  // (1-θ) (F^n - B^n V^n - A^n U^n)
  vmult_D_intermediate(mass_matrix, system_tmp2, tmp, this->solver_tolerance);

  vmult_C_intermediate(tmp, solution_v);
  system_tmp2.add(1.0 / time_step, tmp);

  // system_tmp2 contains
  // Y^n = (1-θ) (D^n)^-1 D^{n-1} M^{-1} (F^n - B^n V^n - A^n U^n) + 1/dt * C^{n,n-1} V^n

  system_tmp1.equ(1.0 / time_step, solution_u);
  system_tmp1.add(1.0 - theta, solution_v);

  // system_tmp1 contains
  // X^n = 1/dt U^n + (1-θ) V^n
}

// everything until this point of assembling for u depends on the old mesh and the old matrices
// -> interpolate system_rhs and tmp_u to the new grid and calculate new matrices on new grid

template <int dim>
void AbstractEquation<dim>::assemble_u(double time, double time_step) {
  Vector<double> tmp(solution_u.size());

  system_rhs.equ(theta, system_tmp2);
  system_rhs.add(theta * theta, rhs);

  matrix_C.vmult(tmp, system_tmp1);
  system_rhs.add(1.0 / time_step, tmp);

  matrix_B.vmult(tmp, system_tmp1);
  system_rhs.add(theta, tmp);

  // system_rhs contains
  // θ \bar Y^n + θ² F^{n+1} + (1/dt C^{n+1} + θB^{n+1}) \bar X^n

  system_matrix.copy_from(matrix_C);
  system_matrix *= 1.0 / (time_step * time_step);
  system_matrix.add(theta / time_step, matrix_B);
  system_matrix.add(theta * theta, matrix_A);

  // system_matrix contains
  // θ² A^{n+1} + θ/dt * B^{n+1} + 1/dt² C^{n+1}

  // needed, because hanging node constraints are not already built into the sparsity pattern
  constraints->condense(system_matrix, system_rhs);

  apply_boundary_conditions_u(time);
}

template <int dim>
void AbstractEquation<dim>::assemble_v(double time, double time_step) {
  Vector<double> tmp(solution_u.size());

  system_rhs.equ(1.0, system_tmp2);
  system_rhs.add(theta, rhs);

  matrix_A.vmult(tmp, solution_u);
  system_rhs.add(-1.0 * theta, tmp);

  // system_rhs contains
  // \bar Y^n + θF^{n+1} - θA^{n+1} U^{n+1}

  system_matrix.copy_from(matrix_C);
  system_matrix *= 1.0 / time_step;

  system_matrix.add(theta, matrix_B);

  // system_matrix contains
  // θB^{n+1} + 1/dt C^{n+1}

  // needed, because hanging node constraints are not already built into the sparsity pattern
  constraints->condense(system_matrix, system_rhs);

  apply_boundary_conditions_v(time);
}

template <int dim>
void AbstractEquation<dim>::solve_u() {
  LogStream::Prefix p("solve_u");

  double norm_rhs = system_rhs.l2_norm();

  SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
  SolverCG<> cg(solver_control);

  if (precondition_max_age < 0) {
    PreconditionIdentity precondition = PreconditionIdentity();
    cg.solve(system_matrix, solution_u, system_rhs, precondition);
  } else {
    Timer timer;

    if (precondition_u_age < 0 || precondition_u_age > precondition_max_age) {
      timer.start();

      // precondition_u.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(.8));
      precondition_u.initialize(system_matrix);
      precondition_u_age = 0;

      timer.stop();

      std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
      deallog << "Computed new preconditioner for u in " << timer.wall_time() << "s" << std::endl;
      deallog.flags(f);
    }
    precondition_u_age++;

    timer.restart();
    cg.solve(system_matrix, solution_u, system_rhs, precondition_u);
    timer.stop();

    std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
    deallog << "Solved system for u in " << timer.wall_time() << "s" << std::endl;
    deallog.flags(f);
  }

  constraints->distribute(solution_u);

  std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
  deallog << "Steps: " << solver_control.last_step();
  deallog << ", ‖res‖ = " << solver_control.last_value();
  deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

  deallog.flags(f);
}

template <int dim>
void AbstractEquation<dim>::solve_v() {
  LogStream::Prefix p("solve_v");

  double norm_rhs = system_rhs.l2_norm();

  SolverControl solver_control(this->solver_max_iter, this->solver_tolerance * norm_rhs);
  SolverCG<> cg(solver_control);

  if (precondition_max_age < 0) {
    PreconditionIdentity precondition = PreconditionIdentity();
    cg.solve(system_matrix, solution_v, system_rhs, precondition);
  } else {
    Timer timer;

    if (precondition_v_age < 0 || precondition_v_age > precondition_max_age) {
      timer.restart();

      // precondition_v.initialize(system_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(.8));
      precondition_v.initialize(system_matrix);
      precondition_v_age = 0;

      timer.stop();

      std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
      deallog << "Computed new preconditioner for v in " << timer.wall_time() << "s" << std::endl;
      deallog.flags(f);
    }
    precondition_v_age++;

    timer.restart();
    cg.solve(system_matrix, solution_v, system_rhs, precondition_v);
    timer.stop();

    std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
    deallog << "Solved system for v in " << timer.wall_time() << "s" << std::endl;
    deallog.flags(f);
  }

  constraints->distribute(solution_v);

  std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));

  deallog << "Steps: " << solver_control.last_step();
  deallog << ", ‖res‖ = " << solver_control.last_value();
  deallog << ", ‖rhs‖ = " << norm_rhs << std::endl;

  deallog.flags(f);
}

template <int dim>
DiscretizedFunction<dim> AbstractEquation<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side,
                                                    Direction direction) {
  LogStream::Prefix p("AbstractEq");
  Assert(mesh->length() >= 2, ExcInternalError());

  Timer timer, assembly_timer;
  timer.start();

  // this is going to be the result
  DiscretizedFunction<dim> u(mesh, std::make_shared<InvalidNorm<DiscretizedFunction<dim>>>(), true);

  // save handle to rhs function so that `assemble` can use it
  this->right_hand_side = right_hand_side;

  for (size_t i = 0; i < mesh->length(); i++) {
    LogStream::Prefix pp("step-" + Utilities::int_to_string(i, 4));
    int time_idx = direction == Backward ? mesh->length() - 1 - i : i;

    if (i == 0) {
      // set dof_handler to first grid,
      // initialize everything and project/interpolate initial values
      init_system(time_idx);

      // create matrices and rhs for time step zero (needed by assemble_pre in next step)
      assemble(mesh->get_time(time_idx));
    } else {
      int last_time_idx = direction == Backward ? mesh->length() - i : i - 1;

      double time      = mesh->get_time(time_idx);
      double last_time = mesh->get_time(last_time_idx);
      double dt        = time - last_time;

      // vector assembling that needs to take place on the old grid
      assemble_pre(*mesh->get_mass_matrix(last_time_idx), dt);

      // set dof_handler to mesh for this time step,
      // interpolate temporary vectors to new mesh
      // set solution_u and solution_v to zero
      next_mesh(last_time_idx, time_idx);

      // assemble new matrices and rhs
      assembly_timer.start();
      assemble(time_idx);
      assembly_timer.stop();

      // finish assembling of rhs_u
      // and solve for u^i
      assemble_u(time, dt);
      solve_u();

      // finish assembling of rhs_u
      // and solve for v^i
      assemble_v(time, dt);
      solve_v();
    }

    u.set(time_idx, solution_u, solution_v);

    std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
    deallog << std::setprecision(2) << "t=" << time << std::scientific << ", ";
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

template class AbstractEquation<1>;
template class AbstractEquation<2>;
template class AbstractEquation<3>;

} /* namespace forward */
} /* namespace wavepi */
