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

template <int dim>
void AbstractEquationAdjoint<dim>::next_mesh(size_t source_idx, size_t target_idx) {
  if (source_idx == mesh->length()) {
    dof_handler = mesh->get_dof_handler(target_idx);

    system_rhs_u.reinit(dof_handler->n_dofs());
    system_rhs_v.reinit(dof_handler->n_dofs());

    tmp_u.reinit(dof_handler->n_dofs());
    tmp_v.reinit(dof_handler->n_dofs());
  } else
    dof_handler =
        mesh->transfer(source_idx, target_idx, {&system_rhs_u, &system_rhs_v, &tmp_u, &tmp_v, &tmp_R_adjoint});

  sparsity_pattern = mesh->get_sparsity_pattern(target_idx);
  constraints      = mesh->get_constraint_matrix(target_idx);

  matrix_A.reinit(*sparsity_pattern);
  matrix_B.reinit(*sparsity_pattern);
  matrix_C.reinit(*sparsity_pattern);

  system_matrix.reinit(*sparsity_pattern);

  rhs.reinit(dof_handler->n_dofs());

  solution_u.reinit(dof_handler->n_dofs());
  solution_v.reinit(dof_handler->n_dofs());
}

template <int dim>
void AbstractEquationAdjoint<dim>::cleanup() {
  matrix_A.clear();
  matrix_B.clear();
  matrix_C.clear();

  solution_u.reinit(0);
  solution_v.reinit(0);

  system_rhs_u.reinit(0);
  system_rhs_v.reinit(0);

  tmp_u.reinit(0);
  tmp_v.reinit(0);

  rhs.reinit(0);
}

template <int dim>
void AbstractEquationAdjoint<dim>::next_step(size_t time_idx) {
  LogStream::Prefix p("next_step");

  double time = mesh->get_time(time_idx);
  right_hand_side->set_time(time);
}

template <int dim>
void AbstractEquationAdjoint<dim>::assemble(size_t i) {
  double time = mesh->get_time(i);
  right_hand_side->set_time(time);

  // this helps only a bit because each of the operations is already parallelized
  Threads::TaskGroup<void> task_group;
  task_group += Threads::new_task(&AbstractEquationAdjoint<dim>::assemble_matrices, *this, i);
  task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side, *dof_handler,
                                  mesh->get_quadrature(), rhs);
  task_group.join_all();
}

template <int dim>
void AbstractEquationAdjoint<dim>::assemble_u_pre(size_t i) {
  // grid has not been changed yet,
  // matrix_* contain the matrices of the *last* time step.

  if (i == mesh->length() - 1) {
    /* i == N
     *
     * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
     *
     * g_N = ((M_N^2)^t)_11 u_N + ((M_N^2)^t)_12 v_N
     *     = [k_N^2 C^N + ?? k_N B^N + ??^2 A^N] u_N + (1-??) A^N v_N
     *     = [k_N^2 C^N + ?? k_N B^N + ??^2 A^N] u_N
     */
  } else if (i == 0) {
    /*
     * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
     *
     * u_0 = g_0 + [-??(1-??) A^i + k_{i+1}(k_{i+1} C^{i+1} + ?? B^{i+1})] u_1
     *              - (1-??) A^i v_1
     */

    double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

    Vector<double> tmp = solution_u;
    tmp *= 1 / (time_step_last * time_step_last);
    matrix_C.vmult(system_rhs_u, tmp);

    tmp *= time_step_last * theta;
    matrix_B.vmult_add(system_rhs_u, tmp);

    tmp_u.equ(-1.0 * theta * (1 - theta), solution_u);
    tmp_u.add(-1.0 * (1 - theta), solution_v);
  } else {
    /*
     * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
     *
     * ((M_i^2)^t)_11 u_i = g_i - (M_{i+1}^1)^t_11 u_{i+1} - (M_{i+1}^1)^t_12 v_{i+1} - ((M_i^2)^t)_12 v_i
     * ??????????????????????????????????????????     = g_i + [-??(1-??) A^i + k_{i+1}(k_{i+1} C^{i+1} + ?? B^{i+1})] u_{i+1}
     *        ???             - (1-??) A^i v_{i+1} - ?? A^i v_i
     *        ???
     *        ????????????  =  [k_i^2 C^i + ?? k_i B^i + ??^2 A^i]
     */

    double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

    Vector<double> tmp = solution_u;
    tmp *= 1 / (time_step_last * time_step_last);
    matrix_C.vmult(system_rhs_u, tmp);

    tmp *= time_step_last * theta;
    matrix_B.vmult_add(system_rhs_u, tmp);

    tmp_u.equ(-1.0 * theta * (1 - theta), solution_u);
    tmp_u.add(-1.0 * (1 - theta), solution_v);
  }
}

template <int dim>
void AbstractEquationAdjoint<dim>::assemble_u(size_t i) {
  if (i == mesh->length() - 1) {
    /* i == N
     *
     * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
     *
     * g_N = ((M_N^2)^t)_11 u_N + ((M_N^2)^t)_12 v_N
     *     = [k_N^2 C^N + ?? k_N B^N + ??^2 A^N] u_N + (1-??) A^N v_N
     *     = [k_N^2 C^N + ?? k_N B^N + ??^2 A^N] u_N
     */

    double time_step = mesh->get_time(i) - mesh->get_time(i - 1);

    system_rhs_u = rhs;

    system_matrix = 0.0;  // important because it still holds the matrix for v !!
    system_matrix.add(1.0 / (time_step * time_step), matrix_C);
    system_matrix.add(theta / time_step, matrix_B);
    system_matrix.add(theta * theta, matrix_A);
  } else if (i == 0) {
    /*
     * (u???, v???)^t = (g???, 0)^t - (M_{i+1}??)^t (u???, v???)^t
     *
     * u??? = g??? + [-??(1-??) D^{0,1}M?????A??? + k???(k??? C?? + ?? B??)] u???
     *              - (1-??) D^{0,1}M?????A??? v???
     * tmp_u = -??(1-??) u??? - (1-??) v???
     *
     *        + some D intermediate transposes, but not applied to v_i !
     */

    Vector<double> tmp1(tmp_u.size());
    Vector<double> tmp2(tmp_u.size());

    // M?????D^{i,i+1} before multiplying with A^i, then add to system_rhs_u
    vmult_D_intermediate_transpose(*mesh->get_mass_matrix(i), tmp1, tmp_u, this->solver_tolerance);

    tmp1.add(-theta, solution_v);
    matrix_A.vmult(tmp2, tmp1);
    system_rhs_u.add(1.0, tmp2);

    system_rhs_u += rhs;

    system_matrix = IdentityMatrix(solution_u.size());
  } else {
    /*
     * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
     *
     * ((M_i^2)^t)_11 u_i = g_i - (M_{i+1}^1)^t_11 u_{i+1} - (M_{i+1}^1)^t_12 v_{i+1} - ((M_i^2)^t)_12 v_i
     * ??????????????????????????????????????????     = g_i + [-??(1-??) A^i + k_{i+1}(k_{i+1} C^{i+1} + ?? B^{i+1})] u_{i+1}
     *        ???             - (1-??) A^i v_{i+1} - ?? A^i v_i
     *        ???
     *        ????????????  =  [k_i^2 C^i + ?? k_i B^i + ??^2 A^i]
     *
     *        + some D intermediate transposes, but not applied to v_i !
     */

    Vector<double> tmp1(tmp_v.size());
    Vector<double> tmp2(tmp_v.size());

    // M?????D^{i,i+1} before multiplying with A^i, then add to system_rhs_u
    vmult_D_intermediate_transpose(*mesh->get_mass_matrix(i), tmp1, tmp_u, this->solver_tolerance);

    tmp1.add(-theta, solution_v);
    matrix_A.vmult(tmp2, tmp1);
    system_rhs_u.add(1.0, tmp2);

    system_rhs_u += rhs;

    double time_step = mesh->get_time(i) - mesh->get_time(i - 1);
    system_matrix    = 0.0;  // important because it still holds the matrix for v !!
    system_matrix.add(1.0 / (time_step * time_step), matrix_C);
    system_matrix.add(theta / time_step, matrix_B);
    system_matrix.add(theta * theta, matrix_A);
  }

  // needed, because hanging node constraints are not already built into the sparsity pattern
  constraints->condense(system_matrix, system_rhs_u);

  apply_boundary_conditions_u(mesh->get_time(i));
}

template <int dim>
void AbstractEquationAdjoint<dim>::assemble_v_pre(size_t i) {
  // grid has not been changed yet,
  // matrix_* contain the matrices of the *last* time step.

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
     *     = [??(k_{i+1} C^i - (1-??) B^i) + (1-??) (k_{i+1} C^{i+1} + ?? B^{i+1})] u_1
     *       + [k_{i+1} C^i - (1-??) B^i)] v_1
     */

    double time_step_last = mesh->get_time(1) - mesh->get_time(0);

    Vector<double> tmp = solution_u;
    tmp *= (1 - theta) / time_step_last;
    matrix_C.vmult(system_rhs_v, tmp);

    tmp *= time_step_last * theta;
    matrix_B.vmult_add(system_rhs_v, tmp);

    tmp_v.equ(theta, solution_u);
    tmp_v += solution_v;
  } else {
    /*
     * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
     *
     * ((M_i^2)^t)_22 v_i = - (M_{i+1}^1)^t_21 u_{i+1} - (M_{i+1}^1)^t_22 v_{i+1} - ((M_i^2)^t)_21 u_i
     * ??????????????????????????????????????????     = [??(k_{i+1} C^i - (1-??) B^i) + (1-??) (k_{i+1} C^{i+1} + ?? B^{i+1})] u_{i+1}
     *        ???             + [k_{i+1} C^i - (1-??) B^i)] v_{i+1}
     *        ???
     *        ????????????  =  [k_i C^i + ?? B^i]
     */

    double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

    Vector<double> tmp = solution_u;
    tmp *= (1 - theta) / time_step_last;
    matrix_C.vmult(system_rhs_v, tmp);

    tmp *= time_step_last * theta;
    matrix_B.vmult_add(system_rhs_v, tmp);

    tmp_v.equ(theta, solution_u);
    tmp_v += solution_v;
  }
}

template <int dim>
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

    system_rhs_v  = 0.0;
    system_matrix = IdentityMatrix(solution_v.size());
  } else if (i == 0) {
    /*
     * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
     *
     * v_0 = - ((M_{i+1}^1)^t)_21 u_1 - ((M_{i+1}^1)^t)_22 v_1
     *     = [??(k_{i+1} C^i - (1-??) B^i) + (1-??) (k_{i+1} C^{i+1} + ?? B^{i+1})] u_1
     *       + [k_{i+1} C^i - (1-??) B^i)] v_1
     *
     *        + some D intermediate transposes
     */

    double time_step_last = mesh->get_time(1) - mesh->get_time(0);

    Vector<double> tmp1(tmp_v.size());
    Vector<double> tmp2(tmp_v.size());

    // C^{0,1} instead of C^0
    vmult_C_intermediate(tmp1, tmp_v);
    system_rhs_v.add(1.0 / time_step_last, tmp1);

    // M?????D^{0,1} before multiplying with B^0, then add to system_rhs_v
    vmult_D_intermediate_transpose(*mesh->get_mass_matrix(i), tmp1, tmp_v, this->solver_tolerance);
    matrix_B.vmult(tmp2, tmp1);
    system_rhs_v.add(-1 * (1 - theta), tmp2);

    system_matrix = IdentityMatrix(solution_u.size());
  } else {
    /*
     * (M_i^2)^t (u_i, v_i)^t = (g_i, 0)^t - (M_{i+1}^1)^t (u_{i+1}, v_{i+1})^t
     *
     * ((M_i^2)^t)_22 v_i = - (M_{i+1}^1)^t_21 u_{i+1} - (M_{i+1}^1)^t_22 v_{i+1} - ((M_i^2)^t)_21 u_i
     * ??????????????????????????????????????????     = [??(k_{i+1} C^i - (1-??) B^i) + (1-??) (k_{i+1} C^{i+1} + ?? B^{i+1})] u_{i+1}
     *        ???             + [k_{i+1} C^i - (1-??) B^i)] v_{i+1}
     *        ???
     *        ????????????  =  [k_i C^i + ?? B^i]
     *
     *        + some D intermediate transposes
     */

    double time_step_last = mesh->get_time(i + 1) - mesh->get_time(i);

    Vector<double> tmp1(tmp_v.size());
    Vector<double> tmp2(tmp_v.size());

    // C^{i,i+1} instead of C^i
    vmult_C_intermediate(tmp1, tmp_v);
    system_rhs_v.add(1.0 / time_step_last, tmp1);

    // M?????D^{i,i+1} before multiplying with B^i, then add to system_rhs_v
    vmult_D_intermediate_transpose(*mesh->get_mass_matrix(i), tmp1, tmp_v, this->solver_tolerance);
    matrix_B.vmult(tmp2, tmp1);
    system_rhs_v.add(-1 * (1 - theta), tmp2);

    // system_matrix <- 0 not needed because matrix was reinited due to possible mesh change
    double time_step = mesh->get_time(i) - mesh->get_time(i - 1);
    system_matrix.add(1.0 / time_step, matrix_C);
    system_matrix.add(theta, matrix_B);
  }

  // needed, because hanging node constraints are not already built into the sparsity pattern
  constraints->condense(system_matrix, system_rhs_v);

  apply_boundary_conditions_v(mesh->get_time(i));
}

template <int dim>
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
  deallog << ", ???res??? = " << solver_control.last_value();
  deallog << ", ???rhs??? = " << norm_rhs << std::endl;

  deallog.flags(f);
}

template <int dim>
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
  deallog << ", ???res??? = " << solver_control.last_value();
  deallog << ", ???rhs??? = " << norm_rhs << std::endl;

  deallog.flags(f);
}

template <int dim>
DiscretizedFunction<dim> AbstractEquationAdjoint<dim>::run(std::shared_ptr<RightHandSide<dim>> right_hand_side) {
  LogStream::Prefix p("AbstractEqAdj");
  Assert(mesh->length() >= 2, ExcInternalError());

  Timer timer, assembly_timer;
  timer.start();

  // this is going to be the result
  DiscretizedFunction<dim> res(mesh);

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
    assemble(i);
    assembly_timer.stop();

    // finish assembling of rhs_v
    // and solve for $v^i$
    assemble_v(i);
    solve_v();

    // finish assembling of rhs_u
    // and solve for $u^i$
    assemble_u(i);
    solve_u();

    /* apply R^t */

    if (i < mesh->length() - 1) {
      Vector<double> tmp(solution_u.size());
      vmult_D_intermediate_transpose(*mesh->get_mass_matrix(i), tmp, tmp_R_adjoint, this->solver_tolerance);

      res[i].add(1.0, tmp);
    }

    if (i > 0) {
      tmp_R_adjoint.reinit(solution_u.size());
      tmp_R_adjoint.equ(theta * (1 - theta), solution_u);
      tmp_R_adjoint.add(1 - theta, solution_v);

      // tmp_R_adjoint has to be transferred to grid i-1 first!
      // res[i - 1] += " vmult_D_intermediate_transpose(tmp_R_adjoint) ";

      res[i].add(theta * theta, solution_u);
      res[i].add(theta, solution_v);
    }

    std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
    deallog << "t=" << time << std::scientific << ", ";
    deallog << "???u???=" << solution_u.l2_norm() << ", ???v???=" << solution_v.l2_norm() << std::endl;
    deallog.flags(f);
  }

  timer.stop();
  std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
  deallog << "solved adjoint PDE in " << timer.wall_time() << "s (setup " << assembly_timer.wall_time() << "s)"
          << std::endl;
  deallog.flags(f);

  cleanup();

  return res;
}

template class AbstractEquationAdjoint<1>;
template class AbstractEquationAdjoint<2>;
template class AbstractEquationAdjoint<3>;

} /* namespace forward */
} /* namespace wavepi */
