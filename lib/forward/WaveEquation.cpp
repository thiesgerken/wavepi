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

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
WaveEquation<dim>::~WaveEquation() {
}

template<int dim>
WaveEquation<dim>::WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : WaveEquationBase<dim>(mesh), initial_values_u(this->zero), initial_values_v(this->zero), boundary_values_u(
            this->zero), boundary_values_v(this->zero) {
}

template<int dim>
WaveEquation<dim>::WaveEquation(const WaveEquation<dim>& weq)
      : WaveEquationBase<dim>(weq.get_mesh()), initial_values_u(weq.initial_values_u), initial_values_v(
            weq.initial_values_v), boundary_values_u(weq.boundary_values_u), boundary_values_v(
            weq.boundary_values_v) {
   this->set_theta(weq.get_theta());

   this->set_param_c(weq.get_param_c());
   this->set_param_q(weq.get_param_q());
   this->set_param_a(weq.get_param_a());
   this->set_param_nu(weq.get_param_nu());

   this->set_right_hand_side(weq.get_right_hand_side());
}

template<int dim>
WaveEquation<dim>& WaveEquation<dim>::operator=(const WaveEquation<dim>& weq) {
   this->set_mesh(weq.get_mesh());
   this->set_theta(weq.get_theta());

   this->set_param_c(weq.get_param_c());
   this->set_param_q(weq.get_param_q());
   this->set_param_a(weq.get_param_a());
   this->set_param_nu(weq.get_param_nu());

   this->set_right_hand_side(weq.get_right_hand_side());

   initial_values_u = weq.initial_values_u;
   initial_values_v = weq.initial_values_v;
   boundary_values_u = weq.boundary_values_u;
   boundary_values_v = weq.boundary_values_v;

   return *this;
}

template<int dim>
void WaveEquation<dim>::init_system(size_t first_idx) {
   dof_handler = mesh->get_dof_handler(first_idx);

   DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
   DoFTools::make_sparsity_pattern(*dof_handler, dsp);
   sparsity_pattern.copy_from(dsp);

   // std::ofstream out("sparsity_pattern.svg");
   // sparsity_pattern.print_svg(out);

   matrix_A.reinit(sparsity_pattern);
   matrix_B.reinit(sparsity_pattern);
   matrix_C.reinit(sparsity_pattern);

   rhs.reinit(dof_handler->n_dofs());

   solution_u.reinit(dof_handler->n_dofs());
   solution_v.reinit(dof_handler->n_dofs());
   system_rhs_u.reinit(dof_handler->n_dofs());
   system_rhs_v.reinit(dof_handler->n_dofs());
   tmp_u.reinit(dof_handler->n_dofs());

   initial_values_u->set_time(mesh->get_time(first_idx));
   initial_values_v->set_time(mesh->get_time(first_idx));

   /* projecting might make more sense, but VectorTools::project
    leads to a mutex error (deadlock) on my laptop (Core i5 6267U) */
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_u, old_solution_u);
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_v, old_solution_v);
   VectorTools::interpolate(*dof_handler, *initial_values_u, solution_u);
   VectorTools::interpolate(*dof_handler, *initial_values_v, solution_v);
}

template<int dim>
void WaveEquation<dim>::next_mesh(size_t source_idx, size_t target_idx) {
   dof_handler = mesh->transfer(source_idx, target_idx, { &system_rhs_u, &system_rhs_v, &tmp_u });

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
void WaveEquation<dim>::next_step(double time) {
   LogStream::Prefix p("next_step");

   param_a->set_time(time);
   param_nu->set_time(time);
   param_q->set_time(time);
   param_c->set_time(time);

   boundary_values_u->set_time(time);
   boundary_values_v->set_time(time);

   right_hand_side->set_time(time);

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

template<int dim>
void WaveEquation<dim>::assemble_matrices() {
   LogStream::Prefix p("assemble_matrices");

   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquation<dim>::fill_A, *this, *dof_handler, matrix_A);
   task_group += Threads::new_task(&WaveEquation<dim>::fill_B, *this, *dof_handler, matrix_B);
   task_group += Threads::new_task(&WaveEquation<dim>::fill_C, *this, *dof_handler, matrix_C);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side,
         *dof_handler, mesh->get_quadrature(), rhs);
   task_group.join_all();
}

template<int dim>
void WaveEquation<dim>::assemble_u_pre(double time_step) {
   Vector<double> tmp(solution_u_old.size());

   matrix_C_old.vmult(tmp, solution_v_old);
   system_rhs_u.equ(theta * time_step, tmp);

   system_rhs_u.add(theta * (1.0 - theta) * time_step * time_step, rhs_old);

   matrix_B_old.vmult(tmp, solution_v_old);
   system_rhs_u.add(-1.0 * theta * (1.0 - theta) * time_step * time_step, tmp);

   matrix_A_old.vmult(tmp, solution_u_old);
   system_rhs_u.add(-1.0 * theta * (1.0 - theta) * time_step * time_step, tmp);

   tmp_u = solution_u_old;
   tmp_u.add((1.0 - theta) * time_step, solution_v_old);
}

// everything until this point of assembling for u depends on the old mesh and the old matrices
// -> interp system_rhs_u and tmp_u to the new grid and calculate new matrices on new grid

template<int dim>
void WaveEquation<dim>::assemble_u(double time_step) {
   Vector<double> tmp(solution_u.size());

   system_rhs_u.add(theta * theta * time_step * time_step, rhs);

   matrix_C.vmult(tmp, tmp_u);
   system_rhs_u.add(1.0, tmp);

   matrix_B.vmult(tmp, tmp_u);
   system_rhs_u.add(theta * time_step, tmp);

   system_matrix.copy_from(matrix_C);
   system_matrix.add(theta * time_step, matrix_B);
   system_matrix.add(theta * theta * time_step * time_step, matrix_A);

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_u, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs_u);
}

template<int dim>
void WaveEquation<dim>::assemble_v_pre(double time_step) {
   Vector<double> tmp(solution_u_old.size());

   matrix_C_old.vmult(system_rhs_v, solution_v_old);

   system_rhs_v.add((1.0 - theta) * time_step, rhs_old);

   matrix_B_old.vmult(tmp, solution_v_old);
   system_rhs_v.add(-1.0 * (1.0 - theta) * time_step, tmp);

   matrix_A_old.vmult(tmp, solution_u_old);
   system_rhs_v.add(-1.0 * (1.0 - theta) * time_step, tmp);
}

// everything until this point depends on the old mesh and the old matrices
// -> interp system_rhs_v to the new grid and calculate new matrices on new grid

template<int dim>
void WaveEquation<dim>::assemble_v(double time_step) {
   Vector<double> tmp(solution_u.size());

   system_rhs_v.add(theta * time_step, rhs);

   matrix_A.vmult(tmp, solution_u);
   system_rhs_v.add(-1.0 * theta * time_step, tmp);

   system_matrix.copy_from(matrix_C);
   system_matrix.add(theta * time_step, matrix_B);

   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_v, boundary_values);
   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs_v);
}

template<int dim>
void WaveEquation<dim>::solve_u() {
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
void WaveEquation<dim>::solve_v() {
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
DiscretizedFunction<dim> WaveEquation<dim>::run() {
   LogStream::Prefix p("WaveEq");
   Assert(mesh->get_times().size() >= 2, ExcInternalError());
   Assert(mesh->get_times().size() < 10000, ExcNotImplemented());

   Timer timer, assembly_timer;
   timer.start();

   // this is going to be the result
   DiscretizedFunction<dim> u(mesh, true);

   bool backwards = run_direction == Backward;
   int first_idx = backwards ? mesh->get_times().size() - 1 : 0;

   // set dof_handler to first grid,
   // initialize everything and project/interpolate initial values
   init_system(first_idx);

   // create matrices for first time step
   assemble_matrices();

   // add initial values to output data
   u.set(first_idx, solution_u, solution_v);

   for (size_t i = 1; i < mesh->get_times().size(); i++) {
      LogStream::Prefix pp("step-" + Utilities::int_to_string(i, 4));

      int time_idx = backwards ? mesh->get_times().size() - 1 - i : i;
      int last_time_idx = backwards ? mesh->get_times().size() - i : i - 1;

      double time = mesh->get_time(time_idx);
      double last_time = mesh->get_time(last_time_idx);
      double dt = time - last_time;

      // u -> u_old, same for v and matrices
      next_step(time);

      // assembling that needs to take place on the old grid
      assemble_u_pre(dt);
      assemble_v_pre(dt);

      // set dof_handler to mesh for this time step,
      // interpolate to new mesh
      next_mesh(last_time_idx, time_idx);

      // assemble new matrices
      assembly_timer.start();
      assemble_matrices();
      assembly_timer.stop();

      // finish assembling of rhs_u
      // and solve for $u^i$
      assemble_u(dt);
      solve_u();

      // finish assembling of rhs_u
      // and solve for $v^i$
      assemble_v(dt);
      solve_v();

      u.set(time_idx, solution_u, solution_v);

      std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
      deallog << "t=" << time << std::scientific << ", ";
      deallog << "‖u‖=" << solution_u.l2_norm() << ", ‖v‖=" << solution_v.l2_norm() << std::endl;
      deallog.flags(f);
   }

   timer.stop();
   std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
   deallog << "solved pde in " << timer.wall_time() << "s (matrix assembly " << assembly_timer.wall_time()
         << "s)" << std::endl;
   deallog.flags(f);

   return u;
}

template<int dim>
inline std::shared_ptr<Function<dim> > WaveEquation<dim>::get_boundary_values_u() const {
   return boundary_values_u;
}

template<int dim>
inline void WaveEquation<dim>::set_boundary_values_u(std::shared_ptr<Function<dim> > boundary_values_u) {
   this->boundary_values_u = boundary_values_u;
}

template<int dim>
inline std::shared_ptr<Function<dim> > WaveEquation<dim>::get_boundary_values_v() const {
   return boundary_values_v;
}

template<int dim>
inline void WaveEquation<dim>::set_boundary_values_v(std::shared_ptr<Function<dim>> boundary_values_v) {
   this->boundary_values_v = boundary_values_v;
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_initial_values_u() const {
   return initial_values_u;
}

template<int dim>
inline void WaveEquation<dim>::set_initial_values_u(std::shared_ptr<Function<dim>> initial_values_u) {
   this->initial_values_u = initial_values_u;
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_initial_values_v() const {
   return initial_values_v;
}

template<int dim>
inline void WaveEquation<dim>::set_initial_values_v(std::shared_ptr<Function<dim>> initial_values_v) {
   this->initial_values_v = initial_values_v;
}

template<int dim>
inline typename WaveEquation<dim>::Direction WaveEquation<dim>::get_run_direction() const {
   return run_direction;
}

template<int dim>
inline void WaveEquation<dim>::set_run_direction(typename WaveEquation<dim>::Direction run_direction) {
   this->run_direction = run_direction;
}

template class WaveEquation<1> ;
template class WaveEquation<2> ;
template class WaveEquation<3> ;

} /* namespace forward */
} /* namespace wavepi */
