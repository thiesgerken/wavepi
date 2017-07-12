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

#include <forward/MatrixCreator.h>
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
WaveEquation<dim>::WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler, const Quadrature<dim> quad)
      : theta(0.5), mesh(mesh), dof_handler(dof_handler), quad(quad), initial_values_u(zero), initial_values_v(
            zero), boundary_values_u(zero), boundary_values_v(zero), param_c(one), param_nu(zero), param_a(
            one), param_q(zero), right_hand_side(zero_rhs) {
}

template<int dim>
WaveEquation<dim>::WaveEquation(const WaveEquation<dim>& weq)
      : theta(weq.theta), mesh(weq.mesh), dof_handler(weq.dof_handler), quad(weq.quad), initial_values_u(
            weq.initial_values_u), initial_values_v(weq.initial_values_v), boundary_values_u(
            weq.boundary_values_u), boundary_values_v(weq.boundary_values_v), param_c(weq.param_c), param_nu(
            weq.param_nu), param_a(weq.param_a), param_q(weq.param_q), right_hand_side(weq.right_hand_side) {
}

template<int dim>
WaveEquation<dim>& WaveEquation<dim>::operator=(const WaveEquation<dim>& weq) {
   theta = weq.theta;
   mesh = weq.mesh;
   dof_handler = weq.dof_handler;
   quad = weq.quad;
   initial_values_u = weq.initial_values_u;
   initial_values_v = weq.initial_values_v;
   boundary_values_u = weq.boundary_values_u;
   boundary_values_v = weq.boundary_values_v;
   param_c = weq.param_c;
   param_nu = weq.param_nu;
   param_a = weq.param_a;
   param_q = weq.param_q;
   right_hand_side = weq.right_hand_side;

   return *this;
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
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_u, old_solution_u);
   //   VectorTools::project(*dof_handler, constraints, QGauss<dim>(3), *initial_values_v, old_solution_v);
   VectorTools::interpolate(*dof_handler, *initial_values_u, solution_u);
   VectorTools::interpolate(*dof_handler, *initial_values_v, solution_v);
}

template<int dim>
void WaveEquation<dim>::setup_step(double time) {
   LogStream::Prefix p("setup_step");

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
   boundary_values_u->set_time(time);
   boundary_values_v->set_time(time);
   right_hand_side->set_time(time);

   matrix_A = 0;
   matrix_B = 0;
   matrix_C = 0;
   rhs = 0;

   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquation<dim>::fill_A, *this);
   task_group += Threads::new_task(&WaveEquation<dim>::fill_B, *this);
   task_group += Threads::new_task(&WaveEquation<dim>::fill_C, *this);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side,
         *dof_handler, quad, rhs);
   task_group.join_all();
}

template<int dim>
void WaveEquation<dim>::fill_A() {
   if ((!param_a_disc && !param_q_disc) || !using_special_assembly())
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, matrix_A, param_a, param_q);
   else if (param_a_disc && !param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, matrix_A,
            param_a_disc->get_function_coefficients()[param_a_disc->get_time_index()], param_q);
   else if (!param_a_disc && param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, matrix_A, param_a,
            param_q_disc->get_function_coefficients()[param_q_disc->get_time_index()]);
   else
      // (param_a_disc && param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, matrix_A,
            param_a_disc->get_function_coefficients()[param_a_disc->get_time_index()],
            param_q_disc->get_function_coefficients()[param_q_disc->get_time_index()]);

}

template<int dim>
void WaveEquation<dim>::fill_B() {
   if (param_nu_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_B,
            param_nu_disc->get_function_coefficients()[param_nu_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_B, param_nu.get());
}

template<int dim>
void WaveEquation<dim>::fill_C() {
   if (param_c_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_C,
            param_c_disc->get_function_coefficients()[param_c_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_C, param_c.get());
}

template<int dim>
void WaveEquation<dim>::assemble_u(double time_step) {
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
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_u, boundary_values);

   system_matrix.copy_from(matrix_C);
   system_matrix.add(theta * time_step, matrix_B);
   system_matrix.add(theta * theta * time_step * time_step, matrix_A);

   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs);
}

template<int dim>
void WaveEquation<dim>::assemble_v(double time_step) {
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
   VectorTools::interpolate_boundary_values(*dof_handler, 0, *boundary_values_v, boundary_values);

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

   // Fewer (~half) iterations using preconditioner, but at least in 2D this is still not worth the effort
   // PreconditionSSOR<SparseMatrix<double> > precondition;
   // precondition.initialize (system_matrix, PreconditionSSOR<SparseMatrix<double> >::AdditionalData(.6));
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_u, system_rhs, precondition);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));
   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << system_rhs.l2_norm() << std::endl;

   deallog.flags(f);
}

template<int dim>
void WaveEquation<dim>::solve_v() {
   LogStream::Prefix p("solve_v");

   SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
   SolverCG<> cg(solver_control);

   // See the comment in solve_u about preconditioning
   PreconditionIdentity precondition = PreconditionIdentity();

   cg.solve(system_matrix, solution_v, system_rhs, precondition);

   std::ios::fmtflags f(deallog.flags(std::ios_base::scientific));

   deallog << "Steps: " << solver_control.last_step();
   deallog << ", ‖res‖ = " << solver_control.last_value();
   deallog << ", ‖rhs‖ = " << system_rhs.l2_norm() << std::endl;

   deallog.flags(f);
}

template<int dim>
DiscretizedFunction<dim> WaveEquation<dim>::run(bool backwards) {
   LogStream::Prefix p("WaveEq");
   Assert(mesh->get_times().size() > 2, ExcInternalError());
   Assert(mesh->get_times().size() < 10000, ExcNotImplemented());

   Timer timer, setup_timer;
   timer.start();

   // initialize everything and project/interpolate initial values
   init_system();

   DiscretizedFunction<dim> u(mesh, dof_handler, true);

   // create matrices for first time step
   int first_idx = backwards ? mesh->get_times().size() - 1 : 0;

   setup_step(mesh->get_times()[first_idx]);

   // add initial values to output data
   u.set(first_idx, solution_u, solution_v);

   for (size_t i = 1; i < mesh->get_times().size(); i++) {
      LogStream::Prefix pp("step-" + Utilities::int_to_string(i, 4));

      int time_idx = backwards ? mesh->get_times().size() - 1 - i : i;
      int last_time_idx = backwards ? mesh->get_times().size() - i : i - 1;

      double time = mesh->get_times()[time_idx];
      double last_time = mesh->get_times()[last_time_idx];

      setup_timer.start();
      setup_step(time);
      setup_timer.stop();

      // solve for $u^{n+1}$
      assemble_u(std::abs(time - last_time));
      solve_u();

      // solve for $v^{n+1}$
      assemble_v(std::abs(time - last_time));
      solve_v();

      u.set(time_idx, solution_u, solution_v);

      std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
      deallog << "t=" << time << std::scientific << ", ";
      deallog << "‖u‖=" << solution_u.l2_norm() << ", ‖v‖=" << solution_v.l2_norm() << std::endl;
      deallog.flags(f);
   }

   timer.stop();
   std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
   deallog << "solved pde in " << timer.wall_time() << "s (setup " << setup_timer.wall_time() << "s)"
         << std::endl;
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
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_param_a() const {
   return param_a;
}

template<int dim>
inline void WaveEquation<dim>::set_param_a(std::shared_ptr<Function<dim>> param_a) {
   this->param_a = param_a;
   this->param_a_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_a);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_param_c() const {
   return param_c;
}

template<int dim>
inline void WaveEquation<dim>::set_param_c(std::shared_ptr<Function<dim>> param_c) {
   this->param_c = param_c;
   this->param_c_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_c);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_param_nu() const {
   return param_nu;
}

template<int dim>
inline void WaveEquation<dim>::set_param_nu(std::shared_ptr<Function<dim>> param_nu) {
   this->param_nu = param_nu;
   this->param_nu_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_nu);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquation<dim>::get_param_q() const {
   return param_q;
}

template<int dim>
inline void WaveEquation<dim>::set_param_q(std::shared_ptr<Function<dim>> param_q) {
   this->param_q = param_q;
   this->param_q_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_q);
}

template<int dim>
inline std::shared_ptr<RightHandSide<dim>> WaveEquation<dim>::get_right_hand_side() const {
   return right_hand_side;
}

template<int dim>
inline void WaveEquation<dim>::set_right_hand_side(std::shared_ptr<RightHandSide<dim> > right_hand_side) {
   this->right_hand_side = right_hand_side;
}

template<int dim> double WaveEquation<dim>::get_theta() const {
   return theta;
}

template<int dim> void WaveEquation<dim>::set_theta(double theta) {
   this->theta = theta;
}

template<int dim> int WaveEquation<dim>::get_special_assembly_tactic() const {
   return special_assembly_tactic;
}

template<int dim> void WaveEquation<dim>::set_special_assembly_tactic(int special_assembly_tactic) {
   if (special_assembly_tactic > 0)
      this->special_assembly_tactic = 1;
   else if (special_assembly_tactic < 0)
      this->special_assembly_tactic = -1;
   else
      this->special_assembly_tactic = 0;
}

template class WaveEquation<1> ;
template class WaveEquation<2> ;
template class WaveEquation<3> ;

} /* namespace forward */
} /* namespace wavepi */
