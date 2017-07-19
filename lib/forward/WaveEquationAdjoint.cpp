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
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

#include <forward/MatrixCreator.h>
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
WaveEquationAdjoint<dim>::WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler, const Quadrature<dim> quad)
      : theta(0.5), mesh(mesh), dof_handler(dof_handler), quad(quad), param_c(one), param_nu(zero), param_a(
            one), param_q(zero), right_hand_side(zero_rhs) {
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquationAdjoint<dim>& weq)
      : theta(weq.theta), mesh(weq.mesh), dof_handler(weq.dof_handler), quad(weq.quad), param_c(weq.param_c), param_nu(
            weq.param_nu), param_a(weq.param_a), param_q(weq.param_q), right_hand_side(weq.right_hand_side) {
}

template<int dim>
WaveEquationAdjoint<dim>::WaveEquationAdjoint(const WaveEquation<dim>& weq)
      : theta(weq.get_theta()), mesh(weq.get_mesh()), dof_handler(weq.get_dof_handler()), quad(weq.get_quad()) {
   set_param_c(weq.get_param_c());
   set_param_q(weq.get_param_q());
   set_param_a(weq.get_param_a());
   set_param_nu(weq.get_param_nu());

   set_right_hand_side(weq.get_right_hand_side());
}

template<int dim>
WaveEquationAdjoint<dim>& WaveEquationAdjoint<dim>::operator=(const WaveEquationAdjoint<dim>& weq) {
   theta = weq.theta;
   mesh = weq.mesh;
   dof_handler = weq.dof_handler;
   quad = weq.quad;
   param_c = weq.param_c;
   param_nu = weq.param_nu;
   param_a = weq.param_a;
   param_q = weq.param_q;
   right_hand_side = weq.right_hand_side;

   return *this;
}

template<int dim>
void WaveEquationAdjoint<dim>::init_system() {
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
}

template<int dim>
void WaveEquationAdjoint<dim>::setup_step(double time) {
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
   right_hand_side->set_time(time);

   matrix_A = 0;
   matrix_B = 0;
   matrix_C = 0;
   rhs = 0;

   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_A, *this);
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_B, *this);
   task_group += Threads::new_task(&WaveEquationAdjoint<dim>::fill_C, *this);
   task_group += Threads::new_task(&RightHandSide<dim>::create_right_hand_side, *right_hand_side,
         *dof_handler, quad, rhs);
   task_group.join_all();
}

template<int dim>
void WaveEquationAdjoint<dim>::fill_A() {
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
void WaveEquationAdjoint<dim>::fill_B() {
   if (param_nu_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_B,
            param_nu_disc->get_function_coefficients()[param_nu_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_B, param_nu.get());
}

template<int dim>
void WaveEquationAdjoint<dim>::fill_C() {
   if (param_c_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_C,
            param_c_disc->get_function_coefficients()[param_c_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, matrix_C, param_c.get());
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_u(size_t i) {
   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, ZeroFunction<dim>(1), boundary_values);

   // kind of ugly, but hypercube has 2 boundaries in 1 dimension
   if (dim == 1)
      VectorTools::interpolate_boundary_values(*dof_handler, 1, ZeroFunction<dim>(1), boundary_values);

   if (i == mesh->get_times().size() - 1) { // i == N
   /*
    * (M_N^2)^t (u_N, v_N)^t = (g_N, 0)^t
    *
    * g_N = ((M_N^2)^t)_11 u_N + ((M_N^2)^t)_12 v_N
    *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N + (1-θ) A^N v_N
    *     = [k_N^2 C^N + θ k_N B^N + θ^2 A^N] u_N
    */

      double time_step = mesh->get_times()[i] - mesh->get_times()[i - 1];

      system_rhs = rhs;

      system_matrix = 0.0;
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

      double time_step_last = mesh->get_times()[i + 1] - mesh->get_times()[i];

      system_rhs = rhs;
      Vector<double> tmp(solution_u.size());

      tmp.add(-1.0 * theta * (1 - theta), solution_u_old);
      tmp.add(-1.0 * (1 - theta), solution_v_old);
      matrix_A.vmult_add(system_rhs, tmp);

      tmp = solution_u_old;
      tmp *= 1 / (time_step_last * time_step_last);
      matrix_C_old.vmult_add(system_rhs, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs, tmp);

      // system_matrix = identity
      system_matrix = 0.0;
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

      double time_step = mesh->get_times()[i] - mesh->get_times()[i - 1];
      double time_step_last = mesh->get_times()[i + 1] - mesh->get_times()[i];

      Vector<double> tmp(solution_u.size());
      system_rhs = rhs;

      tmp.add(-1.0 * theta * (1 - theta), solution_u_old);
      tmp.add(-1.0 * (1 - theta), solution_v_old);
      tmp.add(-theta, solution_v);
      matrix_A.vmult_add(system_rhs, tmp);

      tmp = solution_u_old;
      tmp *= 1 / (time_step_last * time_step_last);
      matrix_C_old.vmult_add(system_rhs, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs, tmp);

      system_matrix = 0.0;
      system_matrix.add(1.0 / (time_step * time_step), matrix_C);
      system_matrix.add(theta / time_step, matrix_B);
      system_matrix.add(theta * theta, matrix_A);
   }

   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_u, system_rhs);
}

template<int dim>
void WaveEquationAdjoint<dim>::assemble_v(size_t i) {
   std::map<types::global_dof_index, double> boundary_values;
   VectorTools::interpolate_boundary_values(*dof_handler, 0, ZeroFunction<dim>(1), boundary_values);

   // kind of ugly, but hypercube has 2 boundaries in 1 dimension
   if (dim == 1)
      VectorTools::interpolate_boundary_values(*dof_handler, 1, ZeroFunction<dim>(1), boundary_values);

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

      system_rhs = 0.0;

      // system_matrix = identity
      system_matrix = 0.0;
   } else if (i == 0) {
      /*
       * (u_0, v_0)^t = (g_0, 0)^t - (M_{i+1}^1)^t (u_1, v_1)^t
       *
       * v_0 = - ((M_{i+1}^1)^t)_21 u_1 - ((M_{i+1}^1)^t)_22 v_1
       *     = [θ(k_{i+1} C^i - (1-θ) B^i) + (1-θ) (k_{i+1} C^{i+1} + θ B^{i+1})] u_1
       *       + [k_{i+1} C^i - (1-θ) B^i)] v_1
       */

      double time_step_last = mesh->get_times()[1] - mesh->get_times()[0];

      Vector<double> tmp(solution_u.size());
      system_rhs = 0.0;

      tmp.add(theta, solution_u_old);
      tmp.add(1.0, solution_v_old);

      tmp *= 1 / time_step_last;
      matrix_C.vmult_add(system_rhs, tmp);

      tmp *= -time_step_last * (1 - theta);
      matrix_B.vmult_add(system_rhs, tmp);

      tmp = solution_u_old;
      tmp *= (1 - theta) / time_step_last;
      matrix_C_old.vmult_add(system_rhs, tmp);

      tmp *= time_step_last * theta;
      matrix_B_old.vmult_add(system_rhs, tmp);

      // system_matrix = identity
      system_matrix = 0.0;
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

      double time_step = mesh->get_times()[i] - mesh->get_times()[i - 1];
      double time_step_last = mesh->get_times()[i + 1] - mesh->get_times()[i];

      Vector<double> tmp(solution_u.size());
      system_rhs = 0.0;

      tmp.add(theta, solution_u_old);
      tmp.add(1.0, solution_v_old);

      tmp *= 1 / time_step_last;
      matrix_C.vmult_add(system_rhs, tmp);

      tmp *= -time_step_last * (1 - theta);
      matrix_B.vmult_add(system_rhs, tmp);

      tmp = solution_u_old;
      tmp *= (1 - theta) / time_step_last;
      matrix_C_old.vmult_add(system_rhs, tmp);

      tmp *= time_step_last * theta;
      matrix_B.vmult_add(system_rhs, tmp);

      system_matrix = 0.0;
      system_matrix.add(1.0 / time_step, matrix_C);
      system_matrix.add(theta, matrix_B);
   }

   MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution_v, system_rhs);
}

template<int dim>
void WaveEquationAdjoint<dim>::solve_u(size_t i) {
   LogStream::Prefix p("solve_u");

   if (i == 0)
      solution_u = system_rhs;
   else {

      SolverControl solver_control(2000, this->tolerance * system_rhs.l2_norm());
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
}

template<int dim>
void WaveEquationAdjoint<dim>::solve_v(size_t i) {
   LogStream::Prefix p("solve_v");

   if (i == 0 || i == mesh->get_times().size() - 1)
      solution_v = system_rhs;
   else {
      SolverControl solver_control(2000, this->tolerance * system_rhs.l2_norm());
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
}

template<int dim>
DiscretizedFunction<dim> WaveEquationAdjoint<dim>::run() {
   LogStream::Prefix p("WaveEqAdj");
   Assert(mesh->get_times().size() >= 2, ExcInternalError());
   Assert(mesh->get_times().size() < 10000, ExcNotImplemented());

   Timer timer, setup_timer;
   timer.start();

   // this is going to be the result
   DiscretizedFunction<dim> u(mesh, dof_handler, true);

   // initialize everything
   init_system();

   for (size_t j = 0; j < mesh->get_times().size(); j++) {
      size_t i = mesh->get_times().size() - 1 - j;

      LogStream::Prefix pp("step-" + Utilities::int_to_string(j, 4));
      double time = mesh->get_times()[i];

      setup_timer.start();
      setup_step(time);
      setup_timer.stop();

      // solve for $v^{n+1}$
      assemble_v(i);
      solve_v(i);

      // solve for $u^{n+1}$
      assemble_u(i);
      solve_u(i);

      u.set(i, solution_u, solution_v);

      std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
      deallog << "t=" << time << std::scientific << ", ";
      deallog << "‖u‖=" << solution_u.l2_norm() << ", ‖v‖=" << solution_v.l2_norm() << std::endl;
      deallog.flags(f);
   }

   timer.stop();
   std::ios::fmtflags f(deallog.flags(std::ios_base::fixed));
   deallog << "solved adjoint pde in " << timer.wall_time() << "s (setup " << setup_timer.wall_time() << "s)"
         << std::endl;
   deallog.flags(f);

   return apply_R_transpose(u);
}

// also applies Mass matrix afterwards
template<int dim>
DiscretizedFunction<dim> WaveEquationAdjoint<dim>::apply_R_transpose(
      const DiscretizedFunction<dim>& u) const {
   DiscretizedFunction<dim> res(mesh, dof_handler, false);

   for (size_t j = 0; j < mesh->get_times().size(); j++) {
      Vector<double> tmp(solution_u.size());

      if (j != mesh->get_times().size() - 1) {
         tmp.add(theta * (1 - theta), u.get_function_coefficients()[j + 1]);
         tmp.add(1 - theta, u.get_derivative_coefficients()[j + 1]);
      }

      if (j != 0) {
         tmp.add(theta * theta, u.get_function_coefficients()[j]);
         tmp.add(theta, u.get_derivative_coefficients()[j]);
      }

      // Vector<double> tmp2(solution_u.size());
      // mesh->get_mass_matrix(j)->vmult(tmp2, tmp);
      // res.set(j, tmp2);
      res.set(j, tmp);

   }

   return res;
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquationAdjoint<dim>::get_param_a() const {
   return param_a;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_param_a(std::shared_ptr<Function<dim>> param_a) {
   this->param_a = param_a;
   this->param_a_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_a);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquationAdjoint<dim>::get_param_c() const {
   return param_c;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_param_c(std::shared_ptr<Function<dim>> param_c) {
   this->param_c = param_c;
   this->param_c_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_c);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquationAdjoint<dim>::get_param_nu() const {
   return param_nu;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_param_nu(std::shared_ptr<Function<dim>> param_nu) {
   this->param_nu = param_nu;
   this->param_nu_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_nu);
}

template<int dim>
inline std::shared_ptr<Function<dim>> WaveEquationAdjoint<dim>::get_param_q() const {
   return param_q;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_param_q(std::shared_ptr<Function<dim>> param_q) {
   this->param_q = param_q;
   this->param_q_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_q);
}

template<int dim>
inline std::shared_ptr<RightHandSide<dim>> WaveEquationAdjoint<dim>::get_right_hand_side() const {
   return right_hand_side;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_right_hand_side(
      std::shared_ptr<RightHandSide<dim> > right_hand_side) {
   this->right_hand_side = right_hand_side;
}

template<int dim> double WaveEquationAdjoint<dim>::get_theta() const {
   return theta;
}

template<int dim> void WaveEquationAdjoint<dim>::set_theta(double theta) {
   this->theta = theta;
}

template<int dim>
const Quadrature<dim> WaveEquationAdjoint<dim>::get_quad() const {
   return quad;
}

template<int dim>
void WaveEquationAdjoint<dim>::set_quad(const Quadrature<dim> quad) {
   this->quad = quad;
}

template<int dim>
inline const std::shared_ptr<DoFHandler<dim> > WaveEquationAdjoint<dim>::get_dof_handler() const {
   return dof_handler;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_dof_handler(const std::shared_ptr<DoFHandler<dim> > dof_handler) {
   this->dof_handler = dof_handler;
}

template<int dim>
inline const std::shared_ptr<SpaceTimeMesh<dim> > WaveEquationAdjoint<dim>::get_mesh() const {
   return mesh;
}

template<int dim>
inline void WaveEquationAdjoint<dim>::set_mesh(const std::shared_ptr<SpaceTimeMesh<dim> > mesh) {
   this->mesh = mesh;
}

template<int dim> int WaveEquationAdjoint<dim>::get_special_assembly_tactic() const {
   return special_assembly_tactic;
}

template<int dim> void WaveEquationAdjoint<dim>::set_special_assembly_tactic(int special_assembly_tactic) {
   if (special_assembly_tactic > 0)
      this->special_assembly_tactic = 1;
   else if (special_assembly_tactic < 0)
      this->special_assembly_tactic = -1;
   else
      this->special_assembly_tactic = 0;
}

template class WaveEquationAdjoint<1> ;
template class WaveEquationAdjoint<2> ;
template class WaveEquationAdjoint<3> ;

} /* namespace forward */
} /* namespace wavepi */
