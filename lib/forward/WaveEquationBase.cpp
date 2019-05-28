/*
 * WaveEquationBase.cpp
 *
 *  Created on: 23.07.2017
 *      Author: thies
 */

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/numerics/matrix_tools.h>
#include <forward/MatrixCreator.h>
#include <forward/WaveEquationBase.h>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template <int dim>
void WaveEquationBase<dim>::fill_matrices(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx,
                                          SparseMatrix<double> &dst_A, SparseMatrix<double> &dst_B,
                                          SparseMatrix<double> &dst_C) {
  //   const double time = mesh->get_time(time_idx);

  // this helps only a bit because each of the operations is already parallelized
  // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690
  // this assembling could be done even more efficient, by looping through the mesh once and assembling all matrices.

  //   Threads::TaskGroup<void> task_group;

  if (!rho_time_dependent || time_idx == mesh->length() - 1) {
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_A, *this, mesh, time_idx, dst_A);
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_B, *this, mesh, time_idx, dst_B);
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_C, *this, mesh, time_idx, dst_C);

    this->fill_A(mesh, time_idx, dst_A);
    this->fill_B(mesh, time_idx, dst_B);
    this->fill_C(mesh, time_idx, dst_C);

    matrix_C_intermediate.clear();
    matrix_D_intermediate.clear();
  } else {
    // possible here because fill_*_intermediate can access different time steps of ρ in parallel
    // (ρ is discretized and this is exploited by assembly tasks)

    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_A, *this, mesh, time_idx, dst_A);
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_B, *this, mesh, time_idx, dst_B);
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_C, *this, mesh, time_idx, dst_C);

    this->fill_A(mesh, time_idx, dst_A);
    this->fill_B(mesh, time_idx, dst_B);
    this->fill_C(mesh, time_idx, dst_C);

    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_C_intermediate, *this, mesh, time_idx);
    //      task_group += Threads::new_task(&WaveEquationBase<dim>::fill_D_intermediate, *this, mesh, time_idx);

    this->fill_C_intermediate(mesh, time_idx);
    this->fill_D_intermediate(mesh, time_idx);
  }

  //   task_group.join_all();
}

// let dst <- (D^n)^{-1} D^{n-1} M^{-1} src
// ( i.e. dst <- src for time-independent D)
template <int dim>
void WaveEquationBase<dim>::vmult_D_intermediate(const SparseMatrix<double> &mass_matrix, Vector<double> &dst,
                                                 const Vector<double> &src, double tolerance) const {
  if (rho_time_dependent) {
    Vector<double> tmp(src.size());

    SolverControl solver_control(2000, tolerance * src.l2_norm());
    SolverCG<> cg(solver_control);

    // PreconditionIdentity precondition = PreconditionIdentity();
    PreconditionSSOR<SparseMatrix<double>> precondition;
    precondition.initialize(mass_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

    cg.solve(mass_matrix, tmp, src, precondition);

    AssertThrow(matrix_D_intermediate.n() > 0, ExcInternalError("matrix_D_intermediate is missing"));
    matrix_D_intermediate.vmult(dst, tmp);
  } else {
    dst.equ(1.0, src);
  }
}

// let dst <- M^{-1} (D^n)^{-1} D^{n-1} src
// ( i.e. dst <- src for time-independent D)
template <int dim>
void WaveEquationBase<dim>::vmult_D_intermediate_transpose(const SparseMatrix<double> &mass_matrix, Vector<double> &dst,
                                                           const Vector<double> &src, double tolerance) const {
  if (rho_time_dependent) {
    Vector<double> tmp(src.size());

    AssertThrow(matrix_D_intermediate.n() > 0, ExcInternalError("matrix_D_intermediate is missing"));
    matrix_D_intermediate.vmult(tmp, src);

    SolverControl solver_control(2000, tolerance * tmp.l2_norm());
    SolverCG<> cg(solver_control);
  
    // PreconditionIdentity precondition = PreconditionIdentity();
    PreconditionSSOR<SparseMatrix<double>> precondition;
    precondition.initialize(mass_matrix, PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

    cg.solve(mass_matrix, dst, tmp, precondition);
  } else {
    dst.equ(1.0, src);
  }
}

// before mesh change, let dst <- (D^n)^{-1} C^{n-1} src
// ( i.e. dst <- matrix_C * src for time-independent D)
template <int dim>
void WaveEquationBase<dim>::vmult_C_intermediate(const SparseMatrix<double> &matrix_C, Vector<double> &dst,
                                                 const Vector<double> &src) const {
  if (rho_time_dependent) {
    AssertThrow(matrix_C_intermediate.n() > 0, ExcInternalError("matrix_C_intermediate is missing"));
    matrix_C_intermediate.vmult(dst, src);
  } else {
    matrix_C.vmult(dst, src);
  }
}

template <int dim>
void WaveEquationBase<dim>::fill_A(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx,
                                   SparseMatrix<double> &destination) {
  const double time = mesh->get_time(time_idx);
  auto dof_handler  = mesh->get_dof_handler(time_idx);

  if ((!param_rho_disc && !param_q_disc) || !using_special_assembly(mesh))
    MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho, param_q, time);
  else if (param_rho_disc && !param_q_disc)
    MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination,
                                        param_rho_disc->get_function_coefficients_by_time(time), param_q, time);
  else if (!param_rho_disc && param_q_disc)
    MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho,
                                        param_q_disc->get_function_coefficients_by_time(time), time);
  else
    // (param_rho_disc && param_q_disc)
    MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination,
                                        param_rho_disc->get_function_coefficients_by_time(time),
                                        param_q_disc->get_function_coefficients_by_time(time));
}

template <int dim>
void WaveEquationBase<dim>::fill_B(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx,
                                   SparseMatrix<double> &destination) {
  const double time = mesh->get_time(time_idx);
  auto dof_handler  = mesh->get_dof_handler(time_idx);

  if (param_nu_disc && using_special_assembly(mesh))
    MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
                                           param_nu_disc->get_function_coefficients_by_time(time));
  else
    MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_nu, time);
}

template <int dim>
void WaveEquationBase<dim>::fill_C(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx,
                                   SparseMatrix<double> &destination) {
  const double time = mesh->get_time(time_idx);
  auto dof_handler  = mesh->get_dof_handler(time_idx);

  if ((!param_rho_disc && !param_c_disc) || !using_special_assembly(mesh))
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho, param_c, time,
                                        time);
  else if (param_rho_disc && !param_c_disc)
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination,
                                        param_rho_disc->get_function_coefficients_by_time(time), param_c, time);
  else if (!param_rho_disc && param_c_disc)
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho,
                                        param_c_disc->get_function_coefficients_by_time(time), time);
  else
    // (param_rho_disc && param_c_disc)
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination,
                                        param_rho_disc->get_function_coefficients_by_time(time),
                                        param_c_disc->get_function_coefficients_by_time(time));
}

template <int dim>
void WaveEquationBase<dim>::fill_C_intermediate(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx) {
  // fill with (D^{n+1})^{-1} C^n, n = time_idx on the mesh of time_idx
  // -> needs to be able to change the time in ρ to the next time if continuous ρ is used

  matrix_C_intermediate.reinit(*mesh->get_sparsity_pattern(time_idx));
  double current_time = mesh->get_time(time_idx);
  double next_time    = mesh->get_time(time_idx + 1);  // does range checking in debug mode
  auto dof_handler    = mesh->get_dof_handler(time_idx);

  if ((!param_rho_disc && !param_c_disc) || !using_special_assembly(mesh))
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), matrix_C_intermediate, param_rho, param_c,
                                        next_time, current_time);
  else if (param_rho_disc && !param_c_disc)
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), matrix_C_intermediate,
                                        param_rho_disc->get_function_coefficients_by_time(next_time), param_c,
                                        current_time);

  else if (!param_rho_disc && param_c_disc)
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), matrix_C_intermediate, param_rho,
                                        param_c_disc->get_function_coefficients_by_time(current_time), next_time);
  else
    MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), matrix_C_intermediate,
                                        param_rho_disc->get_function_coefficients_by_time(next_time),
                                        param_c_disc->get_function_coefficients_by_time(current_time));
}

template <int dim>
void WaveEquationBase<dim>::fill_D_intermediate(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx) {
  // fill with (D^{n+1})^{-1} D^n, n = time_idx on the mesh of time_idx
  // -> needs to be able to change the time in ρ to the next time if continuous ρ is used

  matrix_D_intermediate.reinit(*mesh->get_sparsity_pattern(time_idx));
  double current_time = mesh->get_time(time_idx);
  double next_time    = mesh->get_time(time_idx + 1);  // does range checking in debug mode
  auto dof_handler    = mesh->get_dof_handler(time_idx);

  if (!param_rho_disc || !using_special_assembly(mesh))
    MatrixCreator<dim>::create_D_intermediate_matrix(dof_handler, mesh->get_quadrature(), matrix_D_intermediate,
                                                     param_rho, current_time, next_time);
  else
    MatrixCreator<dim>::create_D_intermediate_matrix(dof_handler, mesh->get_quadrature(), matrix_D_intermediate,
                                                     param_rho_disc->get_function_coefficients_by_time(current_time),
                                                     param_rho_disc->get_function_coefficients_by_time(next_time));
}

template class WaveEquationBase<1>;
template class WaveEquationBase<2>;
template class WaveEquationBase<3>;

} /* namespace forward */
} /* namespace wavepi */
