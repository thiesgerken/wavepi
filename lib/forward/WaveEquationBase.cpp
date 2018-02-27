/*
 * WaveEquationBase.cpp
 *
 *  Created on: 23.07.2017
 *      Author: thies
 */

#include <deal.II/numerics/matrix_tools.h>
#include <forward/MatrixCreator.h>
#include <forward/WaveEquationBase.h>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template <int dim>
WaveEquationBase<dim>::WaveEquationBase(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
    : theta(0.5), mesh(mesh), param_c(one), param_nu(zero), param_a(one), param_q(zero), right_hand_side(zero_rhs) {}

template <int dim>
void WaveEquationBase<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("WaveEquation");
  {
    prm.declare_entry(
        "theta", "0.5", Patterns::Double(0, 1),
        "parameter θ in the time discretization (θ=1 -> backward euler, θ=0 -> forward euler, θ=0.5 -> Crank-Nicolson");
    prm.declare_entry("tol", "1e-8", Patterns::Double(0, 1), "rel. tolerance for the solution of linear systems");
    prm.declare_entry("max iter", "10000", Patterns::Integer(0),
                      "maximum iteration threshold for the solution of linear systems");
  }
  prm.leave_subsection();
}

template <int dim>
void WaveEquationBase<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("WaveEquation");
  {
    theta                  = prm.get_double("theta");
    this->solver_tolerance = prm.get_double("tol");
    this->solver_max_iter  = prm.get_double("max iter");
  }
  prm.leave_subsection();
}

template <int dim>
void WaveEquationBase<dim>::fill_A(DoFHandler<dim> &dof_handler, SparseMatrix<double> &destination) {
  if ((!param_a_disc && !param_q_disc) || !using_special_assembly())
    MatrixCreator<dim>::create_laplace_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_a, param_q);
  else if (param_a_disc && !param_q_disc)
    MatrixCreator<dim>::create_laplace_mass_matrix(
        dof_handler, mesh->get_quadrature(), destination,
        param_a_disc->get_function_coefficients(param_a_disc->get_time_index()), param_q);
  else if (!param_a_disc && param_q_disc)
    MatrixCreator<dim>::create_laplace_mass_matrix(
        dof_handler, mesh->get_quadrature(), destination, param_a,
        param_q_disc->get_function_coefficients(param_q_disc->get_time_index()));
  else
    // (param_a_disc && param_q_disc)
    MatrixCreator<dim>::create_laplace_mass_matrix(
        dof_handler, mesh->get_quadrature(), destination,
        param_a_disc->get_function_coefficients(param_a_disc->get_time_index()),
        param_q_disc->get_function_coefficients(param_q_disc->get_time_index()));
}

template <int dim>
void WaveEquationBase<dim>::fill_B(DoFHandler<dim> &dof_handler, SparseMatrix<double> &destination) {
  if (param_nu_disc && using_special_assembly())
    MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
                                           param_nu_disc->get_function_coefficients(param_nu_disc->get_time_index()));
  else
    dealii::MatrixCreator::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_nu.get());
}

template <int dim>
void WaveEquationBase<dim>::fill_C(DoFHandler<dim> &dof_handler, SparseMatrix<double> &destination) {
  if (param_c_disc && using_special_assembly())
    MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
                                           param_c_disc->get_function_coefficients(param_c_disc->get_time_index()));
  else
    dealii::MatrixCreator::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_c.get());
}

template class WaveEquationBase<1>;
template class WaveEquationBase<2>;
template class WaveEquationBase<3>;

} /* namespace forward */
} /* namespace wavepi */
