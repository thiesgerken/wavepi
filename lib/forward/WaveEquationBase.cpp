/*
 * WaveEquationBase.cpp
 *
 *  Created on: 23.07.2017
 *      Author: thies
 */

#include <deal.II/numerics/matrix_tools.h>
#include <forward/WaveEquationBase.h>
#include <util/MatrixCreator.h>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::util;

template<int dim>
WaveEquationBase<dim>::WaveEquationBase(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : theta(0.5), mesh(mesh), param_c(one), param_nu(zero), param_a(
            one), param_q(zero), right_hand_side(zero_rhs) {
}

template<int dim>
WaveEquationBase<dim>::~WaveEquationBase() {
}

template<int dim>
void WaveEquationBase<dim>::fill_A(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination) {
   if ((!param_a_disc && !param_q_disc) || !using_special_assembly())
      MatrixCreator<dim>::create_laplace_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_a, param_q);
   else if (param_a_disc && !param_q_disc)
      MatrixCreator<dim>::create_laplace_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_a_disc->get_function_coefficient(param_a_disc->get_time_index()), param_q);
   else if (!param_a_disc && param_q_disc)
      MatrixCreator<dim>::create_laplace_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_a,
            param_q_disc->get_function_coefficient(param_q_disc->get_time_index()));
   else
      // (param_a_disc && param_q_disc)
      MatrixCreator<dim>::create_laplace_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_a_disc->get_function_coefficient(param_a_disc->get_time_index()),
            param_q_disc->get_function_coefficient(param_q_disc->get_time_index()));

}

template<int dim>
void WaveEquationBase<dim>::fill_B(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination) {
   if (param_nu_disc && using_special_assembly())
      MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_nu_disc->get_function_coefficient(param_nu_disc->get_time_index()));
   else
      dealii::MatrixCreator::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_nu.get());
}

template<int dim>
void WaveEquationBase<dim>::fill_C(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination) {
   if (param_c_disc && using_special_assembly())
      MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_c_disc->get_function_coefficient(param_c_disc->get_time_index()));
   else
      dealii::MatrixCreator::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_c.get());
}

template class WaveEquationBase<1> ;
template class WaveEquationBase<2> ;
template class WaveEquationBase<3> ;

} /* namespace forward */
} /* namespace wavepi */
