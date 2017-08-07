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

template<int dim>
WaveEquationBase<dim>::WaveEquationBase(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler, const Quadrature<dim> quad)
      : theta(0.5), mesh(mesh), dof_handler(dof_handler), quad(quad), param_c(one), param_nu(zero), param_a(
            one), param_q(zero), right_hand_side(zero_rhs) {
}

template<int dim>
WaveEquationBase<dim>::~WaveEquationBase() {
}

template<int dim>
void WaveEquationBase<dim>::fill_A(SparseMatrix<double>& destination) {
   if ((!param_a_disc && !param_q_disc) || !using_special_assembly())
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, destination, param_a, param_q);
   else if (param_a_disc && !param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, destination,
            param_a_disc->get_function_coefficients()[param_a_disc->get_time_index()], param_q);
   else if (!param_a_disc && param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, destination, param_a,
            param_q_disc->get_function_coefficients()[param_q_disc->get_time_index()]);
   else
      // (param_a_disc && param_q_disc)
      MatrixCreator::create_laplace_mass_matrix(*dof_handler, quad, destination,
            param_a_disc->get_function_coefficients()[param_a_disc->get_time_index()],
            param_q_disc->get_function_coefficients()[param_q_disc->get_time_index()]);

}

template<int dim>
void WaveEquationBase<dim>::fill_B(SparseMatrix<double>& destination) {
   if (param_nu_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, destination,
            param_nu_disc->get_function_coefficients()[param_nu_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, destination, param_nu.get());
}

template<int dim>
void WaveEquationBase<dim>::fill_C(SparseMatrix<double>& destination) {
   if (param_c_disc && using_special_assembly())
      MatrixCreator::create_mass_matrix(*dof_handler, quad, destination,
            param_c_disc->get_function_coefficients()[param_c_disc->get_time_index()]);
   else
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, destination, param_c.get());
}

template class WaveEquationBase<1> ;
template class WaveEquationBase<2> ;
template class WaveEquationBase<3> ;

} /* namespace forward */
} /* namespace wavepi */