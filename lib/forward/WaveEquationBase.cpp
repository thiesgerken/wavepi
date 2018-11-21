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

template<int dim>
void WaveEquationBase<dim>::fill_matrices(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
      SparseMatrix<double> &dst_A, SparseMatrix<double> &dst_B, SparseMatrix<double> &dst_C) {
   // TODO: this assembling could be done more efficient, i.e. loop through the mesh once and assemble all matrices?
   // this helps only a bit because each of the operations is already parallelized
   // tests show about 20%-30% (depending on dim) speedup on my Intel i5 4690

   Threads::TaskGroup<void> task_group;
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_A, *this, mesh, dof_handler, dst_A);
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_B, *this, mesh, dof_handler, dst_B);
   task_group += Threads::new_task(&WaveEquationBase<dim>::fill_C, *this, mesh, dof_handler, dst_C);
   task_group.join_all();
}

// before mesh change, let dst <- (D^n)^{-1} D^{n-1} M^{-1} src
// ( i.e. dst <- src for time-independent D)
template<int dim>
void WaveEquationBase<dim>::vmult_D_intermediate(Vector<double>& dst, const Vector<double>& src) const {
   if (rho_time_dependent) {
      // TODO
      AssertThrow(false, ExcNotImplemented("implement vmult_D_intermediate"));
   } else {
      dst.equ(1.0, src);
   }
}

// before mesh change, let dst <- (D^n)^{-1} C^{n-1} src
// ( i.e. dst <- matrix_C * src for time-independent D)
template<int dim>
void WaveEquationBase<dim>::vmult_C_intermediate(const SparseMatrix<double>& matrix_C, Vector<double>& dst,
      const Vector<double>& src) const {
   if (rho_time_dependent) {
      // TODO
      AssertThrow(false, ExcNotImplemented("implement vmult_C_intermediate"));
   } else {
      matrix_C.vmult(dst, src);
   }
}

template<int dim>
void WaveEquationBase<dim>::fill_A(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
      SparseMatrix<double> &destination) {
   if ((!param_rho_disc && !param_q_disc) || !using_special_assembly(mesh))
      MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho, param_q);
   else if (param_rho_disc && !param_q_disc)
      MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_rho_disc->get_function_coefficients(param_rho_disc->get_time_index()), param_q);
   else if (!param_rho_disc && param_q_disc)
      MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho,
            param_q_disc->get_function_coefficients(param_q_disc->get_time_index()));
   else
      // (param_rho_disc && param_q_disc)
      MatrixCreator<dim>::create_A_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_rho_disc->get_function_coefficients(param_rho_disc->get_time_index()),
            param_q_disc->get_function_coefficients(param_q_disc->get_time_index()));
}

template<int dim>
void WaveEquationBase<dim>::fill_B(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
      SparseMatrix<double> &destination) {
   if (param_nu_disc && using_special_assembly(mesh))
      MatrixCreator<dim>::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_nu_disc->get_function_coefficients(param_nu_disc->get_time_index()));
   else
      dealii::MatrixCreator::create_mass_matrix(dof_handler, mesh->get_quadrature(), destination, param_nu.get());
}

template<int dim>
void WaveEquationBase<dim>::fill_C(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
      SparseMatrix<double> &destination) {
   if ((!param_rho_disc && !param_c_disc) || !using_special_assembly(mesh))
      MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho, param_c);
   else if (param_rho_disc && !param_c_disc)
      MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_rho_disc->get_function_coefficients(param_rho_disc->get_time_index()), param_c);
   else if (!param_rho_disc && param_c_disc)
      MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination, param_rho,
            param_c_disc->get_function_coefficients(param_c_disc->get_time_index()));
   else
      // (param_rho_disc && param_c_disc)
      MatrixCreator<dim>::create_C_matrix(dof_handler, mesh->get_quadrature(), destination,
            param_rho_disc->get_function_coefficients(param_rho_disc->get_time_index()),
            param_c_disc->get_function_coefficients(param_c_disc->get_time_index()));
}

template class WaveEquationBase<1> ;
template class WaveEquationBase<2> ;
template class WaveEquationBase<3> ;

} /* namespace forward */
} /* namespace wavepi */
