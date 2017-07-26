/*
 * ConstantMesh.cpp
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>

#include <forward/ConstantMesh.h>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
ConstantMesh<dim>::ConstantMesh(std::vector<double> times, std::shared_ptr<DoFHandler<dim>> dof_handler,
      Quadrature<dim> quad)
      : SpaceTimeMesh<dim>(times), dof_handler(dof_handler) {

   DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
   DoFTools::make_sparsity_pattern(*dof_handler, dsp);
   sparsity_pattern.copy_from(dsp);

   mass_matrix = std::make_shared<SparseMatrix<double>>(sparsity_pattern);
   dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, *mass_matrix);
}

template<int dim>
ConstantMesh<dim>::~ConstantMesh() {
}

template<int dim>
std::shared_ptr<SparseMatrix<double>> ConstantMesh<dim>::get_mass_matrix(
      int time_index __attribute((unused))) {
   return mass_matrix;
}

template class ConstantMesh<1> ;
template class ConstantMesh<2> ;
template class ConstantMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */
