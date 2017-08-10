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
#include <deal.II/base/memory_consumption.h>

#include <forward/ConstantMesh.h>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
ConstantMesh<dim>::ConstantMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
      std::shared_ptr<Triangulation<dim>> tria)
      : SpaceTimeMesh<dim>(times, fe, quad), triangulation(tria) {
   dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(*tria, fe);
}

template<int dim>
std::shared_ptr<SparseMatrix<double>> ConstantMesh<dim>::get_mass_matrix(
      size_t time_index __attribute((unused))) {
   if (!mass_matrix) {
      DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(*dof_handler, dsp);
      sparsity_pattern.copy_from(dsp);

      mass_matrix = std::make_shared<SparseMatrix<double>>(sparsity_pattern);
      dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, *mass_matrix);
   }

   return mass_matrix;
}

template<int dim> size_t ConstantMesh<dim>::get_n_dofs(size_t time_index __attribute((unused))) {
   return dof_handler->n_dofs();
}

template<int dim> std::shared_ptr<DoFHandler<dim> > ConstantMesh<dim>::get_dof_handler(
      size_t time_index __attribute((unused))) {
   return dof_handler;
}

template<int dim> std::shared_ptr<Triangulation<dim> > ConstantMesh<dim>::get_triangulation(
      size_t time_index __attribute((unused))) {
   return triangulation;
}

template<int dim> std::shared_ptr<DoFHandler<dim> > ConstantMesh<dim>::transfer_to(
      size_t source_time_index __attribute((unused)), size_t target_time_index __attribute((unused)),
      std::vector<std::shared_ptr<Vector<double>>> vectors __attribute((unused))) {
   return dof_handler;
}

template<int dim> size_t ConstantMesh<dim>::memory_consumption() const {
   size_t mem = MemoryConsumption::memory_consumption(dof_handler)
         + MemoryConsumption::memory_consumption(*triangulation) + MemoryConsumption::memory_consumption(triangulation)
         + MemoryConsumption::memory_consumption(times)
         + MemoryConsumption::memory_consumption(sparsity_pattern)
         + MemoryConsumption::memory_consumption(mass_matrix);

   if (mass_matrix)
      mem += MemoryConsumption::memory_consumption(*mass_matrix);

   return mem;
}

template class ConstantMesh<1> ;
template class ConstantMesh<2> ;
template class ConstantMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */
