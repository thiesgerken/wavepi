/*
 * ConstantMesh.cpp
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#include <deal.II/base/memory_consumption.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>

#include <base/ConstantMesh.h>

namespace wavepi {
namespace base {
using namespace dealii;

template <int dim>
ConstantMesh<dim>::ConstantMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
                                std::shared_ptr<Triangulation<dim>> tria)
    : SpaceTimeMesh<dim>(times, fe, quad), triangulation(tria) {
  dof_handler = std::make_shared<DoFHandler<dim>>();
  dof_handler->initialize(*tria, this->fe);
}

template <int dim>
std::shared_ptr<SparseMatrix<double>> ConstantMesh<dim>::get_mass_matrix(size_t time_index) {
  if (!mass_matrix) {
    get_sparsity_pattern(time_index);

    mass_matrix = std::make_shared<SparseMatrix<double>>(*sparsity_pattern);
    dealii::MatrixCreator::create_mass_matrix(*dof_handler, quad, *mass_matrix);
  }

  return mass_matrix;
}

template <int dim>
std::shared_ptr<SparseMatrix<double>> ConstantMesh<dim>::get_laplace_matrix(size_t time_index) {
  if (!laplace_matrix) {
    get_sparsity_pattern(time_index);

    laplace_matrix = std::make_shared<SparseMatrix<double>>(*sparsity_pattern);
    dealii::MatrixCreator::create_laplace_matrix(*dof_handler, quad, *laplace_matrix);
  }

  return laplace_matrix;
}

template <int dim>
std::shared_ptr<ConstraintMatrix> ConstantMesh<dim>::get_constraint_matrix(size_t idx __attribute((unused))) {
  if (!constraints) {
    constraints = std::make_shared<ConstraintMatrix>();
    DoFTools::make_hanging_node_constraints(*dof_handler, *constraints);
    constraints->close();
  }

  return constraints;
}

template <int dim>
std::shared_ptr<SparsityPattern> ConstantMesh<dim>::get_sparsity_pattern(size_t time_index) {
  if (!sparsity_pattern) {
    get_constraint_matrix(time_index);

    DynamicSparsityPattern dsp(dof_handler->n_dofs(), dof_handler->n_dofs());
    DoFTools::make_sparsity_pattern(*dof_handler, dsp, *constraints, true);

    sparsity_pattern = std::make_shared<SparsityPattern>();
    sparsity_pattern->copy_from(dsp);
  }

  return sparsity_pattern;
}

template <int dim>
size_t ConstantMesh<dim>::n_dofs(size_t time_index __attribute((unused))) {
  return dof_handler->n_dofs();
}

template <int dim>
std::shared_ptr<DoFHandler<dim>> ConstantMesh<dim>::get_dof_handler(size_t time_index __attribute((unused))) {
  return dof_handler;
}

template <int dim>
std::shared_ptr<Triangulation<dim>> ConstantMesh<dim>::get_triangulation(size_t time_index __attribute((unused))) {
  return triangulation;
}

template <int dim>
std::shared_ptr<DoFHandler<dim>> ConstantMesh<dim>::transfer(size_t source_time_index __attribute((unused)),
                                                             size_t target_time_index __attribute((unused)),
                                                             std::initializer_list<Vector<double>*> vectors
                                                             __attribute((unused))) {
  // LogStream::Prefix p("ConstantMesh");
  // deallog << "Mesh transfer: " << source_time_index << " → " << target_time_index << ", taking "
  //         << vectors.size() << " vector(s) along" << std::endl;

  return dof_handler;
}

template <int dim>
size_t ConstantMesh<dim>::memory_consumption() const {
  size_t mem =
      MemoryConsumption::memory_consumption(*dof_handler) + MemoryConsumption::memory_consumption(dof_handler) +
      MemoryConsumption::memory_consumption(*triangulation) + MemoryConsumption::memory_consumption(triangulation) +
      MemoryConsumption::memory_consumption(times) + MemoryConsumption::memory_consumption(sparsity_pattern) +
      MemoryConsumption::memory_consumption(mass_matrix);

  if (mass_matrix) mem += MemoryConsumption::memory_consumption(*mass_matrix);

  if (sparsity_pattern) mem += MemoryConsumption::memory_consumption(*sparsity_pattern);

  return mem;
}

template class ConstantMesh<1>;
template class ConstantMesh<2>;
template class ConstantMesh<3>;

}  // namespace base
} /* namespace wavepi */
