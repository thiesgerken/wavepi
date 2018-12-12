/*
 * ConstantMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_CONSTANTMESH_H_
#define FORWARD_CONSTANTMESH_H_

#include <base/SpaceTimeMesh.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <memory>
#include <vector>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * Mesh that is constant in time and space.
 */
template <int dim>
class ConstantMesh : public SpaceTimeMesh<dim> {
 public:
  virtual ~ConstantMesh() = default;

  ConstantMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad, std::shared_ptr<Triangulation<dim>> tria);

  virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t idx) override;

  virtual std::shared_ptr<SparseMatrix<double>> get_laplace_matrix(size_t idx) override;

  virtual std::shared_ptr<SparsityPattern> get_sparsity_pattern(size_t idx) override;

  virtual size_t n_dofs(size_t idx);

  virtual std::shared_ptr<DoFHandler<dim>> get_dof_handler(size_t idx) override;

  virtual std::shared_ptr<AffineConstraints<double>> get_constraint_matrix(size_t idx) override;

  virtual std::shared_ptr<Triangulation<dim>> get_triangulation(size_t idx) override;

  virtual std::shared_ptr<DoFHandler<dim>> transfer(size_t source_time_index, size_t target_time_index,
                                                    std::initializer_list<Vector<double>*> vectors) override;

  virtual std::size_t memory_consumption() const override;

 private:
  using SpaceTimeMesh<dim>::times;
  using SpaceTimeMesh<dim>::fe;
  using SpaceTimeMesh<dim>::quad;

  std::shared_ptr<Triangulation<dim>> triangulation;
  std::shared_ptr<DoFHandler<dim>> dof_handler;

  // the order here is important! (matrices have to be deconstructed first)
  std::shared_ptr<SparsityPattern> sparsity_pattern;
  std::shared_ptr<SparseMatrix<double>> mass_matrix;
  std::shared_ptr<SparseMatrix<double>> laplace_matrix;

  std::shared_ptr<AffineConstraints<double>> constraints;
};

}  // namespace base
} /* namespace wavepi */

#endif /* LIB_FORWARD_CONSTANTMESH_H_ */
