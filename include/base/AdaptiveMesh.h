/*
 * AdaptiveMesh.h
 *
 *  Created on: 09.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_BASE_ADAPTIVEMESH_H_
#define INCLUDE_BASE_ADAPTIVEMESH_H_

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <stddef.h>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * Patch = list of consecutive grid refinements (first item in the pair) and coarsenings (second item)
 */
typedef std::vector<std::pair<std::vector<bool>, std::vector<bool>>> Patch;

/**
 * Mesh that is adaptive in time and space
 */
template <int dim>
class AdaptiveMesh : public SpaceTimeMesh<dim> {
 public:
  virtual ~AdaptiveMesh() = default;

  AdaptiveMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad, std::shared_ptr<Triangulation<dim>> tria);

  virtual size_t n_dofs(size_t idx);

  virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t idx) override;

  virtual std::shared_ptr<SparseMatrix<double>> get_laplace_matrix(size_t idx) override;

  virtual std::shared_ptr<SparsityPattern> get_sparsity_pattern(size_t idx) override;

  virtual std::shared_ptr<AffineConstraints<double>> get_constraint_matrix(size_t idx) override;

  virtual std::shared_ptr<DoFHandler<dim>> get_dof_handler(size_t idx) override;

  virtual std::shared_ptr<Triangulation<dim>> get_triangulation(size_t idx) override;

  virtual std::shared_ptr<DoFHandler<dim>> transfer(size_t source_time_index, size_t target_time_index,
                                                    std::initializer_list<Vector<double> *> vectors) override;

  virtual size_t memory_consumption() const override;

  /**
   * refine / coarsen this Adaptive Mesh.
   *
   * @param refine_intervals list of time intervals that should be refined
   * @param coarsen_time_points list of time indices that should be deleted
   * @param refine_trias for each spatial mesh in the old AdaptiveMesh, a list of cells that should be refined
   * @param coarsen_trias for each spatial mesh in the old AdaptiveMesh, a list of cells that should be coarsened
   * @param interpolate_vectors vectors that you want to take with you to the new mesh (other ones will have invalid
   * content after this action)
   */
  void refine_and_coarsen(std::vector<size_t> refine_intervals, std::vector<size_t> coarsen_time_points,
                          std::vector<std::vector<bool>> refine_trias, std::vector<std::vector<bool>> coarsen_trias,
                          std::initializer_list<DiscretizedFunction<dim> *> interpolate_vectors);

  /**
   * Getter for how the mesh is advanced from one time step to the next.
   */
  const std::vector<Patch> &get_forward_patches() const;

  /**
   * Setter for how the mesh is advanced from one time step to the next.
   */
  void set_forward_patches(const std::vector<Patch> &forward_patches);

 private:
  using SpaceTimeMesh<dim>::quad;
  using SpaceTimeMesh<dim>::times;
  using SpaceTimeMesh<dim>::fe;

  /**
   * triangulation for time step zero
   */
  std::shared_ptr<Triangulation<dim>> initial_triangulation;

  /**
   * how to advance the spatial mesh of one time step to the next one
   * i-th entry: patch to go from step i to step i+1
   */
  std::vector<Patch> forward_patches;

  /**
   * how to advance the spatial mesh of one time step to the previous one
   * i-th entry: patch to go from step i+1 to step i
   */
  std::vector<Patch> backward_patches;

  /**
   * n_dofs for each time
   */
  std::vector<size_t> vector_sizes;

  /**
   * all the sparsity patterns (or empty `std::shared_ptr` if they have not been constructed yet)
   *
   * Note: the lifetime of this object has to be longer than that of mass_matrices,
   * therefore the order of declaration is important! (mass_matrices is deconstructed first)
   */
  std::vector<std::shared_ptr<SparsityPattern>> sparsity_patterns;

  /**
   * all the mass matrices (or empty `std::shared_ptr` if they have not been constructed yet)
   */
  std::vector<std::shared_ptr<SparseMatrix<double>>> mass_matrices;

  /**
   * all the laplace matrices (or empty `std::shared_ptr` if they have not been constructed yet)
   */
  std::vector<std::shared_ptr<SparseMatrix<double>>> laplace_matrices;

  /**
   * all the constraint matrices (or empty `std::shared_ptr` if they have not been constructed yet)
   */
  std::vector<std::shared_ptr<AffineConstraints<double>>> constraints;

  // the Triangulation and DoFHandler for the last requested mass matrix or DoFHandler.
  size_t working_time_idx;
  std::shared_ptr<Triangulation<dim>> working_triangulation;
  std::shared_ptr<DoFHandler<dim>> working_dof_handler;

  /**
   * Deletes internal caches (mass matrices, ...), necessary after a mesh change.
   */
  void reset();

  /**
   * `transfer` with `source_time_index` set to `working_time_idx`.
   */
  void transfer(size_t target_time_index, std::vector<Vector<double> *> &vectors);

  /**
   * `transfer` with `source_time_index` set to `working_time_idx` and no interpolation.
   */
  void transfer(size_t target_time_index);

  /**
   * Applies a given patch.
   *
   * @param vectors interpolate those vectors to the patched mesh
   */
  void patch(const Patch &p, std::vector<Vector<double> *> &vectors);

  /**
   * Use `forward_patches` to fill/recreate the vector `backward_patches`.
   * `working_time_idx` has to be zero upon calling this function, and will be `this->length-1` afterwards.
   */
  void generate_backward_patches();

  /**
   * Utility function, checks whether any entry of the given vector is `true`.
   */
  bool any(const std::vector<bool> &v) const;
};

}  // namespace base
} /* namespace wavepi */
#endif /* INCLUDE_BASE_ADAPTIVEMESH_H_ */
