/*
 * SpaceTimeMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_SPACETIMEMESH_H_
#define FORWARD_SPACETIMEMESH_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <vector>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * Mesh in Space and Time. Holds a list of time steps and virtual functions how to get spatial meshes for each time.
 */
template <int dim>
class SpaceTimeMesh {
 public:
  virtual ~SpaceTimeMesh() = default;

  /**
   * @param times nodal values in time in ascending order.
   * @param fe `FiniteElement` that should be used for the construction of `DoFHandlers`.
   * @param quad `Quadrature` that should be used for mass matrices, is also exposed through `get_quadrature()`.
   */
  SpaceTimeMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad);

  /**
   * @return `Quadrature` that was passed to the constructor.
   */
  const Quadrature<dim>& get_quadrature() const { return quad; }

  /**
   * @return an estimate for the memory consumption (in bytes) of this object.
   */
  virtual std::size_t memory_consumption() const = 0;

  /**
   *  In some cases one does not need a whole DoFHandler, only the number of degrees of freedom.
   * (e.g. in empty vector creation)
   */
  virtual size_t n_dofs(size_t idx) = 0;

  /**
   * get a mass matrix for the selected time index.
   * If this is not in storage (yet), then the result of get_dof_handler is used to create one
   * Might invalidate the last return value of get_dof_handler!
   */
  virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t idx) = 0;

  /**
   * get a laplace matrix for the selected time index.
   * If this is not in storage (yet), then the result of get_dof_handler is used to create one
   * Might invalidate the last return value of get_dof_handler!
   */
  virtual std::shared_ptr<SparseMatrix<double>> get_laplace_matrix(size_t idx) = 0;

  /**
   * get the sparsity pattern for the selected time index.
   * If this is not in storage (yet), then the result of get_dof_handler is used to create one
   * Might invalidate the last return value of get_dof_handler!
   */
  virtual std::shared_ptr<SparsityPattern> get_sparsity_pattern(size_t idx) = 0;

  /**
   * get a `DofHandler` for the selected time_index. It might become invalid when this function is called again with a
   * different time_index. has to decide, whether getting a new dof_handler from the initial triangulation and advancing
   * it until time_index is smarter than reusing the current working_dof_handler and moving it.
   */
  virtual std::shared_ptr<DoFHandler<dim>> get_dof_handler(size_t idx) = 0;

  /**
   * get a `AffineConstraints<double>` (for hanging nodes) for the selected time_index. It might become invalid when this
   * function is called again with a different time_index. has to decide, whether getting a new dof_handler from the
   * initial triangulation and advancing it until time_index is smarter than reusing the current working_dof_handler and
   * moving it.
   */
  virtual std::shared_ptr<AffineConstraints<double>> get_constraint_matrix(size_t idx) = 0;

  /**
   * get a triangulation for the selected time_index. It might become invalid when this function is called again with a
   * different time_index. has to decide, whether getting a new triangulation from the initial triangulation and
   * advancing it until time_index is smarter than reusing the current triangulation and moving it.
   */
  virtual std::shared_ptr<Triangulation<dim>> get_triangulation(size_t idx) = 0;

  /**
   * takes some vectors defined on the mesh of time step source_time_index and interpolates them onto the mesh for
   * target_time_index, changing the given Vectors. Also returns an appropriate DoFHandler for target_time_index
   * (invalidating all other DoFHandlers)
   */
  virtual std::shared_ptr<DoFHandler<dim>> transfer(size_t source_time_index, size_t target_time_index,
                                                    std::initializer_list<Vector<double>*> vectors) = 0;

  /**
   * @return time points used in this discretization (in ascending order).
   */
  const std::vector<double>& get_times() const { return times; }

  /**
   * Shortcut for `get_times()[idx]` (with range checking in debug mode).
   *
   * @return `get_times()[idx]`
   */
  double get_time(size_t idx) const {
    Assert(0 <= idx && idx < times.size(), ExcIndexRange(idx, 0, times.size()));

    return times[idx];
  }

  /**
   * shortcut for the number of time steps.
   *
   * @return `get_times().size()`
   */
  size_t length() const { return times.size(); }

  /**
   * Tries to find a given time in the times vector using a binary search.
   * Must not be called on a empty discretization.
   *
   * @param time the searched for time.
   * @returns the index of the nearest time, the caller has to decide whether it is good enough.
   */
  size_t nearest_time(double time) const;

  /**
   * Tries to find a given time in the times vector using a binary search.
   * Must not be called on a empty discretization.
   *
   * @param time the searched for time.
   * @returns the index of the time point
   * @throws Exc
   */
  size_t find_time(double time) const;

 protected:
  std::vector<double> times;
  FE_Q<dim> fe;
  Quadrature<dim> quad;

 private:
  size_t nearest_time(double time, size_t low, size_t up) const;
};

}  // namespace base
}  // namespace wavepi

#endif /* LIB_FORWARD_SPACETIMEMESH_H_ */
