/*
 * AdaptiveMesh.h
 *
 *  Created on: 09.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_FORWARD_ADAPTIVEMESH_H_
#define INCLUDE_FORWARD_ADAPTIVEMESH_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>

#include <stddef.h>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

// Patch = list of consecutive grid refinements (first item in the pair) and coarsenings (second item)
typedef std::vector<std::pair<std::vector<bool>, std::vector<bool>>> Patch;

template<int dim>
class AdaptiveMesh: public SpaceTimeMesh<dim> {
   public:

      /**
       * Default destructor.
       */
      virtual ~AdaptiveMesh() = default;

      // The FiniteElement is used for the construction of DoFHandlers.
      // The Quadrature is only used for the mass matrix.
      AdaptiveMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
            std::shared_ptr<Triangulation<dim>> tria);

      // in some cases one does not need a whole DoFHandler, only the number of degrees of freedom.
      // (e.g. in empty vector creation)
      virtual size_t n_dofs(size_t idx);

      // get a mass matrix for the selected time index.
      // If this is not in storage (yet), then the result of get_dof_handler is used to create one
      // Might invalidate the last return value of get_dof_handler!
      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t idx);

      // If this is not in storage (yet), then the result of get_dof_handler is used to create one
      // Might invalidate the last return value of get_dof_handler!
      virtual std::shared_ptr<SparsityPattern> get_sparsity_pattern(size_t idx);

      // get a ConstraintMatrix (for hanging nodes) for the selected time_index. It might become invalid when this function is called again with a different time_index.
      // has to decide, whether getting a new dof_handler from the initial triangulation and advancing it until time_index is smarter
      // than reusing the current working_dof_handler and moving it.
      virtual std::shared_ptr<ConstraintMatrix> get_constraint_matrix(size_t idx);

      // get a dof_handler for the selected time_index. It might become invalid when this function is called again with a different time_index.
      // has to decide, whether getting a new dof_handler from the initial triangulation and advancing it until time_index is smarter
      // than reusing the current working_dof_handler and moving it.
      virtual std::shared_ptr<DoFHandler<dim> > get_dof_handler(size_t idx);

      // get a triangulation for the selected time_index. It might become invalid when this function is called again with a different time_index.
      // has to decide, whether getting a new triangulation from the initial triangulation and advancing it until time_index is smarter
      // than reusing the current triangulation and moving it.
      virtual std::shared_ptr<Triangulation<dim> > get_triangulation(size_t idx);

      // takes some vectors defined on the mesh of time step source_time_index and interpolates them onto the mesh for target_time_index,
      // changing the given Vectors. Also returns an appropriate DoFHandler for target_time_index (invalidating all other DoFHandlers)
      virtual std::shared_ptr<DoFHandler<dim> > transfer(size_t source_time_index, size_t target_time_index,
            std::initializer_list<Vector<double>*> vectors);

      virtual size_t memory_consumption() const;

      virtual bool allows_parallel_access() const {
         return false;
      }

      // refine / coarsen this Adaptive Mesh.
      // `refine_intervals`: list of time intervals that should be refined
      // `coarsen_time_points`: list of time indices that should be deleted
      // `refine_trias`: for each spatial mesh in the old AdaptiveMesh, a list of cells that should be refined
      // `coarsen_trias`: for each spatial mesh in the old AdaptiveMesh, a list of cells that should be coarsened
      // `interpolate_vectors`: vectors that you want to take with you to the new mesh (other ones will have invalid content after this action)
      void refine_and_coarsen(std::vector<size_t> refine_intervals, std::vector<size_t> coarsen_time_points,
            std::vector<std::vector<bool>> refine_trias, std::vector<std::vector<bool>> coarsen_trias,
            std::initializer_list<DiscretizedFunction<dim>*> interpolate_vectors);

      const std::vector<Patch>& get_forward_patches() const;

      void set_forward_patches(const std::vector<Patch>& forward_patches);

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
       * all the constraint matrices (or empty `std::shared_ptr` if they have not been constructed yet)
       */
      std::vector<std::shared_ptr<ConstraintMatrix>> constraints;

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
      void transfer(size_t target_time_index, std::vector<Vector<double>*> &vectors);

      /**
       * `transfer` with `source_time_index` set to `working_time_idx` and no interpolation.
       */
      void transfer(size_t target_time_index);

      /**
       * Applies a given patch.
       *
       * @param vectors interpolate those vectors to the patched mesh
       */
      void patch(const Patch &p, std::vector<Vector<double>*> &vectors);

      /**
       * Use `forward_patches` to fill/recreate the vector `backward_patches`.
       * `working_time_idx` has to be zero upon calling this function, and will be `this->length-1` afterwards.
       */
      void generate_backward_patches();
};

} /* namespace forward */
} /* namespace wavepi */
#endif /* INCLUDE_FORWARD_ADAPTIVEMESH_H_ */
