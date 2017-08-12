/*
 * SpaceTimeMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_SPACETIMEMESH_H_
#define FORWARD_SPACETIMEMESH_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <cstddef>
#include <memory>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class SpaceTimeMesh {
   public:
      virtual ~SpaceTimeMesh() {
      }

      // The FiniteElement is used for the construction of DoFHandlers.
      // The Quadrature is only used for the mass matrix.
      SpaceTimeMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad);

      // get a mass matrix for the selected time index.
      // If this is not in storage (yet), then the result of get_dof_handler is used to create one
      // Might invalidate the last return value of get_dof_handler!
      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t time_index) = 0;

      // in some cases one does not need a whole DoFHandler, only the number of degrees of freedom.
      // (e.g. in empty vector creation)
      virtual size_t get_n_dofs(size_t time_index) = 0;

      // get a dof_handler for the selected time_index. It might become invalid when this function is called again with a different time_index.
      // has to decide, whether getting a new dof_handler from the initial triangulation and advancing it until time_index is smarter
      // than reusing the current working_dof_handler and moving it.
      virtual std::shared_ptr<DoFHandler<dim> > get_dof_handler(size_t time_index) = 0;

      // get a triangulation for the selected time_index. It might become invalid when this function is called again with a different time_index.
      // has to decide, whether getting a new triangulation from the initial triangulation and advancing it until time_index is smarter
      // than reusing the current triangulation and moving it.
      virtual std::shared_ptr<Triangulation<dim> > get_triangulation(size_t time_index) = 0;

      // essentially, whether get_dof_handler is thread-safe.
      virtual bool allows_parallel_access() const = 0;

      // takes some vectors defined on the mesh of time step source_time_index and interpolates them onto the mesh for target_time_index,
      // changing the given Vectors. Also returns an appropriate DoFHandler for target_time_index (invalidating all other DoFHandlers)
      virtual std::shared_ptr<DoFHandler<dim> > transfer(size_t source_time_index, size_t target_time_index,
            std::initializer_list <Vector<double>*> vectors) = 0;

      //  Determine an estimate for the memory consumption (in bytes) of this object.
      virtual std::size_t memory_consumption() const = 0;

      // time points used in this discretization in ascending order.
      // The first time point should be 0, and the last one should be equal to T.
      inline const std::vector<double>& get_times() const {
         return times;
      }

      inline double get_time(size_t time_index) const {
         return times[time_index];
      }

      // tries to find a given time in the times vector (using a binary search)
      // returns the index of the nearest time, the caller has to decide whether it is good enough.
      // must not be called on a empty discretization!
      size_t find_time(double time) const;
      size_t find_nearest_time(double time) const;
      bool near_enough(double time, size_t idx) const;

     inline const Quadrature<dim>& get_quadrature() const {
         return quad;
      }

   protected:
      std::vector<double> times;
      FE_Q<dim> fe;
      Quadrature<dim> quad;

      size_t find_time(double time, size_t low, size_t up, bool increasing) const;

};

} /* namespace wavepi */
} /* namespace forward */

#endif /* LIB_FORWARD_SPACETIMEMESH_H_ */
