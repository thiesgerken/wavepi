/*
 * SpaceTimeMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_SPACETIMEMESH_H_
#define FORWARD_SPACETIMEMESH_H_

#include <deal.II/base/types.h>
#include <deal.II/lac/sparse_matrix.h>

#include <stddef.h>
#include <iterator>
#include <memory>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class SpaceTimeMesh {
   public:
      SpaceTimeMesh(std::vector<double> times);

      virtual ~SpaceTimeMesh();


      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(int time_index) = 0;

//      // construct an adaptive mesh where the spatial mesh does not vary in time (yet).
//      // The triangulation is not copied and must not be modified afterwards.
//      // The Quadrature is only used for the mass matrix.
//      AdaptiveMesh(std::vector<double> times, std::shared_ptr<Triangulation<dim>> triangulation, FiniteElement<dim> fe,
//            Quadrature<dim> quad);
//
//      // return the memory used by the mass matrices, the `Triangulation`s and the `DoFHandler`.
//      virtual double estimate_memory_usage();
//
//      // get a mass matrix for the selected time index.
//      // If this is not in storage (yet), then the result of get_dof_handler is used to create one
//      // Might invalidate the last return value of get_dof_handler!
//      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(int time_index);
//
//      // get a dof_handler for the selected time_index. It might become invalid when this function is called again with a different time_index.
//      // has to decide, whether getting a new dof_handler from the initial triangulation and advancing it until time_index is smarter
//      // than reusing the current working_dof_handler and moving it.
//      virtual  std::shared_ptr<DoFHandler<dim> > get_dof_handler(int time_index);
//
//      // takes some vectors defined on the mesh of time step source_time_index and interpolates them onto the mesh for target_time_index,
//      // changing the given Vectors. Also returns an appropriate DoFHandler for target_time_index (invalidating all other DoFHandlers)
//      virtual  std::shared_ptr<DoFHandler<dim> > transfer_to(int source_time_index, int target_time_index,
//            std::vector<std::shared_ptr<Vector<double>>> vectors);

      inline const std::vector<double>& get_times() const {
         return times;
      }

      size_t find_time(double time) const;
      size_t find_nearest_time(double time) const;
      bool near_enough(double time, size_t idx) const;

   protected:
      std::vector<double> times;

      size_t find_time(double time, size_t low, size_t up, bool increasing) const;

};

} /* namespace wavepi */
} /* namespace forward */

#endif /* LIB_FORWARD_SPACETIMEMESH_H_ */
