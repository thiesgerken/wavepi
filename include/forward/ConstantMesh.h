/*
 * ConstantMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_CONSTANTMESH_H_
#define FORWARD_CONSTANTMESH_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <forward/SpaceTimeMesh.h>

#include <memory>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class ConstantMesh: public SpaceTimeMesh<dim> {
   public:
      virtual ~ConstantMesh() {
      }

      // The FiniteElement is used for the construction of DoFHandlers.
      // The Quadrature is only used for the mass matrix.
      ConstantMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
            std::shared_ptr<Triangulation<dim>> tria);

      // get a mass matrix for the selected time index.
      // If this is not in storage yet, then the result of get_dof_handler is used to create one
      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(size_t time_index);

      // get a sparsity pattern for the selected time index.
      // If this is not in storage (yet), then the result of get_dof_handler is used to create one
      virtual std::shared_ptr<SparsityPattern> get_sparsity_pattern(size_t time_index);

      // in some cases one does not need a whole DoFHandler, only the number of degrees of freedom.
      // (e.g. in empty vector creation)
      virtual size_t get_n_dofs(size_t time_index);

      // returns the same DoFHandler for all times without changing it
      virtual std::shared_ptr<DoFHandler<dim> > get_dof_handler(size_t time_index);

      // returns the same Triangulation for all times without changing it
      virtual std::shared_ptr<Triangulation<dim> > get_triangulation(size_t time_index);

      // does nothing.
      virtual std::shared_ptr<DoFHandler<dim> > transfer(size_t source_time_index, size_t target_time_index,
            std::initializer_list<Vector<double>*> vectors);

      //  Determine an estimate for the memory consumption (in bytes) of this object.
      virtual std::size_t memory_consumption() const;

      // essentially, whether get_dof_handler is thread-safe.
      virtual bool allows_parallel_access() const {
         return true;
      }

   private:
      using SpaceTimeMesh<dim>::times;
      using SpaceTimeMesh<dim>::fe;
      using SpaceTimeMesh<dim>::quad;

      std::shared_ptr<Triangulation<dim>> triangulation;
      std::shared_ptr<DoFHandler<dim>> dof_handler;

      // don't need it, but its lifetime has to be larger than that of mass_matrix ...
      // therefore, the order here is also important! (mass_matrix is deconstructed first)
      std::shared_ptr<SparsityPattern> sparsity_pattern;

      std::shared_ptr<SparseMatrix<double>> mass_matrix;

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_FORWARD_CONSTANTMESH_H_ */
