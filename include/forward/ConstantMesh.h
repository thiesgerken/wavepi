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
      ConstantMesh(std::vector<double> times, std::shared_ptr<DoFHandler<dim>> dof_handler,
            Quadrature<dim> quad);
      virtual ~ConstantMesh();

      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(int time_index);

      inline std::shared_ptr<DoFHandler<dim> > get_dof_handler() const {
         return dof_handler;
      }

   private:
      // don't need it, but its lifetime has to be larger than that of mass_matrix ...
      // therefore, the order here is also important! (mass_matrix is deconstructed first)
      SparsityPattern sparsity_pattern;

      std::shared_ptr<DoFHandler<dim>> dof_handler;
      std::shared_ptr<SparseMatrix<double>> mass_matrix;

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_FORWARD_CONSTANTMESH_H_ */
