/*
 * MatrixCreator.h
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#ifndef INCLUDE_MATRIXCREATOR_H_
#define INCLUDE_MATRIXCREATOR_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>

namespace dealii {
  namespace MatrixCreator {
     using namespace dealii;

     /**
      * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
      * a and q must be valid function handles.
      */
     template <int dim, int spacedim, typename number>
     void create_laplace_mass_matrix (const DoFHandler<dim,spacedim>    &dof,
                              const Quadrature<dim>    &quad,
                              SparseMatrix<number>     &matrix,
                              const Function<spacedim,number> *const a,
                              const Function<spacedim,number> *const q);



  }
}
#endif /* INCLUDE_MATRIXCREATOR_H_ */
