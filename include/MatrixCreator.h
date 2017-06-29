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

namespace wavepi {
   namespace MatrixCreator {
      using namespace dealii;

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a and q must be valid function handles.
       */
      template<int dim>
      void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
            SparseMatrix<double> &matrix, const Function<dim> * const a,
            const Function<dim> * const q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a must be a valid function handle and q a discretized function on the same mesh.
       */
      template<int dim>
      void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
            SparseMatrix<double> &matrix, const Function<dim> * const a,
            const Vector<double>& q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * q must be a valid function handle and a a discretized function on the same mesh.
       */
      template<int dim>
      void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
            SparseMatrix<double> &matrix, const Vector<double>& a,
            const Function<dim> * const q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a and q are supplied as discretized FE functions  (living on the same mesh).
       */
      template<int dim>
      void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
            SparseMatrix<double> &matrix, const Vector<double>& a, const Vector<double>& q);

      /**
       * like dealii::MatrixCreator::create_mass_matrix, but with a discretized coefficient c (living on the same mesh)
       * you could just pass this coefficient to dealii::MatrixCreator::create_mass_matrix, but in tests this took 20x longer than
       *  when using the continuous version. This implementation did it in 2x the time.
       */
      template<int dim>
      void create_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
            SparseMatrix<double> &matrix, const Vector<double>& c);

   }
}
#endif /* INCLUDE_MATRIXCREATOR_H_ */
