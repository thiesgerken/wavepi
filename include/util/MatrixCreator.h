/*
 * MatrixCreator.h
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#ifndef UTIL_MATRIXCREATOR_H_
#define UTIL_MATRIXCREATOR_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>

#include <memory>
#include <vector>

namespace wavepi {
namespace util {

using namespace dealii;

template<int dim>
class MatrixCreator {
   public:
      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a and q must be valid function handles.
       */
      static void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
            std::shared_ptr<Function<dim>> a, std::shared_ptr<Function<dim>> q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a must be a valid function handle and q a discretized function on the same mesh.
       */
      static void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
            std::shared_ptr<Function<dim>> a, const Vector<double>& q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * q must be a valid function handle and a a discretized function on the same mesh.
       */
      static void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
            const Vector<double>& a, std::shared_ptr<Function<dim>> q);

      /**
       * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well.
       * a and q are supplied as discretized FE functions  (living on the same mesh).
       */
      static void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
            const Vector<double>& a, const Vector<double>& q);

      /**
       * like dealii::MatrixCreator::create_mass_matrix, but with a discretized coefficient c (living on the same mesh)
       * you could just pass this coefficient to dealii::MatrixCreator::create_mass_matrix, but in tests this took 20x longer than
       *  when using the continuous version. This implementation did it in 2x the time.
       */
      static void create_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
            const Vector<double>& c);

   private:

      struct LaplaceAssemblyScratchData {
            LaplaceAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            LaplaceAssemblyScratchData(const LaplaceAssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      struct MassAssemblyScratchData {
            MassAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      struct AssemblyCopyData {
            FullMatrix<double> cell_matrix;
            std::vector<types::global_dof_index> local_dof_indices;
      };

      static void copy_local_to_global(SparseMatrix<double> &matrix, const AssemblyCopyData &copy_data);

      static void local_assemble_mass(const Vector<double> &c, const typename DoFHandler<dim>::active_cell_iterator &cell,
            MassAssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);

      static void local_assemble_laplace_mass_dc(const Vector<double> &a, const Function<dim> * const q,
            const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);

      static void local_assemble_laplace_mass_cd(const Function<dim> * const a, const Vector<double> &q,
            const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);

      static void local_assemble_laplace_mass_dd(const Vector<double> &a, const Vector<double> &q,
            const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);

      static void local_assemble_laplace_mass_cc(const Function<dim> * const a, const Function<dim> * const q,
            const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);

};
} /* namespace util */
} /* namespace wavepi */

#endif /* UTIL_MATRIXCREATOR_H_ */
