/*
 * MatrixCreator.cc
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#include "MatrixCreator.h"

namespace dealii {
   namespace MatrixCreator {
      using namespace dealii;

      template<int dim>
      struct AssemblyScratchData {
            AssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            AssemblyScratchData(const AssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      template struct AssemblyScratchData<1> ;
      template struct AssemblyScratchData<2> ;
      template struct AssemblyScratchData<3> ;

      template<typename number>
      struct AssemblyCopyData {
            FullMatrix<number> cell_matrix;
            std::vector<types::global_dof_index> local_dof_indices;
      };

      template struct AssemblyCopyData<double> ;

      template<int dim>
      AssemblyScratchData<dim>::AssemblyScratchData(const FiniteElement<dim> &fe,
            const Quadrature<dim> &quad)
            : fe_values(fe, quad,
                  update_values | update_gradients | update_quadrature_points | update_JxW_values) {
      }

      template<int dim>
      AssemblyScratchData<dim>::AssemblyScratchData(const AssemblyScratchData &scratch_data)
            : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                  update_values | update_gradients | update_quadrature_points | update_JxW_values) {
      }

      template<int dim, int spacedim, typename number>
      void local_assemble_system(const Function<spacedim, number> * const a,
            const Function<spacedim, number> * const q,
            const typename DoFHandler<dim>::active_cell_iterator &cell,
            AssemblyScratchData<dim> &scratch_data, AssemblyCopyData<number> &copy_data) {
         const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
         const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

         copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
         copy_data.local_dof_indices.resize(dofs_per_cell);
         scratch_data.fe_values.reinit(cell);

         for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
            const double val_a = a->value(scratch_data.fe_values.quadrature_point(q_point));
            const double val_q = q->value(scratch_data.fe_values.quadrature_point(q_point));

            for (unsigned int i = 0; i < dofs_per_cell; ++i) {
               for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  copy_data.cell_matrix(i, j) += (val_a
                        * scratch_data.fe_values.shape_grad(i, q_point)
                        * scratch_data.fe_values.shape_grad(j, q_point)
                        + val_q * scratch_data.fe_values.shape_value(i, q_point)
                              * scratch_data.fe_values.shape_value(j, q_point))
                        * scratch_data.fe_values.JxW(q_point);
            }
         }

         cell->get_dof_indices(copy_data.local_dof_indices);
      }

      template void local_assemble_system(const Function<1, double> * const,
            const Function<1, double> * const, const typename DoFHandler<1>::active_cell_iterator &,
            AssemblyScratchData<1> &scratch_data, AssemblyCopyData<double> &);

      template void local_assemble_system(const Function<2, double> * const,
            const Function<2, double> * const, const typename DoFHandler<2>::active_cell_iterator &,
            AssemblyScratchData<2> &scratch_data, AssemblyCopyData<double> &);

      template void local_assemble_system(const Function<3, double> * const,
            const Function<3, double> * const, const typename DoFHandler<3>::active_cell_iterator &,
            AssemblyScratchData<3> &scratch_data, AssemblyCopyData<double> &);

      template<typename number>
      void copy_local_to_global(SparseMatrix<number> &matrix,
            const AssemblyCopyData<number> &copy_data) {
         for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i) {
            for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
               matrix.add(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j],
                     copy_data.cell_matrix(i, j));
         }
      }

      template void copy_local_to_global(SparseMatrix<double>&, const AssemblyCopyData<double> &);

      template<int dim, int spacedim, typename number>
      void create_laplace_mass_matrix(const DoFHandler<dim, spacedim> &dof,
            const Quadrature<dim> &quad, SparseMatrix<number> &matrix,
            const Function<spacedim, number> * const a,
            const Function<spacedim, number> * const q) {
         Assert(a != nullptr && q != nullptr, ExcZero());
         WorkStream::run(dof.begin_active(), dof.end(),
               std::bind(&local_assemble_system<dim, spacedim, number>, a, q, std::placeholders::_1,
                     std::placeholders::_2, std::placeholders::_3),
               std::bind(&copy_local_to_global<number>, std::ref(matrix), std::placeholders::_1),
               AssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData<double>());
      }
      template void create_laplace_mass_matrix(const DoFHandler<1, 1>&, const Quadrature<1>&,
            SparseMatrix<double>&, const Function<1, double> * const,
            const Function<1, double> * const);

      template void create_laplace_mass_matrix(const DoFHandler<2, 2>&, const Quadrature<2>&,
            SparseMatrix<double>&, const Function<2, double> * const,
            const Function<2, double> * const);

      template void create_laplace_mass_matrix(const DoFHandler<3, 3>&, const Quadrature<3>&,
            SparseMatrix<double>&, const Function<3, double> * const,
            const Function<3, double> * const);
   }
}

