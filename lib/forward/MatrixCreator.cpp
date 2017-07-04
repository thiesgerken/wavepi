/*
 * MatrixCreator.cc
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#include "forward/MatrixCreator.h"

namespace wavepi {
namespace forward {
namespace MatrixCreator {
using namespace dealii;

template<int dim>
struct LaplaceAssemblyScratchData {
      LaplaceAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
      LaplaceAssemblyScratchData(const LaplaceAssemblyScratchData &scratch_data);
      FEValues<dim> fe_values;
};

template<int dim>
struct MassAssemblyScratchData {
      MassAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
      MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data);
      FEValues<dim> fe_values;
};

struct AssemblyCopyData {
      FullMatrix<double> cell_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
};

template<int dim>
LaplaceAssemblyScratchData<dim>::LaplaceAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
LaplaceAssemblyScratchData<dim>::LaplaceAssemblyScratchData(const LaplaceAssemblyScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
            update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
MassAssemblyScratchData<dim>::MassAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
MassAssemblyScratchData<dim>::MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
            update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
void local_assemble_laplace_mass_cc(const Function<dim> * const a, const Function<dim> * const q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData<dim> &scratch_data,
      AssemblyCopyData &copy_data) {
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
            copy_data.cell_matrix(i, j) += (val_a * scratch_data.fe_values.shape_grad(i, q_point)
                  * scratch_data.fe_values.shape_grad(j, q_point)
                  + val_q * scratch_data.fe_values.shape_value(i, q_point)
                        * scratch_data.fe_values.shape_value(j, q_point)) * scratch_data.fe_values.JxW(q_point);
      }
   }

   cell->get_dof_indices(copy_data.local_dof_indices);
}

template<int dim>
void local_assemble_laplace_mass_dd(const Vector<double> &a, const Vector<double> &q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData<dim> &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
               copy_data.cell_matrix(i, j) += (a[copy_data.local_dof_indices[k]]
                     * scratch_data.fe_values.shape_value(k, q_point) * scratch_data.fe_values.shape_grad(i, q_point)
                     * scratch_data.fe_values.shape_grad(j, q_point)
                     + q[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point)
                           * scratch_data.fe_values.shape_value(i, q_point)
                           * scratch_data.fe_values.shape_value(j, q_point)) * scratch_data.fe_values.JxW(q_point);
      }
   }
}

template<int dim>
void local_assemble_laplace_mass_cd(const Function<dim> * const a, const Vector<double> &q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData<dim> &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_a = a->value(scratch_data.fe_values.quadrature_point(q_point));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            copy_data.cell_matrix(i, j) += val_a * scratch_data.fe_values.shape_grad(i, q_point)
                  * scratch_data.fe_values.shape_grad(j, q_point) * scratch_data.fe_values.JxW(q_point);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
               copy_data.cell_matrix(i, j) += q[copy_data.local_dof_indices[k]]
                     * scratch_data.fe_values.shape_value(k, q_point) * scratch_data.fe_values.shape_value(i, q_point)
                     * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
         }
      }
   }
}

template<int dim>
void local_assemble_laplace_mass_dc(const Vector<double> &a, const Function<dim> * const q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData<dim> &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_q = q->value(scratch_data.fe_values.quadrature_point(q_point));

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            copy_data.cell_matrix(i, j) += val_q * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
               copy_data.cell_matrix(i, j) += a[copy_data.local_dof_indices[k]]
                     * scratch_data.fe_values.shape_value(k, q_point) * scratch_data.fe_values.shape_grad(i, q_point)
                     * scratch_data.fe_values.shape_grad(j, q_point) * scratch_data.fe_values.JxW(q_point);
         }
      }
   }
}

template<int dim>
void local_assemble_mass(const Vector<double> &c, const typename DoFHandler<dim>::active_cell_iterator &cell,
      MassAssemblyScratchData<dim> &scratch_data, AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            for (unsigned int k = 0; k < dofs_per_cell; ++k)
               copy_data.cell_matrix(i, j) += c[copy_data.local_dof_indices[k]]
                     * scratch_data.fe_values.shape_value(k, q_point) * scratch_data.fe_values.shape_value(i, q_point)
                     * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
   }

}

void copy_local_to_global(SparseMatrix<double> &matrix, const AssemblyCopyData &copy_data) {
   for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i) {
      for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
         matrix.add(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j], copy_data.cell_matrix(i, j));
   }
}

template<int dim>
void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
      const Function<dim> * const a, const Function<dim> * const q) {
   Assert(a != nullptr && q != nullptr, ExcZero());

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&local_assemble_laplace_mass_cc<dim>, a, q, std::placeholders::_1, std::placeholders::_2,
               std::placeholders::_3), std::bind(&copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
      const Function<dim> * const a, const Vector<double> &q) {
   Assert(a != nullptr, ExcZero());
   Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch (q.size() , dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&local_assemble_laplace_mass_cd<dim>, a, std::ref(q), std::placeholders::_1, std::placeholders::_2,
               std::placeholders::_3), std::bind(&copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
      const Vector<double> &a, const Function<dim> * const q) {
   Assert(q != nullptr, ExcZero());
   Assert(a.size() == dof.n_dofs(), ExcDimensionMismatch (a.size() , dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&local_assemble_laplace_mass_dc<dim>, std::ref(a), q, std::placeholders::_1, std::placeholders::_2,
               std::placeholders::_3), std::bind(&copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
      const Vector<double> &a, const Vector<double> &q) {
   Assert(a.size() == dof.n_dofs(), ExcDimensionMismatch (a.size() , dof.n_dofs()));
   Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch (q.size() , dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&local_assemble_laplace_mass_dd<dim>, std::ref(a), std::ref(q), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void create_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
      const Vector<double> &c) {
   Assert(c.size() == dof.n_dofs(), ExcDimensionMismatch (c.size() , dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&local_assemble_mass<dim>, std::ref(c), std::placeholders::_1, std::placeholders::_2,
               std::placeholders::_3), std::bind(&copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData<dim>(dof.get_fe(), quad), AssemblyCopyData());
}

template void create_laplace_mass_matrix(const DoFHandler<1>&, const Quadrature<1>&, SparseMatrix<double>&,
      const Function<1, double> * const, const Function<1> * const);

template void create_laplace_mass_matrix(const DoFHandler<2>&, const Quadrature<2>&, SparseMatrix<double>&,
      const Function<2, double> * const, const Function<2> * const);

template void create_laplace_mass_matrix(const DoFHandler<3>&, const Quadrature<3>&, SparseMatrix<double>&,
      const Function<3> * const, const Function<3> * const);

template void create_laplace_mass_matrix(const DoFHandler<1>&, const Quadrature<1>&, SparseMatrix<double>&,
      const Vector<double> &, const Function<1> * const);

template void create_laplace_mass_matrix(const DoFHandler<2>&, const Quadrature<2>&, SparseMatrix<double>&,
      const Vector<double> &, const Function<2> * const);

template void create_laplace_mass_matrix(const DoFHandler<3>&, const Quadrature<3>&, SparseMatrix<double>&,
      const Vector<double> &, const Function<3> * const);

template void create_laplace_mass_matrix(const DoFHandler<1>&, const Quadrature<1>&, SparseMatrix<double>&,
      const Function<1, double> * const, const Vector<double> &);

template void create_laplace_mass_matrix(const DoFHandler<2>&, const Quadrature<2>&, SparseMatrix<double>&,
      const Function<2, double> * const, const Vector<double> &);

template void create_laplace_mass_matrix(const DoFHandler<3>&, const Quadrature<3>&, SparseMatrix<double>&,
      const Function<3> * const, const Vector<double> &);

template void create_laplace_mass_matrix(const DoFHandler<1>&, const Quadrature<1>&, SparseMatrix<double>&,
      const Vector<double> &, const Vector<double> &);

template void create_laplace_mass_matrix(const DoFHandler<2>&, const Quadrature<2>&, SparseMatrix<double>&,
      const Vector<double> &, const Vector<double> &);

template void create_laplace_mass_matrix(const DoFHandler<3>&, const Quadrature<3>&, SparseMatrix<double>&,
      const Vector<double> &, const Vector<double> &);

template void create_mass_matrix(const DoFHandler<1>&, const Quadrature<1>&, SparseMatrix<double>&,
      const Vector<double> &);

template void create_mass_matrix(const DoFHandler<2>&, const Quadrature<2>&, SparseMatrix<double>&,
      const Vector<double> &);

template void create_mass_matrix(const DoFHandler<3>&, const Quadrature<3>&, SparseMatrix<double>&,
      const Vector<double> &);
}
} /* namespace forward */
} /* namespace wavepi */
