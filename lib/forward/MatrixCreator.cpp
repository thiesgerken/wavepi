/*
 * MatrixCreator.cc
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_update_flags.h>
#include <forward/MatrixCreator.h>

#include <functional>

namespace wavepi {
namespace forward {

using namespace dealii;

inline double square(const double x) {
   return x * x;
}

template<int dim>
MatrixCreator<dim>::LaplaceAssemblyScratchData::LaplaceAssemblyScratchData(const FiniteElement<dim> &fe,
      const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
MatrixCreator<dim>::LaplaceAssemblyScratchData::LaplaceAssemblyScratchData(
      const LaplaceAssemblyScratchData &scratch_data)
      :
            fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                  update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
MatrixCreator<dim>::MassAssemblyScratchData::MassAssemblyScratchData(const FiniteElement<dim> &fe,
      const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
MatrixCreator<dim>::MassAssemblyScratchData::MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data)
      :
            fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                  update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
void MatrixCreator<dim>::local_assemble_A_cc(const Function<dim> * const rho, const Function<dim> * const q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_a = 1.0 / rho->value(scratch_data.fe_values.quadrature_point(q_point));
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
void MatrixCreator<dim>::local_assemble_A_dd(const Vector<double> &rho, const Vector<double> &q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      double val_a = 0;
      double val_q = 0;

      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
         const double val_shape = scratch_data.fe_values.shape_value(k, q_point);

         val_a += rho[copy_data.local_dof_indices[k]] * val_shape;
         val_q += q[copy_data.local_dof_indices[k]] * val_shape;
      }
      val_a = 1.0 / val_a;

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += (val_a * scratch_data.fe_values.shape_grad(i, q_point)
                  * scratch_data.fe_values.shape_grad(j, q_point)
                  + val_q * scratch_data.fe_values.shape_value(i, q_point)
                        * scratch_data.fe_values.shape_value(j, q_point)) * scratch_data.fe_values.JxW(q_point);
   }
}

template<int dim>
void MatrixCreator<dim>::local_assemble_A_cd(const Function<dim> * const rho, const Vector<double> &q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_a = 1.0 / rho->value(scratch_data.fe_values.quadrature_point(q_point));

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
void MatrixCreator<dim>::local_assemble_A_dc(const Vector<double> &rho, const Function<dim> * const q,
      const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_q = q->value(scratch_data.fe_values.quadrature_point(q_point));

      double val_a = 0;
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
         val_a += rho[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point);
      val_a = 1.0 / val_a;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j) {
            copy_data.cell_matrix(i, j) += (val_q * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point)
                  + val_a * scratch_data.fe_values.shape_grad(i, q_point)
                        * scratch_data.fe_values.shape_grad(j, q_point)) * scratch_data.fe_values.JxW(q_point);
         }
      }
   }
}

template<int dim>
void MatrixCreator<dim>::local_assemble_mass(const Vector<double> &c,
      const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
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

template<int dim>
void MatrixCreator<dim>::local_assemble_C_cc(const Function<dim> * const rho, const Function<dim> * const c,
      const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_a = 1.0 / rho->value(scratch_data.fe_values.quadrature_point(q_point));
      const double val_c = 1.0 / square(c->value(scratch_data.fe_values.quadrature_point(q_point)));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += val_c * val_a * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
   }

   cell->get_dof_indices(copy_data.local_dof_indices);
}

template<int dim>
void MatrixCreator<dim>::local_assemble_C_dd(const Vector<double> &rho, const Vector<double> &c,
      const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      double val_a = 0;
      double val_c = 0;

      for (unsigned int k = 0; k < dofs_per_cell; ++k) {
         const double val_shape = scratch_data.fe_values.shape_value(k, q_point);

         val_a += rho[copy_data.local_dof_indices[k]] * val_shape;
         val_c += c[copy_data.local_dof_indices[k]] * val_shape;
      }

      val_a = 1.0 / val_a;
      val_c = 1.0 / (val_c * val_c);

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += val_c * val_a * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
   }
}

template<int dim>
void MatrixCreator<dim>::local_assemble_C_cd(const Function<dim> * const rho, const Vector<double> &c,
      const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_a = 1.0 / rho->value(scratch_data.fe_values.quadrature_point(q_point));

      double val_c = 0;
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
         val_c += c[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point);
      val_c = 1.0 / (val_c * val_c);

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += val_c * val_a * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
      }
   }
}

template<int dim>
void MatrixCreator<dim>::local_assemble_C_dc(const Vector<double> &rho, const Function<dim> * const c,
      const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      const double val_c = 1.0 / square(c->value(scratch_data.fe_values.quadrature_point(q_point)));

      double val_a = 0;
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
         val_a += rho[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point);
      val_a = 1.0 / val_a;

      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
         for (unsigned int j = 0; j < dofs_per_cell; ++j)
            copy_data.cell_matrix(i, j) += val_c * val_a * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.shape_value(j, q_point) * scratch_data.fe_values.JxW(q_point);
      }
   }
}

template<int dim>
void MatrixCreator<dim>::copy_local_to_global(SparseMatrix<double> &matrix, const AssemblyCopyData &copy_data) {
   for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i) {
      for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
         matrix.add(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j], copy_data.cell_matrix(i, j));
   }
}

template<int dim>
void MatrixCreator<dim>::create_A_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> rho, std::shared_ptr<Function<dim>> q) {
   AssertThrow(rho, ExcZero());
   AssertThrow(q, ExcZero());

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_A_cc, rho.get(), q.get(), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_A_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> rho, const Vector<double> &q) {
   AssertThrow(rho, ExcZero());
   Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch(q.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_A_cd, rho.get(), std::ref(q), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_A_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, const Vector<double> &rho, std::shared_ptr<Function<dim>> q) {
   AssertThrow(q, ExcZero());
   Assert(rho.size() == dof.n_dofs(), ExcDimensionMismatch(rho.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_A_dc, std::ref(rho), q.get(), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_A_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, const Vector<double> &rho, const Vector<double> &q) {
   Assert(rho.size() == dof.n_dofs(), ExcDimensionMismatch(rho.size(), dof.n_dofs()));
   Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch(q.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_A_dd, std::ref(rho), std::ref(q), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_C_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> rho, std::shared_ptr<Function<dim>> c) {
   AssertThrow(rho, ExcZero());
   AssertThrow(c, ExcZero());

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_C_cc, rho.get(), c.get(), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_C_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> rho, const Vector<double> &c) {
   AssertThrow(rho, ExcZero());
   Assert(c.size() == dof.n_dofs(), ExcDimensionMismatch(c.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_C_cd, rho.get(), std::ref(c), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_C_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, const Vector<double> &rho, std::shared_ptr<Function<dim>> c) {
   AssertThrow(c, ExcZero());
   Assert(rho.size() == dof.n_dofs(), ExcDimensionMismatch(rho.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_C_dc, std::ref(rho), c.get(), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_C_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, const Vector<double> &rho, const Vector<double> &c) {
   Assert(rho.size() == dof.n_dofs(), ExcDimensionMismatch(rho.size(), dof.n_dofs()));
   Assert(c.size() == dof.n_dofs(), ExcDimensionMismatch(c.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_C_dd, std::ref(rho), std::ref(c), std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
void MatrixCreator<dim>::create_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      SparseMatrix<double> &matrix, const Vector<double> &c) {
   Assert(c.size() == dof.n_dofs(), ExcDimensionMismatch(c.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&MatrixCreator<dim>::local_assemble_mass, std::ref(c), std::placeholders::_1, std::placeholders::_2,
               std::placeholders::_3),
         std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
         MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template class MatrixCreator<1> ;
template class MatrixCreator<2> ;
template class MatrixCreator<3> ;

}  // namespace forward
} /* namespace wavepi */
