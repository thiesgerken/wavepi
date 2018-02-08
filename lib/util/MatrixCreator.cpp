/*
 * MatrixCreator.cc
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_update_flags.h>

#include <util/MatrixCreator.h>

#include <functional>

namespace wavepi {
namespace util {

using namespace dealii;

template <int dim>
MatrixCreator<dim>::LaplaceAssemblyScratchData::LaplaceAssemblyScratchData(const FiniteElement<dim> &fe,
                                                                           const Quadrature<dim> &quad)
    : fe_values(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values) {}

template <int dim>
MatrixCreator<dim>::LaplaceAssemblyScratchData::LaplaceAssemblyScratchData(
    const LaplaceAssemblyScratchData &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                update_values | update_gradients | update_quadrature_points | update_JxW_values) {}

template <int dim>
MatrixCreator<dim>::MassAssemblyScratchData::MassAssemblyScratchData(const FiniteElement<dim> &fe,
                                                                     const Quadrature<dim> &quad)
    : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
MatrixCreator<dim>::MassAssemblyScratchData::MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
void MatrixCreator<dim>::local_assemble_laplace_mass_cc(const Function<dim> *const a, const Function<dim> *const q,
                                                        const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                        LaplaceAssemblyScratchData &scratch_data,
                                                        AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
    const double val_a = a->value(scratch_data.fe_values.quadrature_point(q_point));
    const double val_q = q->value(scratch_data.fe_values.quadrature_point(q_point));

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        copy_data.cell_matrix(i, j) +=
            (val_a * scratch_data.fe_values.shape_grad(i, q_point) * scratch_data.fe_values.shape_grad(j, q_point) +
             val_q * scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.shape_value(j, q_point)) *
            scratch_data.fe_values.JxW(q_point);
    }
  }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void MatrixCreator<dim>::local_assemble_laplace_mass_dd(const Vector<double> &a, const Vector<double> &q,
                                                        const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                        LaplaceAssemblyScratchData &scratch_data,
                                                        AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          copy_data.cell_matrix(i, j) +=
              (a[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
                   scratch_data.fe_values.shape_grad(i, q_point) * scratch_data.fe_values.shape_grad(j, q_point) +
               q[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
                   scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.shape_value(j, q_point)) *
              scratch_data.fe_values.JxW(q_point);
    }
  }
}

template <int dim>
void MatrixCreator<dim>::local_assemble_laplace_mass_cd(const Function<dim> *const a, const Vector<double> &q,
                                                        const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                        LaplaceAssemblyScratchData &scratch_data,
                                                        AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
    const double val_a = a->value(scratch_data.fe_values.quadrature_point(q_point));

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        copy_data.cell_matrix(i, j) += val_a * scratch_data.fe_values.shape_grad(i, q_point) *
                                       scratch_data.fe_values.shape_grad(j, q_point) *
                                       scratch_data.fe_values.JxW(q_point);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          copy_data.cell_matrix(i, j) +=
              q[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
              scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.shape_value(j, q_point) *
              scratch_data.fe_values.JxW(q_point);
      }
    }
  }
}

template <int dim>
void MatrixCreator<dim>::local_assemble_laplace_mass_dc(const Vector<double> &a, const Function<dim> *const q,
                                                        const typename DoFHandler<dim>::active_cell_iterator &cell,
                                                        LaplaceAssemblyScratchData &scratch_data,
                                                        AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
    const double val_q = q->value(scratch_data.fe_values.quadrature_point(q_point));

    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      for (unsigned int j = 0; j < dofs_per_cell; ++j) {
        copy_data.cell_matrix(i, j) += val_q * scratch_data.fe_values.shape_value(i, q_point) *
                                       scratch_data.fe_values.shape_value(j, q_point) *
                                       scratch_data.fe_values.JxW(q_point);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          copy_data.cell_matrix(i, j) +=
              a[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
              scratch_data.fe_values.shape_grad(i, q_point) * scratch_data.fe_values.shape_grad(j, q_point) *
              scratch_data.fe_values.JxW(q_point);
      }
    }
  }
}

template <int dim>
void MatrixCreator<dim>::local_assemble_mass(const Vector<double> &c,
                                             const typename DoFHandler<dim>::active_cell_iterator &cell,
                                             MassAssemblyScratchData &scratch_data, AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_matrix.reinit(dofs_per_cell, dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
          copy_data.cell_matrix(i, j) +=
              c[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
              scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.shape_value(j, q_point) *
              scratch_data.fe_values.JxW(q_point);
  }
}

template <int dim>
void MatrixCreator<dim>::copy_local_to_global(SparseMatrix<double> &matrix, const AssemblyCopyData &copy_data) {
  for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i) {
    for (unsigned int j = 0; j < copy_data.local_dof_indices.size(); ++j)
      matrix.add(copy_data.local_dof_indices[i], copy_data.local_dof_indices[j], copy_data.cell_matrix(i, j));
  }
}

template <int dim>
void MatrixCreator<dim>::create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                    SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> a,
                                                    std::shared_ptr<Function<dim>> q) {
  AssertThrow(a, ExcZero());
  AssertThrow(q, ExcZero());

  WorkStream::run(dof.begin_active(), dof.end(),
                  std::bind(&MatrixCreator<dim>::local_assemble_laplace_mass_cc, a.get(), q.get(),
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                  std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
                  LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template <int dim>
void MatrixCreator<dim>::create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                    SparseMatrix<double> &matrix, std::shared_ptr<Function<dim>> a,
                                                    const Vector<double> &q) {
  AssertThrow(a, ExcZero());
  Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch(q.size(), dof.n_dofs()));

  WorkStream::run(dof.begin_active(), dof.end(),
                  std::bind(&MatrixCreator<dim>::local_assemble_laplace_mass_cd, a.get(), std::ref(q),
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                  std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
                  LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template <int dim>
void MatrixCreator<dim>::create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                    SparseMatrix<double> &matrix, const Vector<double> &a,
                                                    std::shared_ptr<Function<dim>> q) {
  AssertThrow(q, ExcZero());
  Assert(a.size() == dof.n_dofs(), ExcDimensionMismatch(a.size(), dof.n_dofs()));

  WorkStream::run(dof.begin_active(), dof.end(),
                  std::bind(&MatrixCreator<dim>::local_assemble_laplace_mass_dc, std::ref(a), q.get(),
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                  std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
                  LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template <int dim>
void MatrixCreator<dim>::create_laplace_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                    SparseMatrix<double> &matrix, const Vector<double> &a,
                                                    const Vector<double> &q) {
  Assert(a.size() == dof.n_dofs(), ExcDimensionMismatch(a.size(), dof.n_dofs()));
  Assert(q.size() == dof.n_dofs(), ExcDimensionMismatch(q.size(), dof.n_dofs()));

  WorkStream::run(dof.begin_active(), dof.end(),
                  std::bind(&MatrixCreator<dim>::local_assemble_laplace_mass_dd, std::ref(a), std::ref(q),
                            std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
                  std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
                  LaplaceAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template <int dim>
void MatrixCreator<dim>::create_mass_matrix(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                            SparseMatrix<double> &matrix, const Vector<double> &c) {
  Assert(c.size() == dof.n_dofs(), ExcDimensionMismatch(c.size(), dof.n_dofs()));

  WorkStream::run(dof.begin_active(), dof.end(),
                  std::bind(&MatrixCreator<dim>::local_assemble_mass, std::ref(c), std::placeholders::_1,
                            std::placeholders::_2, std::placeholders::_3),
                  std::bind(&MatrixCreator<dim>::copy_local_to_global, std::ref(matrix), std::placeholders::_1),
                  MassAssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template class MatrixCreator<1>;
template class MatrixCreator<2>;
template class MatrixCreator<3>;

} /* namespace util */
} /* namespace wavepi */
