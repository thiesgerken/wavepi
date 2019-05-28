/*
 * L2RightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>

#include <functional>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

template <int dim>
L2RightHandSide<dim>::L2RightHandSide(std::shared_ptr<Function<dim>> f) : base_rhs(f) {}

template <int dim>
L2RightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe,
                                                               const Quadrature<dim> &quad)
    : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
L2RightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
void L2RightHandSide<dim>::copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data) {
  for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
    result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template <int dim>
std::shared_ptr<Function<dim>> L2RightHandSide<dim>::get_base_rhs() const {
  return base_rhs;
}

template <int dim>
void L2RightHandSide<dim>::set_base_rhs(std::shared_ptr<Function<dim>> base_rhs) {
  this->base_rhs = base_rhs;
}

template <int dim>
void L2RightHandSide<dim>::local_assemble(const Vector<double> &f,
                                          const typename DoFHandler<dim>::active_cell_iterator &cell,
                                          AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data) {
  const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
  const unsigned int n_q_points    = scratch_data.fe_values.get_quadrature().size();

  copy_data.cell_rhs.reinit(dofs_per_cell);
  copy_data.local_dof_indices.resize(dofs_per_cell);
  scratch_data.fe_values.reinit(cell);

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        copy_data.cell_rhs(i) += f[copy_data.local_dof_indices[k]] * scratch_data.fe_values.shape_value(k, q_point) *
                                 scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.JxW(q_point);
}

template <int dim>
void L2RightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                  Vector<double> &rhs) const {
  AssertThrow(base_rhs, ExcInternalError());
  base_rhs->set_time(this->get_time());

  auto base_rhs_d = std::dynamic_pointer_cast<DiscretizedFunction<dim>>(base_rhs);

  if (base_rhs_d) {
    Vector<double> coeffs = base_rhs_d->get_function_coefficients(base_rhs_d->get_time_index());
    Assert(coeffs.size() == dof.n_dofs(), ExcDimensionMismatch(coeffs.size(), dof.n_dofs()));

    WorkStream::run(dof.begin_active(), dof.end(),
                    std::bind(&L2RightHandSide<dim>::local_assemble, *this, std::ref(coeffs), std::placeholders::_1,
                              std::placeholders::_2, std::placeholders::_3),
                    std::bind(&L2RightHandSide<dim>::copy_local_to_global, *this, std::ref(rhs), std::placeholders::_1),
                    AssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
  } else
    VectorTools::create_right_hand_side(dof, quad, *base_rhs.get(), rhs);
}

template class L2RightHandSide<1>;
template class L2RightHandSide<2>;
template class L2RightHandSide<3>;

} /* namespace forward */
} /* namespace wavepi */
