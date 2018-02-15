/*
 * L2RightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <forward/L2ProductRightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

template <int dim>
L2ProductRightHandSide<dim>::L2ProductRightHandSide(std::shared_ptr<DiscretizedFunction<dim>> f1,
                                                    std::shared_ptr<DiscretizedFunction<dim>> f2)
    : func1(f1), func2(f2) {}

template <int dim>
L2ProductRightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe,
                                                                      const Quadrature<dim> &quad)
    : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
L2ProductRightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
    : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                update_values | update_quadrature_points | update_JxW_values) {}

template <int dim>
void L2ProductRightHandSide<dim>::copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data) {
  for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
    result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template <int dim>
void L2ProductRightHandSide<dim>::local_assemble(const Vector<double> &f1, const Vector<double> &f2,
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
      for (unsigned int k1 = 0; k1 < dofs_per_cell; ++k1)
        for (unsigned int k2 = 0; k2 < dofs_per_cell; ++k2)
          copy_data.cell_rhs(i) -=
              f1[copy_data.local_dof_indices[k1]] * scratch_data.fe_values.shape_value(k1, q_point) *
              f2[copy_data.local_dof_indices[k2]] * scratch_data.fe_values.shape_value(k2, q_point) *
              scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.JxW(q_point);
}

template <int dim>
void L2ProductRightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
                                                         Vector<double> &rhs) const {
  func1->set_time(this->get_time());
  func2->set_time(this->get_time());

  Vector<double> coeffs1 = func1->get_function_coefficients(func1->get_time_index());
  Assert(coeffs1.size() == dof.n_dofs(), ExcDimensionMismatch(coeffs1.size(), dof.n_dofs()));

  Vector<double> coeffs2 = func2->get_function_coefficients(func2->get_time_index());
  Assert(coeffs2.size() == dof.n_dofs(), ExcDimensionMismatch(coeffs2.size(), dof.n_dofs()));

  WorkStream::run(
      dof.begin_active(), dof.end(),
      std::bind(&L2ProductRightHandSide<dim>::local_assemble, *this, std::ref(coeffs1), std::ref(coeffs2),
                std::placeholders::_1, std::placeholders::_2, std::placeholders::_3),
      std::bind(&L2ProductRightHandSide<dim>::copy_local_to_global, *this, std::ref(rhs), std::placeholders::_1),
      AssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template <int dim>
inline std::shared_ptr<DiscretizedFunction<dim>> L2ProductRightHandSide<dim>::get_func1() const {
  return func1;
}

template <int dim>
inline void L2ProductRightHandSide<dim>::set_func1(std::shared_ptr<DiscretizedFunction<dim>> func1) {
  this->func1 = func1;
}

template <int dim>
inline std::shared_ptr<DiscretizedFunction<dim>> L2ProductRightHandSide<dim>::get_func2() const {
  return func2;
}

template <int dim>
inline void L2ProductRightHandSide<dim>::set_func2(std::shared_ptr<DiscretizedFunction<dim>> func2) {
  this->func2 = func2;
}

template class L2ProductRightHandSide<1>;
template class L2ProductRightHandSide<2>;
template class L2ProductRightHandSide<3>;

} /* namespace forward */
} /* namespace wavepi */
