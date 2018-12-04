/*
 * DivRightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <base/DiscretizedFunction.h>
#include <forward/DivRightHandSide.h>

#include <functional>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

template<int dim>
DivRightHandSide<dim>::DivRightHandSide(std::shared_ptr<DiscretizedFunction<dim>> a,
      std::shared_ptr<DiscretizedFunction<dim>> b, std::shared_ptr<DiscretizedFunction<dim>> u)
      : a(a), b(b), u(u) {
}

template<int dim>
DivRightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe,
      const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
DivRightHandSide<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
      :
            fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
                  update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
void DivRightHandSide<dim>::copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data) {
   for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
      result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template<int dim>
void DivRightHandSide<dim>::local_assemble(const Vector<double> &a, const Vector<double> &b, const Vector<double> &u,
      const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_rhs.reinit(dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         for (unsigned int ka = 0; ka < dofs_per_cell; ++ka) {
            for (unsigned int ku = 0; ku < dofs_per_cell; ++ku)
               copy_data.cell_rhs(i) -= a[copy_data.local_dof_indices[ka]]
                     * scratch_data.fe_values.shape_value(ka, q_point) * u[copy_data.local_dof_indices[ku]]
                     * scratch_data.fe_values.shape_grad(ku, q_point) * scratch_data.fe_values.shape_grad(i, q_point)
                     * scratch_data.fe_values.JxW(q_point);

            copy_data.cell_rhs(i) += b[copy_data.local_dof_indices[ka]]
                  * scratch_data.fe_values.shape_value(ka, q_point) * scratch_data.fe_values.shape_value(i, q_point)
                  * scratch_data.fe_values.JxW(q_point);
         }
}

template<int dim>
void DivRightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      Vector<double> &rhs) const {
   AssertThrow(a != nullptr, ExcZero());
   AssertThrow(b != nullptr, ExcZero());
   AssertThrow(u != nullptr, ExcZero());

   const Vector<double> &ca = a->get_function_coefficients_by_time(this->get_time());
   const Vector<double> &cb = b->get_function_coefficients_by_time(this->get_time());
   const Vector<double> &cu = u->get_function_coefficients_by_time(this->get_time());

   Assert(ca.size() == dof.n_dofs(), ExcDimensionMismatch(ca.size(), dof.n_dofs()));
   Assert(cb.size() == dof.n_dofs(), ExcDimensionMismatch(cb.size(), dof.n_dofs()));
   Assert(cu.size() == dof.n_dofs(), ExcDimensionMismatch(cu.size(), dof.n_dofs()));

   WorkStream::run(dof.begin_active(), dof.end(),
         std::bind(&DivRightHandSide<dim>::local_assemble, *this, ca, cb, cu, std::placeholders::_1,
               std::placeholders::_2, std::placeholders::_3),
         std::bind(&DivRightHandSide<dim>::copy_local_to_global, *this, std::ref(rhs), std::placeholders::_1),
         AssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template class DivRightHandSide<1> ;
template class DivRightHandSide<2> ;
template class DivRightHandSide<3> ;

} /* namespace forward */
} /* namespace wavepi */
