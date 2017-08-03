/*
 * DivRightHandSideAdjoint.cpp
 *
 *  Created on: 03.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/types.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>

#include <forward/DiscretizedFunction.h>
#include <forward/DivRightHandSideAdjoint.h>

#include <functional>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
DivRightHandSideAdjoint<dim>::DivRightHandSideAdjoint(std::shared_ptr<Function<dim>> a, std::shared_ptr<Function<dim>> u)
      : a(a), u(u) {
}

template<int dim>
DivRightHandSideAdjoint<dim>::~DivRightHandSideAdjoint() {
}

template<int dim>
DivRightHandSideAdjoint<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
DivRightHandSideAdjoint<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
            update_values | update_gradients | update_quadrature_points | update_JxW_values) {
}

template<int dim>
void DivRightHandSideAdjoint<dim>::copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data) {
   for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
      result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template<int dim>
void DivRightHandSideAdjoint<dim>::local_assemble_dd(const Vector<double> &a, const Vector<double> &u,
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
         for (unsigned int ka = 0; ka < dofs_per_cell; ++ka)
            for (unsigned int ku = 0; ku < dofs_per_cell; ++ku)
               copy_data.cell_rhs(i) -= a[copy_data.local_dof_indices[ka]]
                     * scratch_data.fe_values.shape_grad(ka, q_point) * u[copy_data.local_dof_indices[ku]]
                     * scratch_data.fe_values.shape_grad(ku, q_point)
                     * scratch_data.fe_values.shape_value(i, q_point) * scratch_data.fe_values.JxW(q_point);
}

template<int dim>
void DivRightHandSideAdjoint<dim>::local_assemble_cc(const Function<dim> * const a, const Function<dim> * const u,
      const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
      AssemblyCopyData &copy_data) {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_rhs.reinit(dofs_per_cell);
   copy_data.local_dof_indices.resize(dofs_per_cell);
   scratch_data.fe_values.reinit(cell);

   cell->get_dof_indices(copy_data.local_dof_indices);

   for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
      auto grad_a = a->gradient(scratch_data.fe_values.quadrature_point(q_point));
      auto grad_u = u->gradient(scratch_data.fe_values.quadrature_point(q_point));

      for (unsigned int i = 0; i < dofs_per_cell; ++i)
         copy_data.cell_rhs(i) -= grad_a * grad_u * scratch_data.fe_values.shape_value(i, q_point)
               * scratch_data.fe_values.JxW(q_point);
   }
}

template<int dim>
void DivRightHandSideAdjoint<dim>::create_right_hand_side(const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
      Vector<double> &rhs) const {
   AssertThrow(a != nullptr, ExcZero());
   AssertThrow(u != nullptr, ExcZero());

   a->set_time(this->get_time());
   u->set_time(this->get_time());

   auto a_d = dynamic_cast<DiscretizedFunction<dim>*>(a.get());
   auto u_d = dynamic_cast<DiscretizedFunction<dim>*>(u.get());

   if (a_d != nullptr && u_d != nullptr) {
      Vector<double> ca = a_d->get_function_coefficients()[a_d->get_time_index()];
      Vector<double> cu = u_d->get_function_coefficients()[u_d->get_time_index()];

      Assert(ca.size() == dof.n_dofs(), ExcDimensionMismatch (ca.size() , dof.n_dofs()));
      Assert(cu.size() == dof.n_dofs(), ExcDimensionMismatch (cu.size() , dof.n_dofs()));

      WorkStream::run(dof.begin_active(), dof.end(),
            std::bind(&DivRightHandSideAdjoint<dim>::local_assemble_dd, *this, std::ref(ca), std::ref(cu), std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3),
            std::bind(&DivRightHandSideAdjoint<dim>::copy_local_to_global, *this, std::ref(rhs), std::placeholders::_1),
            AssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
   } else
      WorkStream::run(dof.begin_active(), dof.end(),
            std::bind(&DivRightHandSideAdjoint<dim>::local_assemble_cc, *this, a.get(), u.get(), std::placeholders::_1, std::placeholders::_2,
                  std::placeholders::_3),
            std::bind(&DivRightHandSideAdjoint<dim>::copy_local_to_global, *this, std::ref(rhs), std::placeholders::_1),
            AssemblyScratchData(dof.get_fe(), quad), AssemblyCopyData());
}

template<int dim>
DiscretizedFunction<dim> DivRightHandSideAdjoint<dim>::run_adjoint( std::shared_ptr<SpaceTimeMesh<dim>> mesh, std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad)  {
 DiscretizedFunction<dim> target (mesh, dof);

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      this->set_time(mesh->get_times()[i]);

      Vector<double> tmp(dof->n_dofs());
      this->create_right_hand_side(*dof.get(), quad, tmp);
      target.set(i, tmp);
   }

   return target;
}

template class DivRightHandSideAdjoint<1> ;
template class DivRightHandSideAdjoint<2> ;
template class DivRightHandSideAdjoint<3> ;

} /* namespace forward */
} /* namespace wavepi */

