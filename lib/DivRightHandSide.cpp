/*
 * DivRightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <DivRightHandSide.h>

namespace wavepi {
using namespace dealii;

template<int dim>
DivRightHandSide<dim>::DivRightHandSide(Function<dim>* a, Function<dim>* u) :
		a(a), u(u) {
}

template<int dim>
DivRightHandSide<dim>::~DivRightHandSide() {
}

template<int dim>
struct AssemblyScratchData {
	AssemblyScratchData(const FiniteElement<dim> &fe,
			const Quadrature<dim> &quad);
	AssemblyScratchData(const AssemblyScratchData &scratch_data);
	FEValues<dim> fe_values;
};

template<int dim>
AssemblyScratchData<dim>::AssemblyScratchData(const FiniteElement<dim> &fe,
		const Quadrature<dim> &quad) :
		fe_values(fe, quad,
				update_values | update_gradients | update_quadrature_points
						| update_JxW_values) {
}

template<int dim>
AssemblyScratchData<dim>::AssemblyScratchData(
		const AssemblyScratchData &scratch_data) :
		fe_values(scratch_data.fe_values.get_fe(),
				scratch_data.fe_values.get_quadrature(),
				update_values | update_gradients | update_quadrature_points
						| update_JxW_values) {
}

struct AssemblyCopyData {
	Vector<double> cell_rhs;
	std::vector<types::global_dof_index> local_dof_indices;
};

void copy_local_to_global(Vector<double> &result,
		const AssemblyCopyData &copy_data) {
	for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
		result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template<int dim>
void local_assemble_dd(const Vector<double> &a, const Vector<double> &u,
		const typename DoFHandler<dim>::active_cell_iterator &cell,
		AssemblyScratchData<dim> &scratch_data, AssemblyCopyData &copy_data) {
	const unsigned int dofs_per_cell =
			scratch_data.fe_values.get_fe().dofs_per_cell;
	const unsigned int n_q_points =
			scratch_data.fe_values.get_quadrature().size();

	copy_data.cell_rhs.reinit(dofs_per_cell);
	copy_data.local_dof_indices.resize(dofs_per_cell);
	scratch_data.fe_values.reinit(cell);

	cell->get_dof_indices(copy_data.local_dof_indices);

	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		for (unsigned int i = 0; i < dofs_per_cell; ++i)
			for (unsigned int ka = 0; ka < dofs_per_cell; ++ka)
				for (unsigned int ku = 0; ku < dofs_per_cell; ++ku)
					copy_data.cell_rhs(i) -= a[copy_data.local_dof_indices[ka]]
							* scratch_data.fe_values.shape_value(ka, q_point)
							* u[copy_data.local_dof_indices[ku]]
							* scratch_data.fe_values.shape_grad(ku, q_point)
							* scratch_data.fe_values.shape_grad(i, q_point)
							* scratch_data.fe_values.JxW(q_point);
}

template<int dim>
void local_assemble_cc(const Function<dim> * const a,
		const Function<dim> * const u,
		const typename DoFHandler<dim>::active_cell_iterator &cell,
		AssemblyScratchData<dim> &scratch_data, AssemblyCopyData &copy_data) {
	const unsigned int dofs_per_cell =
			scratch_data.fe_values.get_fe().dofs_per_cell;
	const unsigned int n_q_points =
			scratch_data.fe_values.get_quadrature().size();

	copy_data.cell_rhs.reinit(dofs_per_cell);
	copy_data.local_dof_indices.resize(dofs_per_cell);
	scratch_data.fe_values.reinit(cell);

	cell->get_dof_indices(copy_data.local_dof_indices);

	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
		const double val_a = a->value(
				scratch_data.fe_values.quadrature_point(q_point));
		auto grad_u = u->gradient(
				scratch_data.fe_values.quadrature_point(q_point));

		for (unsigned int i = 0; i < dofs_per_cell; ++i)
			copy_data.cell_rhs(i) -= val_a * grad_u
					* scratch_data.fe_values.shape_grad(i, q_point)
					* scratch_data.fe_values.JxW(q_point);
	}
}

template<int dim>
void DivRightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof,
		const Quadrature<dim> &quad, Vector<double> &rhs) const {
	Assert(a != nullptr, ExcZero());
	Assert(u != nullptr, ExcZero());

	a->set_time(this->get_time());
	u->set_time(this->get_time());

	auto a_d = dynamic_cast<DiscretizedFunction<dim>*>(a);
	auto u_d = dynamic_cast<DiscretizedFunction<dim>*>(u);

	if (a_d != nullptr && u_d != nullptr) {
		Vector<double> ca =
				a_d->get_function_coefficients()[a_d->get_time_index()];
		Vector<double> cu =
				u_d->get_function_coefficients()[u_d->get_time_index()];

		Assert(ca.size() == dof.n_dofs(),
				ExcDimensionMismatch (ca.size() , dof.n_dofs()));
		Assert(cu.size() == dof.n_dofs(),
				ExcDimensionMismatch (cu.size() , dof.n_dofs()));

		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_dd<dim>, std::ref(ca), std::ref(cu),
						std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
	} else
		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_cc<dim>, a, u, std::placeholders::_1,
						std::placeholders::_2, std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
}

template class DivRightHandSide<1> ;
template class DivRightHandSide<2> ;
template class DivRightHandSide<3> ;

} /* namespace wavepi */
