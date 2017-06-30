/*
 * DistributionRightHandSide.cc
 *
 *  Created on: 30.06.2017
 *      Author: thies
 */

#include "DistributionRightHandSide.h"

namespace wavepi {
using namespace dealii;

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

template<int dim>
DistributionRightHandSide<dim>::DistributionRightHandSide(Function<dim>* f1,
		Function<dim>* f2) {
	this->f1 = f1;
	this->f2 = f2;
}

template<int dim>
DistributionRightHandSide<dim>::~DistributionRightHandSide() {
}

void copy_local_to_global(Vector<double> &result,
		const AssemblyCopyData &copy_data) {
	for (unsigned int i = 0; i < copy_data.local_dof_indices.size(); ++i)
		result(copy_data.local_dof_indices[i]) += copy_data.cell_rhs(i);
}

template<int dim>
void local_assemble_cc(const Function<dim> * const f1,
		const Function<dim> * const f2,
		const typename DoFHandler<dim>::active_cell_iterator &cell,
		AssemblyScratchData<dim> &scratch_data, AssemblyCopyData &copy_data) {
	const unsigned int dofs_per_cell =
			scratch_data.fe_values.get_fe().dofs_per_cell;
	const unsigned int n_q_points =
			scratch_data.fe_values.get_quadrature().size();

	copy_data.cell_rhs.reinit(dofs_per_cell);
	copy_data.local_dof_indices.resize(dofs_per_cell);
	scratch_data.fe_values.reinit(cell);

	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
		if (f1 != nullptr) {
			const double val1 = f1->value(
					scratch_data.fe_values.quadrature_point(q_point));

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				copy_data.cell_rhs(i) += val1
						* scratch_data.fe_values.shape_value(i, q_point)
						* scratch_data.fe_values.JxW(q_point);
		}

		if (f2 != nullptr) {
			const double val2 = f2->value(
					scratch_data.fe_values.quadrature_point(q_point));

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				copy_data.cell_rhs(i) += val2
						* scratch_data.fe_values.shape_grad(i, q_point)
						* scratch_data.fe_values.JxW(q_point);
		}
	}

	cell->get_dof_indices(copy_data.local_dof_indices);
}

template<int dim>
void local_assemble_dd(const Vector<double> &f1, const Vector<double> &f2,
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
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
				copy_data.cell_rhs(i) += (f1[copy_data.local_dof_indices[k]]
						* scratch_data.fe_values.shape_value(k, q_point)
						* scratch_data.fe_values.shape_value(i, q_point)
						+ f2[copy_data.local_dof_indices[k]]
								* scratch_data.fe_values.shape_value(k, q_point)
								* scratch_data.fe_values.shape_grad(i, q_point))
						* scratch_data.fe_values.JxW(q_point);
}

template<int dim>
void local_assemble_cd(const Function<dim> * const f1, const Vector<double> &f2,
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
		if (f1 != nullptr) {
			const double val1 = f1->value(
					scratch_data.fe_values.quadrature_point(q_point));

			for (unsigned int i = 0; i < dofs_per_cell; ++i)
				copy_data.cell_rhs(i) += val1
						* scratch_data.fe_values.shape_value(i, q_point)
						* scratch_data.fe_values.JxW(q_point);
		}

		for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
				copy_data.cell_rhs(i) += f2[copy_data.local_dof_indices[k]]
						* scratch_data.fe_values.shape_value(k, q_point)
						* scratch_data.fe_values.shape_grad(i, q_point)
						* scratch_data.fe_values.JxW(q_point);
		}
	}
}

template<int dim>
void local_assemble_dc(const Vector<double> &f1, const Function<dim> * const f2,
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
		for (unsigned int i = 0; i < dofs_per_cell; ++i) {
			for (unsigned int k = 0; k < dofs_per_cell; ++k)
				copy_data.cell_rhs(i) += f1[copy_data.local_dof_indices[k]]
						* scratch_data.fe_values.shape_value(k, q_point)
						* scratch_data.fe_values.shape_value(i, q_point)
						* scratch_data.fe_values.JxW(q_point);

			if (f2 != nullptr) {
				const double val2 = f2->value(
						scratch_data.fe_values.quadrature_point(q_point));

				for (unsigned int i = 0; i < dofs_per_cell; ++i)
					copy_data.cell_rhs(i) += val2
							* scratch_data.fe_values.shape_grad(i, q_point)
							* scratch_data.fe_values.JxW(q_point);
			}
		}

}

template<int dim>
void DistributionRightHandSide<dim>::create_right_hand_side(
		const DoFHandler<dim> &dof, const Quadrature<dim> &quad,
		Vector<double> &rhs) const {
	f1->set_time(this->get_time());
	f2->set_time(this->get_time());

	auto f1_d = dynamic_cast<DiscretizedFunction<dim>*>(f1);
	auto f2_d = dynamic_cast<DiscretizedFunction<dim>*>(f2);

	if (f1_d != nullptr)
	Assert(f1_d->get_function_coefficients()[f1_d->get_time_index()].size() == dof.n_dofs(),
			ExcDimensionMismatch (f1_d->get_function_coefficients()[f1_d->get_time_index()].size() , dof.n_dofs()));

	if (f2_d != nullptr)
	Assert(f2_d->get_function_coefficients()[f2_d->get_time_index()].size() == dof.n_dofs(),
			ExcDimensionMismatch (f2_d->get_function_coefficients()[f2_d->get_time_index()].size() , dof.n_dofs()));

	if (f1_d != nullptr && f2_d != nullptr)
		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_cc<dim>,
						std::ref(f1_d->get_function_coefficients()[f1_d->get_time_index()]),
						std::ref(f2_d->get_function_coefficients()[f2_d->get_time_index()]),
						std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
	else if (f1_d == nullptr && f2_d != nullptr)
		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_cd<dim>, f1,
						std::ref(f2_d->get_function_coefficients()[f2_d->get_time_index()]),
						std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
	else if (f1_d != nullptr && f2_d == nullptr)
		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_dc<dim>,
						std::ref(f1_d->get_function_coefficients()[f1_d->get_time_index()]),
						f2, std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
	else
		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble_cc<dim>, f1, f2,
						std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());

}

} /* namespace wavepi */

