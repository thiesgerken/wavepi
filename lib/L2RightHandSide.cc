/*
 * L2RightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <L2RightHandSide.h>

namespace wavepi {
using namespace dealii;

template<int dim>
L2RightHandSide<dim>::L2RightHandSide(Function<dim>* f) :
		base_rhs(f) {
}

template<int dim>
L2RightHandSide<dim>::~L2RightHandSide() {
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
				update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
AssemblyScratchData<dim>::AssemblyScratchData(
		const AssemblyScratchData &scratch_data) :
		fe_values(scratch_data.fe_values.get_fe(),
				scratch_data.fe_values.get_quadrature(),
				update_values | update_quadrature_points | update_JxW_values) {
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
void local_assemble(const Vector<double> &f,
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
				copy_data.cell_rhs(i) += f[copy_data.local_dof_indices[k]]
						* scratch_data.fe_values.shape_value(k, q_point)
						* scratch_data.fe_values.shape_value(i, q_point)
						* scratch_data.fe_values.JxW(q_point);
}

template<int dim>
void L2RightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof,
		const Quadrature<dim> &quad, Vector<double> &rhs) const {
	base_rhs->set_time(this->get_time());

	auto base_rhs_d = dynamic_cast<DiscretizedFunction<dim>*>(base_rhs);

	if (base_rhs_d != nullptr) {
		Vector<double> coeffs =
				base_rhs_d->get_function_coefficients()[base_rhs_d->get_time_index()];
		Assert(coeffs.size() == dof.n_dofs(),
				ExcDimensionMismatch (coeffs.size() , dof.n_dofs()));

		WorkStream::run(dof.begin_active(), dof.end(),
				std::bind(&local_assemble<dim>, std::ref(coeffs),
						std::placeholders::_1, std::placeholders::_2,
						std::placeholders::_3),
				std::bind(&copy_local_to_global, std::ref(rhs),
						std::placeholders::_1),
				AssemblyScratchData<dim>(dof.get_fe(), quad),
				AssemblyCopyData());
	} else
		VectorTools::create_right_hand_side(dof, quad, *base_rhs, rhs);
}

template class L2RightHandSide<1> ;
template class L2RightHandSide<2> ;
template class L2RightHandSide<3> ;

} /* namespace wavepi */
