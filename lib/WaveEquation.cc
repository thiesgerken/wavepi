/*
 * WaveEquation.cc
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

/*
 * based on step23.cc from the deal.II tutorials
 */

#include "WaveEquation.h"

using namespace dealii;

namespace wavepi {
template class WaveEquation<1> ;
template class WaveEquation<2> ;
template class WaveEquation<3> ;

template<int dim>
WaveEquation<dim>::WaveEquation() :
		initialValuesU(&zero), initialValuesV(&zero), boundaryValuesU(
				&zero), boundaryValuesV(&zero), rightHandSide(&zero), paramC(&one), paramNu(
				&zero), paramA(&one), paramQ(&zero), theta(0.5), endTime(1), time_step(1. / 64), fe(1), dof_handler(triangulation), time(time_step), timestep_number(
						1) {
}

// set up the mesh, DoFHandler, and
// matrices and vectors at the beginning of the program, i.e. before the
// first time step. The first few lines are pretty much standard if you've
// read through the tutorial programs at least up to step-6:
template<int dim>
void WaveEquation<dim>::setup_system() {
	GridGenerator::hyper_cube(triangulation, -1, 1);
	triangulation.refine_global(5);

	std::cout << "Number of active cells: " << triangulation.n_active_cells()
			<< std::endl;

	dof_handler.distribute_dofs(fe);

	std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< std::endl << std::endl;

	DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
	DoFTools::make_sparsity_pattern(dof_handler, dsp);
	sparsity_pattern.copy_from(dsp);

	// Then comes a block where we have to initialize the 3 matrices we need
	// in the course of the program: the mass matrix, the Laplace matrix, and
	// the matrix $M+k^2\theta^2A$ used when solving for $U^n$ in each time
	// step.
	//
	// When setting up these matrices, note that they all make use of the same
	// sparsity pattern object. Finally, the reason why matrices and sparsity
	// patterns are separate objects in deal.II (unlike in many other finite
	// element or linear algebra classes) becomes clear: in a significant
	// fraction of applications, one has to hold several matrices that happen
	// to have the same sparsity pattern, and there is no reason for them not
	// to share this information, rather than re-building and wasting memory
	// on it several times.
	//
	// After initializing all of these matrices, we call library functions
	// that build the Laplace and mass matrices. All they need is a DoFHandler
	// object and a quadrature formula object that is to be used for numerical
	// integration. Note that in many respects these functions are better than
	// what we would usually do in application programs, for example because
	// they automatically parallelize building the matrices if multiple
	// processors are available in a machine. The matrices for solving linear
	// systems will be filled in the run() method because we need to re-apply
	// boundary conditions every time step.
	mass_matrix.reinit(sparsity_pattern);
	laplace_matrix.reinit(sparsity_pattern);
	matrix_u.reinit(sparsity_pattern);
	matrix_v.reinit(sparsity_pattern);

	MatrixCreator::create_mass_matrix(dof_handler, QGauss<dim>(3), mass_matrix);
	MatrixCreator::create_laplace_matrix(dof_handler, QGauss<dim>(3),
			laplace_matrix);

	// The rest of the function is spent on setting vector sizes to the
	// correct value. The final line closes the hanging node constraints
	// object. Since we work on a uniformly refined mesh, no constraints exist
	// or have been computed (i.e. there was no need to call
	// DoFTools::make_hanging_node_constraints as in other programs), but we
	// need a constraints object in one place further down below anyway.
	solution_u.reinit(dof_handler.n_dofs());
	solution_v.reinit(dof_handler.n_dofs());
	old_solution_u.reinit(dof_handler.n_dofs());
	old_solution_v.reinit(dof_handler.n_dofs());
	system_rhs.reinit(dof_handler.n_dofs());

	constraints.close();
}

template<int dim>
void WaveEquation<dim>::solve_u() {
	SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
	SolverCG<> cg(solver_control);

	cg.solve(matrix_u, solution_u, system_rhs, PreconditionIdentity());

	std::cout << "   u-equation: " << solver_control.last_step()
			<< " CG iterations." << std::endl;
}

template<int dim>
void WaveEquation<dim>::solve_v() {
	SolverControl solver_control(1000, 1e-8 * system_rhs.l2_norm());
	SolverCG<> cg(solver_control);

	cg.solve(matrix_v, solution_v, system_rhs, PreconditionIdentity());

	std::cout << "   v-equation: " << solver_control.last_step()
			<< " CG iterations." << std::endl;
}

template<int dim>
void WaveEquation<dim>::output_results() const {
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);
	data_out.add_data_vector(solution_u, "U");
	data_out.add_data_vector(solution_v, "V");

	data_out.build_patches();

	const std::string filename = "solution-"
			+ Utilities::int_to_string(timestep_number, 3) + ".vtu";
	std::ofstream output(filename.c_str());
	data_out.write_vtu(output);

	static std::vector<std::pair<double, std::string> > times_and_names;
	times_and_names.push_back(std::pair<double, std::string>(time, filename));
	std::ofstream pvd_output("solution.pvd");
	DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template<int dim>
void WaveEquation<dim>::run() {
	setup_system();

	VectorTools::project(dof_handler, constraints, QGauss<dim>(3),
			*initialValuesU, old_solution_u);
	VectorTools::project(dof_handler, constraints, QGauss<dim>(3),
			*initialValuesV, old_solution_v);

	// The next thing is to loop over all the time steps until we reach the
	// end time ($T=5$ in this case). In each time step, we first have to
	// solve for $U^n$, using the equation $(M^n + k^2\theta^2 A^n)U^n =$
	// $(M^{n,n-1} - k^2\theta(1-\theta) A^{n,n-1})U^{n-1} + kM^{n,n-1}V^{n-1}
	// +$ $k\theta \left[k \theta F^n + k(1-\theta) F^{n-1} \right]$. Note
	// that we use the same mesh for all time steps, so that $M^n=M^{n,n-1}=M$
	// and $A^n=A^{n,n-1}=A$. What we therefore have to do first is to add up
	// $MU^{n-1} - k^2\theta(1-\theta) AU^{n-1} + kMV^{n-1}$ and the forcing
	// terms, and put the result into the <code>system_rhs</code> vector. (For
	// these additions, we need a temporary vector that we declare before the
	// loop to avoid repeated memory allocations in each time step.)
	Vector<double> tmp(solution_u.size());
	Vector<double> forcing_terms(solution_u.size());

	for (; time <= 5; time += time_step, ++timestep_number) {
		std::cout << "Time step " << timestep_number << " at t=" << time
				<< std::endl;

		mass_matrix.vmult(system_rhs, old_solution_u);

		mass_matrix.vmult(tmp, old_solution_v);
		system_rhs.add(time_step, tmp);

		laplace_matrix.vmult(tmp, old_solution_u);
		system_rhs.add(-theta * (1 - theta) * time_step * time_step, tmp);

		rightHandSide->set_time(time);
		VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2),
				*rightHandSide, tmp);
		forcing_terms = tmp;
		forcing_terms *= theta * time_step;

		rightHandSide->set_time(time - time_step);
		VectorTools::create_right_hand_side(dof_handler, QGauss<dim>(2),
				*rightHandSide, tmp);

		forcing_terms.add((1 - theta) * time_step, tmp);

		system_rhs.add(theta * time_step, forcing_terms);

		// After so constructing the right hand side vector of the first
		// equation, all we have to do is apply the correct boundary
		// values. As for the right hand side, this is a space-time function
		// evaluated at a particular time, which we interpolate at boundary
		// nodes and then use the result to apply boundary values as we
		// usually do. The result is then handed off to the solve_u()
		// function:
		{
			boundaryValuesU->set_time(time);

			std::map<types::global_dof_index, double> boundary_values;
			VectorTools::interpolate_boundary_values(dof_handler, 0,
					*boundaryValuesU, boundary_values);

			// The matrix for solve_u() is the same in every time steps, so one
			// could think that it is enough to do this only once at the
			// beginning of the simulation. However, since we need to apply
			// boundary values to the linear system (which eliminate some matrix
			// rows and columns and give contributions to the right hand side),
			// we have to refill the matrix in every time steps before we
			// actually apply boundary data. The actual content is very simple:
			// it is the sum of the mass matrix and a weighted Laplace matrix:
			matrix_u.copy_from(mass_matrix);
			matrix_u.add(theta * theta * time_step * time_step, laplace_matrix);
			MatrixTools::apply_boundary_values(boundary_values, matrix_u,
					solution_u, system_rhs);
		}
		solve_u();

		// The second step, i.e. solving for $V^n$, works similarly, except
		// that this time the matrix on the left is the mass matrix (which we
		// copy again in order to be able to apply boundary conditions, and
		// the right hand side is $MV^{n-1} - k\left[ \theta A U^n +
		// (1-\theta) AU^{n-1}\right]$ plus forcing terms. %Boundary values
		// are applied in the same way as before, except that now we have to
		// use the BoundaryValuesV class:
		laplace_matrix.vmult(system_rhs, solution_u);
		system_rhs *= -theta * time_step;

		mass_matrix.vmult(tmp, old_solution_v);
		system_rhs += tmp;

		laplace_matrix.vmult(tmp, old_solution_u);
		system_rhs.add(-time_step * (1 - theta), tmp);

		system_rhs += forcing_terms;

		{
			boundaryValuesV->set_time(time);

			std::map<types::global_dof_index, double> boundary_values;
			VectorTools::interpolate_boundary_values(dof_handler, 0,
					*boundaryValuesV, boundary_values);
			matrix_v.copy_from(mass_matrix);
			MatrixTools::apply_boundary_values(boundary_values, matrix_v,
					solution_v, system_rhs);
		}
		solve_v();

		output_results();

		// compute $\left<V^n,MV^n\right>$ and $\left<U^n,AU^n\right>$
		std::cout << "   Total energy: "
				<< (mass_matrix.matrix_norm_square(solution_v)
						+ laplace_matrix.matrix_norm_square(solution_u)) / 2
				<< std::endl;

		old_solution_u = solution_u;
		old_solution_v = solution_v;
	}
}
}
