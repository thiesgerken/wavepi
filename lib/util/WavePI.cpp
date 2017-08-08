/*
 * WavePI.cpp
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>

#include <forward/ConstantMesh.h>
#include <forward/L2RightHandSide.h>

#include <inversion/ConjugateGradients.h>
#include <inversion/ConstantToleranceChoice.h>
#include <inversion/GradientDescent.h>
#include <inversion/InversionProgress.h>
#include <inversion/Landweber.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/REGINN.h>
#include <inversion/RiederToleranceChoice.h>

#include <problems/L2AProblem.h>
#include <problems/L2CProblem.h>
#include <problems/L2NuProblem.h>
#include <problems/L2QProblem.h>

#include <util/WavePI.h>

#include <stddef.h>

#include <cmath>
#include <iostream>
#include <vector>

namespace wavepi {
namespace util {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::problems;

template<int dim> const std::string WavePI<dim>::KEY_FE_DEGREE = "finite element degree";
template<int dim> const std::string WavePI<dim>::KEY_QUAD_ORDER = "quadrature order";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_TYPE = "problem";
template<int dim> const std::string WavePI<dim>::KEY_EPSILON = "epsilon";
template<int dim> const std::string WavePI<dim>::KEY_END_TIME = "end time";
template<int dim> const std::string WavePI<dim>::KEY_TAU = "tau";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_REFINES = "initial refines";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_TIME_STEPS = "initial time steps";

template<int dim> void WavePI<dim>::declare_parameters(ParameterHandler &prm) {
   prm.declare_entry(KEY_FE_DEGREE, "1", Patterns::Integer(1, 4), "polynomial degree of finite elements");
   prm.declare_entry(KEY_QUAD_ORDER, "3", Patterns::Integer(1, 20),
         "order of quadrature (QGauss, exact in polynomials of degree ≤ 2n-1) ");
   prm.declare_entry(KEY_PROBLEM_TYPE, "L2A", Patterns::Selection("L2A|L2Q|L2Nu|L2C"),
         "parameter that is reconstructed, and which spaces are used");
   prm.declare_entry(KEY_END_TIME, "2", Patterns::Double(0), "time horizon T");
   prm.declare_entry(KEY_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");
   prm.declare_entry(KEY_TAU, "2", Patterns::Double(0), "parameter τ for discrepancy principle");
   prm.declare_entry(KEY_INITIAL_REFINES, "3", Patterns::Integer(0), "refines of the (initial) spatial grid");
   prm.declare_entry(KEY_INITIAL_TIME_STEPS, "64", Patterns::Integer(2), "(initial) number of time steps");

}

template<int dim> WavePI<dim>::WavePI(ParameterHandler &prm)
      : fe(prm.get_integer(KEY_FE_DEGREE)), quad(prm.get_integer(KEY_QUAD_ORDER)) {
   std::string problem = prm.get(KEY_PROBLEM_TYPE);

   if (problem == "L2A")
      problem_type = ProblemType::L2A;
   else if (problem == "L2Q")
      problem_type = ProblemType::L2Q;
   else if (problem == "L2Nu")
      problem_type = ProblemType::L2Nu;
   else if (problem == "L2C")
      problem_type = ProblemType::L2C;
   else
      AssertThrow(false, ExcInternalError());

   end_time = prm.get_double(KEY_END_TIME);
   epsilon = prm.get_double(KEY_EPSILON);
   tau = prm.get_double(KEY_TAU);
   initial_refines = prm.get_integer(KEY_INITIAL_REFINES);
   initial_time_steps = prm.get_integer(KEY_INITIAL_TIME_STEPS);
}

template<int dim> void WavePI<dim>::initialize_mesh() {
   LogStream::Prefix p("initialize_mesh");

   double dt = end_time / (initial_time_steps - 1);
   std::vector<double> times;

   for (size_t i = 0; i * dt <= end_time; i++)
      times.push_back(i * dt);

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(triangulation, -5, 5);
   triangulation.refine_global(initial_refines);

   dof_handler = std::make_shared<DoFHandler<dim>>();
   dof_handler->initialize(triangulation, fe);

   deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler->n_dofs() << std::endl;
   deallog << "Average cell diameter: "
         << 10.0 * sqrt((double) dim) / pow(triangulation.n_active_cells(), 1.0 / dim) << std::endl;
   deallog << "dt: " << dt << std::endl;

   mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);

   if (dim == 1)
      mesh->set_boundary_ids(std::vector<types::boundary_id> { 0, 1 });
}

template<int dim> void WavePI<dim>::initialize_problem() {
   LogStream::Prefix p("initialize_problem");

   wave_eq = std::make_shared<WaveEquation<dim>>(mesh, dof_handler, quad);
   wave_eq->set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
   wave_eq->set_param_a(std::make_shared<TestA<dim>>());
   wave_eq->set_param_c(std::make_shared<TestC<dim>>());
   wave_eq->set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq->set_param_nu(std::make_shared<TestNu<dim>>());

   initialGuess = std::make_shared<Param>(mesh, dof_handler);

   switch (problem_type) {
      case ProblemType::L2Q:
         /* Reconstruct TestQ */
         param_exact_cont = std::make_shared<TestQ<dim>>();
         param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
         wave_eq->set_param_q(param_exact);
         problem = std::make_shared<L2QProblem<dim>>(*wave_eq);
         *initialGuess = 0;
         break;
      case ProblemType::L2C:
         /* Reconstruct TestC */
         param_exact_cont = std::make_shared<TestC<dim>>();
         param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
         wave_eq->set_param_c(param_exact);
         problem = std::make_shared<L2CProblem<dim>>(*wave_eq);
         *initialGuess = 2;
         break;
      case ProblemType::L2Nu:
         /* Reconstruct TestNu */
         param_exact_cont = std::make_shared<TestNu<dim>>();
         param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
         wave_eq->set_param_nu(param_exact);
         problem = std::make_shared<L2NuProblem<dim>>(*wave_eq);
         *initialGuess = 0;
         break;
      case ProblemType::L2A:
         /* Reconstruct TestA */
         param_exact_cont = std::make_shared<TestA<dim>>();
         param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
         wave_eq->set_param_a(param_exact);
         problem = std::make_shared<L2AProblem<dim>>(*wave_eq);
         *initialGuess = 2;
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }
}

template<int dim> void WavePI<dim>::generate_data() {
   LogStream::Prefix p("generate_data");

   Sol data_exact = wave_eq->run();
   data_exact.throw_away_derivative();
   data_exact.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   double data_exact_norm = data_exact.norm();

   // in itself not wrong, but makes relative errors and noise levels meaningless.
   AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

   data = std::make_shared<Sol>(DiscretizedFunction<dim>::noise(data_exact, epsilon * data_exact_norm));
   data->add(1.0, data_exact);
}

template<int dim> void WavePI<dim>::run() {
   initialize_mesh();
   initialize_problem();
   generate_data();

   auto linear_solver = std::make_shared<ConjugateGradients<Param, Sol>>();
   linear_solver->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol>>("k"));
   linear_solver->add_listener(
         std::make_shared<CtrlCProgressListener<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>());

   auto tol_choice = std::make_shared<RiederToleranceChoice>(0.7, 0.95, 0.9, 1.0);
   REGINN<Param, Sol> reginn(problem, linear_solver, tol_choice, initialGuess);
   reginn.add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol>>("i"));
   reginn.add_listener(std::make_shared<OutputProgressListener<dim>>(10));
   reginn.add_listener(
         std::make_shared<CtrlCProgressListener<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>());

   reginn.invert(*data, tau * epsilon * data->norm(), param_exact);
}

template class WavePI<1> ;
template class WavePI<2> ;
template class WavePI<3> ;

} /* namespace util */
} /* namespace wavepi */
