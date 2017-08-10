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

#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/Regularization.h>
#include <inversion/REGINN.h>

#include <problems/L2AProblem.h>
#include <problems/L2CProblem.h>
#include <problems/L2NuProblem.h>
#include <problems/L2QProblem.h>

#include <util/WavePI.h>
#include <util/GridTools.h>

#include <stddef.h>
#include <tgmath.h>
#include <iostream>
#include <vector>

namespace wavepi {
namespace util {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::problems;
using namespace wavepi::util;

template<int dim> const std::string WavePI<dim>::KEY_FE_DEGREE = "finite element degree";
template<int dim> const std::string WavePI<dim>::KEY_QUAD_ORDER = "quadrature order";
template<int dim> const std::string WavePI<dim>::KEY_END_TIME = "end time";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_REFINES = "initial refines";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_TIME_STEPS = "initial time steps";

template<int dim> const std::string WavePI<dim>::KEY_INVERSION = "inversion";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_PROBLEM_TYPE = "problem";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_EPSILON = "epsilon";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_TAU = "tau";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_METHOD = "method";

template<int dim> void WavePI<dim>::declare_parameters(ParameterHandler &prm) {
   prm.declare_entry(KEY_FE_DEGREE, "1", Patterns::Integer(1, 4), "polynomial degree of finite elements");
   prm.declare_entry(KEY_QUAD_ORDER, "3", Patterns::Integer(1, 20),
         "order of quadrature (QGauss, exact in polynomials of degree ≤ 2n-1, use at least finite element degree + 1) ");
   prm.declare_entry(KEY_END_TIME, "2", Patterns::Double(0), "time horizon T");
   prm.declare_entry(KEY_INITIAL_REFINES, "3", Patterns::Integer(0), "refines of the (initial) spatial grid");
   prm.declare_entry(KEY_INITIAL_TIME_STEPS, "64", Patterns::Integer(2), "(initial) number of time steps");

   prm.enter_subsection(KEY_INVERSION);
   {
      prm.declare_entry(KEY_INVERSION_METHOD, "REGINN", Patterns::Selection("REGINN|NonlinearLandweber"),
            "solver for the inverse problem");
      prm.declare_entry(KEY_INVERSION_PROBLEM_TYPE, "L2A", Patterns::Selection("L2A|L2Q|L2Nu|L2C"),
            "parameter that is reconstructed, and which spaces are used");
      prm.declare_entry(KEY_INVERSION_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");
      prm.declare_entry(KEY_INVERSION_TAU, "2", Patterns::Double(0), "parameter τ for discrepancy principle");

      REGINN<DiscretizedFunction<dim>, DiscretizedFunction<dim>>::declare_parameters(prm);
      NonlinearLandweber<DiscretizedFunction<dim>, DiscretizedFunction<dim>>::declare_parameters(prm);
   }
   prm.leave_subsection();

   OutputProgressListener<dim>::declare_parameters(prm);
}

template<int dim> WavePI<dim>::WavePI(std::shared_ptr<ParameterHandler> prm)
      : prm(prm), fe(prm->get_integer(KEY_FE_DEGREE)), quad(prm->get_integer(KEY_QUAD_ORDER)) {

   end_time = prm->get_double(KEY_END_TIME);
   initial_refines = prm->get_integer(KEY_INITIAL_REFINES);
   initial_time_steps = prm->get_integer(KEY_INITIAL_TIME_STEPS);

   prm->enter_subsection(KEY_INVERSION);
   {
      epsilon = prm->get_double(KEY_INVERSION_EPSILON);
      tau = prm->get_double(KEY_INVERSION_TAU);

      std::string problem = prm->get(KEY_INVERSION_PROBLEM_TYPE);

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

      std::string smethod = prm->get(KEY_INVERSION_METHOD);

      if (smethod == "REGINN")
         method = NonlinearMethod::REGINN;
      else if (smethod == "NonlinearLandweber")
         method = NonlinearMethod::NonlinearLandweber;
      else
         AssertThrow(false, ExcInternalError());

   }
   prm->leave_subsection();
}

template<int dim> void WavePI<dim>::initialize_mesh() {
   LogStream::Prefix p("initialize_mesh");

   double dt = end_time / (initial_time_steps - 1);
   std::vector<double> times;

   for (size_t i = 0; i * dt <= end_time; i++)
      times.push_back(i * dt);

   auto triangulation = std::make_shared<Triangulation<dim>>();

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(*triangulation, -5, 5);
   GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(initial_refines);

   mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

   deallog << "Number of active cells: " << triangulation->n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom in spatial mesh: " << mesh->get_dof_handler(0)->n_dofs()
         << std::endl;
   deallog << "Average cell diameter: "
         << 10.0 * sqrt((double ) dim) / pow(triangulation->n_active_cells(), 1.0 / dim) << std::endl;
   deallog << "dt: " << dt << std::endl;
}

template<int dim> void WavePI<dim>::initialize_problem() {
   LogStream::Prefix p("initialize_problem");

   wave_eq = std::make_shared<WaveEquation<dim>>(mesh);
   wave_eq->set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
   wave_eq->set_param_a(std::make_shared<TestA<dim>>());
   wave_eq->set_param_c(std::make_shared<TestC<dim>>());
   wave_eq->set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq->set_param_nu(std::make_shared<TestNu<dim>>());

   initialGuess = std::make_shared<Param>(mesh);

   switch (problem_type) {
      case ProblemType::L2Q:
         /* Reconstruct TestQ */
         param_exact_cont = std::make_shared<TestQ<dim>>();
         param_exact = std::make_shared<Param>(mesh, *param_exact_cont.get());
         wave_eq->set_param_q(param_exact);
         problem = std::make_shared<L2QProblem<dim>>(*wave_eq);
         *initialGuess = 0;
         break;
      case ProblemType::L2C:
         /* Reconstruct TestC */
         param_exact_cont = std::make_shared<TestC<dim>>();
         param_exact = std::make_shared<Param>(mesh, *param_exact_cont.get());
         wave_eq->set_param_c(param_exact);
         problem = std::make_shared<L2CProblem<dim>>(*wave_eq);
         *initialGuess = 2;
         break;
      case ProblemType::L2Nu:
         /* Reconstruct TestNu */
         param_exact_cont = std::make_shared<TestNu<dim>>();
         param_exact = std::make_shared<Param>(mesh, *param_exact_cont.get());
         wave_eq->set_param_nu(param_exact);
         problem = std::make_shared<L2NuProblem<dim>>(*wave_eq);
         *initialGuess = 0;
         break;
      case ProblemType::L2A:
         /* Reconstruct TestA */
         param_exact_cont = std::make_shared<TestA<dim>>();
         param_exact = std::make_shared<Param>(mesh, *param_exact_cont.get());
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

   std::shared_ptr<Regularization<Param, Sol>> regularization;

   prm->enter_subsection(KEY_INVERSION);
   if (method == NonlinearMethod::REGINN)
      regularization = std::make_shared<REGINN<Param, Sol> >(problem, initialGuess, *prm);
   else if (method == NonlinearMethod::NonlinearLandweber)
      regularization = std::make_shared<NonlinearLandweber<Param, Sol> >(problem, initialGuess, *prm);
   else
      AssertThrow(false, ExcInternalError());
   prm->leave_subsection();

   regularization->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol>>("i"));
   regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Sol>>());
   regularization->add_listener(std::make_shared<OutputProgressListener<dim>>(*prm));

   regularization->invert(*data, tau * epsilon * data->norm(), param_exact);
}

template class WavePI<1> ;
template class WavePI<2> ;
template class WavePI<3> ;

} /* namespace util */
} /* namespace wavepi */
