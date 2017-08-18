/*
 * WavePI.cpp
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

#include <forward/ConstantMesh.h>
#include <forward/AdaptiveMesh.h>
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

template<int dim> const std::string WavePI<dim>::KEY_GENERAL = "general";
template<int dim> const std::string WavePI<dim>::KEY_DIMENSION = "dimension";
template<int dim> const std::string WavePI<dim>::KEY_FE_DEGREE = "finite element degree";
template<int dim> const std::string WavePI<dim>::KEY_QUAD_ORDER = "quadrature order";

template<int dim> const std::string WavePI<dim>::KEY_MESH = "mesh";
template<int dim> const std::string WavePI<dim>::KEY_END_TIME = "end time";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_REFINES = "initial refines";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_TIME_STEPS = "initial time steps";

template<int dim> const std::string WavePI<dim>::KEY_PROBLEM = "problem";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_TYPE = "type";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_EPSILON = "epsilon";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_CONSTANTS = "constants";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_NUM_RHS = "number of right hand sides";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_RHS = "right hand side";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_GUESS = "initial guess";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_PARAM_A = "parameter a";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_PARAM_Q = "parameter q";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_PARAM_C = "parameter c";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_PARAM_NU = "parameter nu";

template<int dim> const std::string WavePI<dim>::KEY_INVERSION = "inversion";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_TAU = "tau";
template<int dim> const std::string WavePI<dim>::KEY_INVERSION_METHOD = "method";

template<int dim> void WavePI<dim>::declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection(KEY_GENERAL);
   {
      prm.declare_entry(KEY_FE_DEGREE, "1", Patterns::Integer(1, 4), "polynomial degree of finite elements");
      prm.declare_entry(KEY_QUAD_ORDER, "3", Patterns::Integer(1, 20),
            "order of quadrature (QGauss, exact in polynomials of degree ≤ 2n-1, use at least finite element degree + 1) ");
   }
   prm.leave_subsection();

   prm.enter_subsection(KEY_MESH);
   {
      prm.declare_entry(KEY_END_TIME, "6", Patterns::Double(0), "time horizon T");
      prm.declare_entry(KEY_INITIAL_REFINES, "4", Patterns::Integer(0),
            "refines of the (initial) spatial grid");
      prm.declare_entry(KEY_INITIAL_TIME_STEPS, "256", Patterns::Integer(2),
            "(initial) number of time steps");
   }
   prm.leave_subsection();

   prm.enter_subsection(KEY_PROBLEM);
   {
      prm.declare_entry(KEY_PROBLEM_TYPE, "L2A", Patterns::Selection("L2A|L2Q|L2Nu|L2C"),
            "parameter that is reconstructed, and which spaces are used");
      prm.declare_entry(KEY_PROBLEM_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");

      prm.declare_entry(KEY_PROBLEM_CONSTANTS, "", Patterns::Anything(),
            "constants for the function declarations, in the form `var1=value1, var2=value2, ...'.");

      // prm.declare_entry(KEY_PROBLEM_NUM_RHS, "1", Patterns::Integer(1), "number of right hand sides");
      // TODO: only works in 2d now without changing the config!
      prm.declare_entry(KEY_PROBLEM_RHS, "if(x*x+y*y < 0.2, sin(t), 0.0)", Patterns::Anything(),
            "right hand side");

      prm.declare_entry(KEY_PROBLEM_GUESS, "0.5", Patterns::Anything(), "initial guess");

      prm.declare_entry(KEY_PROBLEM_PARAM_A, "1.0", Patterns::Anything(), "parameter a");
      prm.declare_entry(KEY_PROBLEM_PARAM_Q, "0.0", Patterns::Anything(), "parameter q");
      prm.declare_entry(KEY_PROBLEM_PARAM_C, "2.0", Patterns::Anything(), "parameter c");
      prm.declare_entry(KEY_PROBLEM_PARAM_NU, "0.0", Patterns::Anything(), "parameter ν");
   }
   prm.leave_subsection();

   prm.enter_subsection(KEY_INVERSION);
   {
      prm.declare_entry(KEY_INVERSION_METHOD, "REGINN", Patterns::Selection("REGINN|NonlinearLandweber"),
            "solver for the inverse problem");
      prm.declare_entry(KEY_INVERSION_TAU, "2", Patterns::Double(0), "parameter τ for discrepancy principle");

      REGINN<DiscretizedFunction<dim>, DiscretizedFunction<dim>, Function<dim>>::declare_parameters(prm);
      NonlinearLandweber<DiscretizedFunction<dim>, DiscretizedFunction<dim>, Function<dim>>::declare_parameters(
            prm);
   }
   prm.leave_subsection();

   WaveEquationBase<dim>::declare_parameters(prm);
   OutputProgressListener<dim>::declare_parameters(prm);
}

template<int dim> WavePI<dim>::WavePI(std::shared_ptr<ParameterHandler> prm)
      : prm(prm) {
   prm->enter_subsection(KEY_GENERAL);
   {
      fe_degree = prm->get_integer(KEY_FE_DEGREE);
      quad_order = prm->get_integer(KEY_QUAD_ORDER);
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_MESH);
   {
      end_time = prm->get_double(KEY_END_TIME);
      initial_refines = prm->get_integer(KEY_INITIAL_REFINES);
      initial_time_steps = prm->get_integer(KEY_INITIAL_TIME_STEPS);
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_PROBLEM);
   {
      epsilon = prm->get_double(KEY_PROBLEM_EPSILON);

      std::string problem = prm->get(KEY_PROBLEM_TYPE);

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

      std::string constants_list = prm->get(KEY_PROBLEM_CONSTANTS);
      std::vector<std::string> const_listed = Utilities::split_string_list(constants_list, ',');

      std::map<std::string, double> constants;
      for (size_t i = 0; i < const_listed.size(); ++i) {
         std::vector<std::string> this_c = Utilities::split_string_list(const_listed[i], '=');
         AssertThrow(this_c.size() == 2, ExcMessage("Invalid format"));
         double tmp;
         AssertThrow(std::sscanf(this_c[1].c_str(), "%lf", &tmp), ExcMessage("Double number?"));
         constants[this_c[0]] = tmp;
      }

      rhs = std::make_shared<FunctionParser<dim>>();
      rhs->initialize(FunctionParser<dim>::default_variable_names() + ",t", prm->get(KEY_PROBLEM_RHS),
            constants, true);

      initial_guess = std::make_shared<FunctionParser<dim>>();
      initial_guess->initialize(FunctionParser<dim>::default_variable_names() + ",t",
            prm->get(KEY_PROBLEM_GUESS), constants, true);

      param_a = std::make_shared<FunctionParser<dim>>();
      param_a->initialize(FunctionParser<dim>::default_variable_names() + ",t", prm->get(KEY_PROBLEM_PARAM_A),
            constants, true);

      param_nu = std::make_shared<FunctionParser<dim>>();
      param_nu->initialize(FunctionParser<dim>::default_variable_names() + ",t",
            prm->get(KEY_PROBLEM_PARAM_NU), constants, true);

      param_c = std::make_shared<FunctionParser<dim>>();
      param_c->initialize(FunctionParser<dim>::default_variable_names() + ",t", prm->get(KEY_PROBLEM_PARAM_C),
            constants, true);

      param_q = std::make_shared<FunctionParser<dim>>();
      param_q->initialize(FunctionParser<dim>::default_variable_names() + ",t", prm->get(KEY_PROBLEM_PARAM_Q),
            constants, true);
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_INVERSION);
   {
      tau = prm->get_double(KEY_INVERSION_TAU);

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

//   mesh = std::make_shared<AdaptiveMesh<dim>>(times, FE_Q<dim>(fe_degree), QGauss<dim>(quad_order),
//         triangulation);

   auto a_mesh = std::make_shared<AdaptiveMesh<dim>>(times, FE_Q<dim>(fe_degree), QGauss<dim>(quad_order),
         triangulation);

   mesh = a_mesh;

   // TEST: flag some cells for refinement, and refine them in some step
   {
      LogStream::Prefix pd("TEST");
      for (auto cell : triangulation->active_cell_iterators())
         if (cell->center()[1] > 0)
            cell->set_refine_flag();

      std::vector<bool> ref;
      std::vector<bool> coa;

      triangulation->save_refine_flags(ref);
      triangulation->save_coarsen_flags(coa);

      std::vector<Patch> patches = a_mesh->get_forward_patches();
      patches[initial_time_steps / 2].emplace_back(ref, coa);

      a_mesh->set_forward_patches(patches);

      mesh->get_dof_handler(0);
   }

   deallog << "Number of active cells: " << triangulation->n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom in the first spatial mesh: " << mesh->get_dof_handler(0)->n_dofs()
         << std::endl;
   deallog << "cell diameters: minimal = " << dealii::GridTools::minimal_cell_diameter(*triangulation)
         << std::endl;
   deallog << "                average = "
         << 10.0 * sqrt((double ) dim) / pow(triangulation->n_active_cells(), 1.0 / dim) << std::endl;
   deallog << "                maximal = " << dealii::GridTools::maximal_cell_diameter(*triangulation)
         << std::endl;
   deallog << "dt: " << dt << std::endl;
}

template<int dim> void WavePI<dim>::initialize_problem() {
   LogStream::Prefix p("initialize_problem");

   wave_eq = std::make_shared<WaveEquation<dim>>(mesh);

   wave_eq->set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(rhs));
   wave_eq->set_param_a(param_a);
   wave_eq->set_param_c(param_c);
   wave_eq->set_param_q(param_q);
   wave_eq->set_param_nu(param_nu);
   wave_eq->get_parameters(*prm);

   switch (problem_type) {
      case ProblemType::L2Q:
         /* Reconstruct TestQ */
         param_exact = wave_eq->get_param_q();
         problem = std::make_shared<L2QProblem<dim>>(*wave_eq);
         break;
      case ProblemType::L2C:
         /* Reconstruct TestC */
         param_exact = wave_eq->get_param_c();
         problem = std::make_shared<L2CProblem<dim>>(*wave_eq);
         break;
      case ProblemType::L2Nu:
         /* Reconstruct TestNu */
         param_exact = wave_eq->get_param_nu();
         problem = std::make_shared<L2NuProblem<dim>>(*wave_eq);
         break;
      case ProblemType::L2A:
         /* Reconstruct TestA */
         param_exact = wave_eq->get_param_a();
         problem = std::make_shared<L2AProblem<dim>>(*wave_eq);
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }
}

template<int dim> void WavePI<dim>::generate_data() {
   LogStream::Prefix p("generate_data");
   LogStream::Prefix pp(" ");

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

   std::shared_ptr<Regularization<Param, Sol, Exact>> regularization;

   deallog.push(" ");
   // discretize initial guess
   auto initial_guess_discretized = std::make_shared<Param>(mesh, *initial_guess);
   deallog.pop();

   prm->enter_subsection(KEY_INVERSION);
   if (method == NonlinearMethod::REGINN)
      regularization = std::make_shared<REGINN<Param, Sol, Exact> >(problem, initial_guess_discretized, *prm);
   else if (method == NonlinearMethod::NonlinearLandweber)
      regularization = std::make_shared<NonlinearLandweber<Param, Sol, Exact> >(problem,
            initial_guess_discretized, *prm);
   else
      AssertThrow(false, ExcInternalError());
   prm->leave_subsection();

   regularization->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol, Exact>>("i"));
   regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Sol, Exact>>());
   regularization->add_listener(std::make_shared<OutputProgressListener<dim>>(*prm));

   prm->log_parameters(deallog);

   regularization->invert(*data, tau * epsilon * data->norm(), param_exact);
}

template class WavePI<1> ;
template class WavePI<2> ;
template class WavePI<3> ;

} /* namespace util */
} /* namespace wavepi */
