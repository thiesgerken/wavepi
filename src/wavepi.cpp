/*
 * wavepi_inverse.cpp
 *
 *  Created on: 01.07.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <forward/ConstantMesh.h>
#include <forward/DiscretizedFunction.h>
#include <forward/L2ProductRightHandSide.h>
#include <forward/L2RightHandSide.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/ConjugateGradients.h>
#include <inversion/GradientDescent.h>
#include <inversion/Landweber.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/REGINN.h>
#include <inversion/ToleranceChoice.h>
#include <inversion/RiederToleranceChoice.h>
#include <inversion/ConstantToleranceChoice.h>

#include <problems/L2QProblem.h>
#include <problems/L2CProblem.h>
#include <problems/L2NuProblem.h>
#include <problems/L2AProblem.h>

#include <util/Version.h>

#include <stddef.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include <boost/program_options.hpp>

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::problems;
using namespace wavepi::util;

namespace po = boost::program_options;

template<int dim>
class TestNu: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p[0] * this->get_time();
      }
};

template<int dim>
class TestF: public Function<dim> {
   public:
      TestF()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         if ((this->get_time() <= 0.5) && (p.distance(actor_position) < 0.4))
            return std::sin(this->get_time() * 2 * numbers::PI);
         else
            return 0.0;
      }
   private:
      static const Point<dim> actor_position;
};

template<> const Point<1> TestF<1>::actor_position = Point<1>(1.0);
template<> const Point<2> TestF<2>::actor_position = Point<2>(1.0, 0.5);
template<> const Point<3> TestF<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template<int dim>
double rho(const Point<dim> &p, double t);

template<>
double rho(const Point<1> &p, double t) {
   return p.distance(Point<1>(t - 3.0)) < 1.0 ? 1.5 : 1.0;
}

template<>
double rho(const Point<2> &p, double t) {
   return p.distance(Point<2>(t - 3.0, t - 2.0)) < 1.2 ? 1.5 : 1.0;
}

template<>
double rho(const Point<3> &p, double t) {
   return p.distance(Point<3>(t - 3.0, t - 2.0, 0.0)) < 1.2 ? 1.5 : 1.0;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / (rho(p, this->get_time()) * 1.0);
      }
};

template<int dim>
class TestA: public Function<dim> {
   public:
      TestA()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / rho(p, this->get_time());
      }
};

template<int dim>
class TestQ: public Function<dim> {
   public:
      TestQ()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p.distance(q_position) < 1.0 ? 10 * std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
      }

      static const Point<dim> q_position;
};

template<> const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<> const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<> const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

// possible problems
enum class ProblemType {
   L2Q = 1, L2A = 2, L2C = 3, L2Nu = 4
};

template<int dim>
class WavePI {
   private:
      static const std::string KEY_FE_DEGREE;
      static const std::string KEY_QUAD_ORDER;
      static const std::string KEY_PROBLEM_TYPE;
      static const std::string KEY_END_TIME;
      static const std::string KEY_EPSILON;
      static const std::string KEY_TAU;
      static const std::string KEY_INITIAL_REFINES;
      static const std::string KEY_INITIAL_TIME_STEPS;

      FE_Q<dim> fe;
      Quadrature<dim> quad;
      ProblemType problem_type;

      double end_time;
      double epsilon;
      double tau;
      int initial_refines;
      int initial_time_steps;

      Triangulation<dim> triangulation;
      std::shared_ptr<DoFHandler<dim>> dof_handler;
      std::shared_ptr<SpaceTimeMesh<dim>> mesh;

   public:
      static void declare_parameters(ParameterHandler &prm) {
         prm.declare_entry(KEY_FE_DEGREE, "1", Patterns::Integer(1, 4),
               "polynomial degree of finite elements");
         prm.declare_entry(KEY_QUAD_ORDER, "3", Patterns::Integer(1, 20),
               "order of quadrature (QGauss, exact in polynomials of degree ≤ 2n-1) ");
         prm.declare_entry(KEY_PROBLEM_TYPE, "L2A", Patterns::Selection("L2A|L2Q|L2Nu|L2C"),
               "parameter that is reconstructed, and which spaces are used");
         prm.declare_entry(KEY_END_TIME, "2", Patterns::Double(0), "time horizon T");
         prm.declare_entry(KEY_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");
         prm.declare_entry(KEY_TAU, "2", Patterns::Double(0), "parameter τ for discrepancy principle");
         prm.declare_entry(KEY_INITIAL_REFINES, "3", Patterns::Integer(0),
               "refines of the (initial) spatial grid");
         prm.declare_entry(KEY_INITIAL_TIME_STEPS, "64", Patterns::Integer(2),
               "(initial) number of time steps");

      }

      WavePI(ParameterHandler &prm)
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

      void initialize_mesh() {
         LogStream::Prefix p("initial_mesh");

         double dt = end_time / (initial_time_steps-1);
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
         deallog << "Average cell diameter: " << 10.0 * sqrt(dim) / pow(triangulation.n_active_cells(), 1.0 / dim)
               << std::endl;
         deallog << "dt: " << dt << std::endl;

         mesh = std::make_shared<ConstantMesh<dim>>(times, dof_handler, quad);

         if (dim == 1)
            mesh->set_boundary_ids(std::vector<types::boundary_id> { 0, 1 });
      }

      void run() {
         initialize_mesh();

         deallog.push("init");

         WaveEquation<dim> wave_eq(mesh, dof_handler, quad);

         wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
         wave_eq.set_param_a(std::make_shared<TestA<dim>>());
         wave_eq.set_param_c(std::make_shared<TestC<dim>>());
         wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
         wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

         using Param = DiscretizedFunction<dim>;
         using Sol = DiscretizedFunction<dim>;

         std::shared_ptr<NonlinearProblem<Param, Sol>> problem;
         std::shared_ptr<Function<dim>> param_exact_cont;
         std::shared_ptr<Param> param_exact;
         Param initialGuess(mesh, dof_handler);

         switch (problem_type) {
            case ProblemType::L2Q:
               /* Reconstruct TestQ */
               param_exact_cont = std::make_shared<TestQ<dim>>();
               param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
               wave_eq.set_param_q(param_exact);
               problem = std::make_shared<L2QProblem<dim>>(wave_eq);
               initialGuess = 0;
               break;
            case ProblemType::L2C:
               /* Reconstruct TestC */
               param_exact_cont = std::make_shared<TestC<dim>>();
               param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
               wave_eq.set_param_c(param_exact);
               problem = std::make_shared<L2CProblem<dim>>(wave_eq);
               initialGuess = 2;
               break;
            case ProblemType::L2Nu:
               /* Reconstruct TestNu */
               param_exact_cont = std::make_shared<TestNu<dim>>();
               param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
               wave_eq.set_param_nu(param_exact);
               problem = std::make_shared<L2NuProblem<dim>>(wave_eq);
               initialGuess = 0;
               break;
            case ProblemType::L2A:
               /* Reconstruct TestA */
               param_exact_cont = std::make_shared<TestA<dim>>();
               param_exact = std::make_shared<Param>(mesh, dof_handler, *param_exact_cont.get());
               wave_eq.set_param_a(param_exact);
               problem = std::make_shared<L2AProblem<dim>>(wave_eq);
               initialGuess = 2;
               break;
            default:
               AssertThrow(false, ExcInternalError())
         }

         deallog.push("generate_data");

         auto data_exact = wave_eq.run();
         data_exact.throw_away_derivative();
         data_exact.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
         double data_exact_norm = data_exact.norm();

         // in itself not wrong, but makes relative errors and noise levels meaningless.
         AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

         auto data = DiscretizedFunction<dim>::noise(data_exact, epsilon * data_exact_norm);
         data.add(1.0, data_exact);

         deallog.pop();
         deallog.pop();

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

         reginn.invert(data, tau * epsilon * data_exact_norm, param_exact);
      }
};

template<int dim> const std::string WavePI<dim>::KEY_FE_DEGREE = "finite element degree";
template<int dim> const std::string WavePI<dim>::KEY_QUAD_ORDER = "quadrature order";
template<int dim> const std::string WavePI<dim>::KEY_PROBLEM_TYPE = "problem";
template<int dim> const std::string WavePI<dim>::KEY_EPSILON = "epsilon";
template<int dim> const std::string WavePI<dim>::KEY_END_TIME = "end time";
template<int dim> const std::string WavePI<dim>::KEY_TAU = "tau";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_REFINES = "initial refines";
template<int dim> const std::string WavePI<dim>::KEY_INITIAL_TIME_STEPS = "initial time steps";

int main(int argc, char * argv[]) {
   try {
      int log_file_depth;
      int log_console_depth;

      po::options_description desc(Version::get_identification() + "\nsupported options");

      desc.add_options()("help,h", "produce help message and exit");
      desc.add_options()("version", "print version information and exit");
      desc.add_options()("make-config",
            "generate config file with default values (unless [config] is specified) and exit");
      desc.add_options()("config,c", po::value<std::string>(), "read config from this file");
      desc.add_options()("log,l", po::value<std::string>(), "external log file");
      desc.add_options()("log-file-depth", po::value<int>(&log_file_depth)->default_value(100),
            "log depth that goes to [log]");
      desc.add_options()("log-console-depth", po::value<int>(&log_console_depth)->default_value(2),
            "log depth that goes to stdout");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
         std::cout << desc << "\n";
         return 1;
      }

      if (vm.count("version")) {
         std::cout << Version::get_identification() << std::endl;
         std::cout << Version::get_infos() << std::endl;
         return 1;
      }

      std::ofstream logout;
      if (vm.count("log-file")) {
         logout = std::ofstream(vm["log-file"].as<std::string>());
         deallog.attach(logout);
         deallog.depth_file(log_file_depth);
      }

      ParameterHandler prm;
      prm.declare_entry("dimension", "2", Patterns::Integer(1, 3), "problem dimension");
      WavePI<2>::declare_parameters(prm);

      if (vm.count("config"))
         prm.parse_input(vm["config"].as<std::string>());
      else
         AssertThrow(vm.count("make-config"),
               ExcMessage("No config file specified. Use `wavepi --make-config` to create one."));

      if (vm.count("make-config")) {
         prm.print_parameters(std::cout, ParameterHandler::Text);
         return 1;
      }

      deallog.depth_console(log_console_depth);
      deallog.precision(3);
      deallog.pop();
      deallog << Version::get_identification() << std::endl;
      // deallog << Version::get_infos() << std::endl;
      // deallog.log_execution_time(true);

      prm.log_parameters(deallog);

      int dim = prm.get_integer("dimension");

      if (dim == 1) {
         WavePI<1> wavepi(prm);
         wavepi.run();
      } else if (dim == 2) {
         WavePI<2> wavepi(prm);
         wavepi.run();
      } else {
         WavePI<3> wavepi(prm);
         wavepi.run();
      }

      // deallog.timestamp();
   } catch (std::exception &exc) {
      std::cerr << "Exception on processing: " << exc.what();
      return 1;
   } catch (...) {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
   }

   return 0;
}
