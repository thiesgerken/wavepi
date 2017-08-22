/*
 * SettingsManager.cpp
 *
 *  Created on: 22.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <forward/DiscretizedFunction.h>
#include <forward/Measure.h>
#include <forward/WaveEquationBase.h>

#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/REGINN.h>

#include <SettingsManager.h>

#include <tgmath.h>
#include <cstdio>
#include <utility>

namespace wavepi {
using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::util;

const std::string SettingsManager::KEY_GENERAL = "general";
const std::string SettingsManager::KEY_GENERAL_DIMENSION = "dimension";
const std::string SettingsManager::KEY_GENERAL_FE_DEGREE = "finite element degree";
const std::string SettingsManager::KEY_GENERAL_QUAD_ORDER = "quadrature order";

const std::string SettingsManager::KEY_LOG = "log";
const std::string SettingsManager::KEY_LOG_FILE = "file";
const std::string SettingsManager::KEY_LOG_FILE_DEPTH = "file depth";
const std::string SettingsManager::KEY_LOG_CONSOLE_DEPTH = "console depth";

const std::string SettingsManager::KEY_MESH = "mesh";
const std::string SettingsManager::KEY_MESH_END_TIME = "end time";
const std::string SettingsManager::KEY_MESH_INITIAL_REFINES = "initial refines";
const std::string SettingsManager::KEY_MESH_INITIAL_TIME_STEPS = "initial time steps";
const std::string SettingsManager::KEY_MESH_SHAPE = "shape";
const std::string SettingsManager::KEY_MESH_SHAPE_GENERATOR = "generator name";
const std::string SettingsManager::KEY_MESH_SHAPE_OPTIONS = "options";

const std::string SettingsManager::KEY_PROBLEM = "problem";
const std::string SettingsManager::KEY_PROBLEM_TYPE = "type";
const std::string SettingsManager::KEY_PROBLEM_EPSILON = "epsilon";
const std::string SettingsManager::KEY_PROBLEM_CONSTANTS = "constants";
const std::string SettingsManager::KEY_PROBLEM_GUESS = "initial guess";
const std::string SettingsManager::KEY_PROBLEM_PARAM_A = "parameter a";
const std::string SettingsManager::KEY_PROBLEM_PARAM_Q = "parameter q";
const std::string SettingsManager::KEY_PROBLEM_PARAM_C = "parameter c";
const std::string SettingsManager::KEY_PROBLEM_PARAM_NU = "parameter nu";

const std::string SettingsManager::KEY_PROBLEM_DATA = "measurement configurations";
const std::string SettingsManager::KEY_PROBLEM_DATA_COUNT = "number of configurations";
const std::string SettingsManager::KEY_PROBLEM_DATA_I = "configuration "; // + number
const std::string SettingsManager::KEY_PROBLEM_DATA_I_RHS = "right hand side";
const std::string SettingsManager::KEY_PROBLEM_DATA_I_MEASURE = "measure";

const std::string SettingsManager::KEY_INVERSION = "inversion";
const std::string SettingsManager::KEY_INVERSION_TAU = "tau";
const std::string SettingsManager::KEY_INVERSION_METHOD = "method";

void SettingsManager::declare_parameters(std::shared_ptr<ParameterHandler> prm) {
   prm->enter_subsection(KEY_GENERAL);
   {
      prm->declare_entry(KEY_GENERAL_DIMENSION, "2", Patterns::Integer(1, 3), "problem dimension");
      prm->declare_entry(KEY_GENERAL_FE_DEGREE, "1", Patterns::Integer(1, 4),
            "polynomial degree of finite elements");
      prm->declare_entry(KEY_GENERAL_QUAD_ORDER, "3", Patterns::Integer(1, 20),
            "order of quadrature (QGauss, exact in polynomials of degree ≤ 2n-1, use at least finite element degree + 1) ");
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_LOG);
   {
      prm->declare_entry(KEY_LOG_FILE, "wavepi.log", Patterns::FileName(Patterns::FileName::output),
            "external log file");
      prm->declare_entry(KEY_LOG_FILE_DEPTH, "100", Patterns::Integer(0), "depth for the log file");
      prm->declare_entry(KEY_LOG_CONSOLE_DEPTH, "2", Patterns::Integer(0), "depth for stdout");
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_MESH);
   {
      prm->declare_entry(KEY_MESH_END_TIME, "6", Patterns::Double(0), "time horizon T");
      prm->declare_entry(KEY_MESH_INITIAL_REFINES, "4", Patterns::Integer(0),
            "refines of the (initial) spatial grid");
      prm->declare_entry(KEY_MESH_INITIAL_TIME_STEPS, "256", Patterns::Integer(2),
            "(initial) number of time steps");

      prm->enter_subsection(KEY_MESH_SHAPE);
      {
         prm->declare_entry(KEY_MESH_SHAPE_GENERATOR, "hyper_cube",
               Patterns::Selection("hyper_cube|hyper_L|hyper_ball|cheese"),
               "generator for the triangulation");
         prm->declare_entry(KEY_MESH_SHAPE_OPTIONS, "left=-5.0, right=5.0", Patterns::Anything(),
               "options for the generator, in the form `var1=value1, var2=value2, ...`.\n Available options: left, right for hyper_cube and hyper_L, center_{x,y,z} and radius for hyper_cube, scale for cheese.");
      }
      prm->leave_subsection();
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_PROBLEM);
   {
      prm->declare_entry(KEY_PROBLEM_TYPE, "L2A", Patterns::Selection("L2A|L2Q|L2Nu|L2C"),
            "parameter that is reconstructed, and which spaces are used");
      prm->declare_entry(KEY_PROBLEM_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");

      prm->declare_entry(KEY_PROBLEM_CONSTANTS, "", Patterns::Anything(),
            "constants for the function declarations,\nin the form `var1=value1, var2=value2, ...`.");

      prm->declare_entry(KEY_PROBLEM_GUESS, "0.5", Patterns::Anything(), "initial guess");

      prm->declare_entry(KEY_PROBLEM_PARAM_A, "1.0", Patterns::Anything(), "parameter a");
      prm->declare_entry(KEY_PROBLEM_PARAM_Q, "0.0", Patterns::Anything(), "parameter q");
      prm->declare_entry(KEY_PROBLEM_PARAM_C, "2.0", Patterns::Anything(), "parameter c");
      prm->declare_entry(KEY_PROBLEM_PARAM_NU, "0.0", Patterns::Anything(), "parameter ν");

      prm->enter_subsection(KEY_PROBLEM_DATA);
      {
         prm->declare_entry(KEY_PROBLEM_DATA_COUNT, "1", Patterns::Integer(1),
               "Number of configurations. Each configuration has its own right hand side and own measurement settings. Make sure that there are at least as many `configuration {i}` blocks as this number.");

         prm->enter_subsection(KEY_PROBLEM_DATA_I + "1");
         {
            prm->declare_entry(KEY_PROBLEM_DATA_I_RHS, "if(norm{x|y|z} < 0.2, sin(t), 0.0)",
                  Patterns::Anything(), "right hand side");

            prm->declare_entry(KEY_PROBLEM_DATA_I_MEASURE, "None", Patterns::Selection("Identical|Grid"),
                  "type of measurements");

            GridPointMeasure<2>::declare_parameters(*prm);
         }
         prm->leave_subsection();
      }
      prm->leave_subsection();
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_INVERSION);
   {
      prm->declare_entry(KEY_INVERSION_METHOD, "REGINN", Patterns::Selection("REGINN|NonlinearLandweber"),
            "solver for the inverse problem");
      prm->declare_entry(KEY_INVERSION_TAU, "2", Patterns::Double(0),
            "parameter τ for discrepancy principle");

      REGINN<DiscretizedFunction<2>, DiscretizedFunction<2>, Function<2>>::declare_parameters(*prm);
      NonlinearLandweber<DiscretizedFunction<2>, DiscretizedFunction<2>, Function<2>>::declare_parameters(
            *prm);
   }
   prm->leave_subsection();

   WaveEquationBase<2>::declare_parameters(*prm);
   OutputProgressListener<2>::declare_parameters(*prm);
}

void SettingsManager::get_parameters(std::shared_ptr<ParameterHandler> prm) {
   this->prm = prm;
   AssertThrow(prm, ExcInternalError());

   prm->enter_subsection(KEY_LOG);
   {
      log_file = prm->get(KEY_LOG_FILE);
      log_file_depth = prm->get_integer(KEY_LOG_FILE_DEPTH);
      log_console_depth = prm->get_integer(KEY_LOG_CONSOLE_DEPTH);
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_GENERAL);
   {
      fe_degree = prm->get_integer(KEY_GENERAL_FE_DEGREE);
      quad_order = prm->get_integer(KEY_GENERAL_QUAD_ORDER);
      dimension = prm->get_integer(KEY_GENERAL_DIMENSION);
   }
   prm->leave_subsection();

   prm->enter_subsection(KEY_MESH);
   {
      end_time = prm->get_double(KEY_MESH_END_TIME);
      initial_refines = prm->get_integer(KEY_MESH_INITIAL_REFINES);
      initial_time_steps = prm->get_integer(KEY_MESH_INITIAL_TIME_STEPS);

      dt = end_time / (initial_time_steps - 1);
      times.clear();
      times.resize(initial_time_steps);

      for (size_t i = 0; i < initial_time_steps; i++)
         times[i] = i * dt;

      prm->enter_subsection(KEY_MESH_SHAPE);
      {
         std::string generator = prm->get(KEY_MESH_SHAPE_GENERATOR);

         std::string options_list = prm->get(KEY_MESH_SHAPE_OPTIONS);
         std::vector<std::string> options_listed = Utilities::split_string_list(options_list, ',');

         shape_options.clear();

         for (size_t i = 0; i < options_listed.size(); ++i) {
            std::vector<std::string> this_c = Utilities::split_string_list(options_listed[i], '=');
            AssertThrow(this_c.size() == 2, ExcMessage("Could not parse generator options"));
            double tmp;
            AssertThrow(std::sscanf(this_c[1].c_str(), "%lf", &tmp),
                  ExcMessage("Could not parse generator options"));
            shape_options[this_c[0]] = tmp;
         }

         if (generator == "hyper_cube") {
            if (!shape_options.count("left"))
               shape_options.emplace("left", -5.0);

            if (!shape_options.count("right"))
               shape_options.emplace("right", 5.0);

            shape = MeshShape::hyper_cube;
         } else if (generator == "hyper_L") {
            if (!shape_options.count("left"))
               shape_options.emplace("left", -5.0);

            if (!shape_options.count("right"))
               shape_options.emplace("right", 5.0);

            shape = MeshShape::hyper_L;
         } else if (generator == "hyper_ball") {
            if (!shape_options.count("center_x"))
               shape_options.emplace("center_x", 0.0);

            if (!shape_options.count("center_y"))
               shape_options.emplace("center_y", 0.0);

            if (!shape_options.count("center_z"))
               shape_options.emplace("center_z", 0.0);

            if (!shape_options.count("radius"))
               shape_options.emplace("radius", 1.0);

            shape = MeshShape::hyper_ball;
         } else if (generator == "cheese") {
            if (!shape_options.count("scale"))
               shape_options.emplace("scale", 1.0);

            AssertThrow(dimension > 1, ExcMessage("cheese only makes sense for dim > 1."));
            shape = MeshShape::cheese;
         } else
            AssertThrow(false, ExcMessage("Unknown grid generator:" + generator));
      }
      prm->leave_subsection();
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

      constants_for_exprs.clear();

      for (size_t i = 0; i < const_listed.size(); ++i) {
         std::vector<std::string> this_c = Utilities::split_string_list(const_listed[i], '=');
         AssertThrow(this_c.size() == 2, ExcMessage("Invalid format"));
         double tmp;
         AssertThrow(std::sscanf(this_c[1].c_str(), "%lf", &tmp), ExcMessage("Double number?"));
         constants_for_exprs[this_c[0]] = tmp;
      }

      expr_initial_guess = prm->get(KEY_PROBLEM_GUESS);
      expr_param_a = prm->get(KEY_PROBLEM_PARAM_A);
      expr_param_nu = prm->get(KEY_PROBLEM_PARAM_NU);
      expr_param_c = prm->get(KEY_PROBLEM_PARAM_C);
      expr_param_q = prm->get(KEY_PROBLEM_PARAM_Q);

      prm->enter_subsection(KEY_PROBLEM_DATA);
      {
         num_configurations = prm->get_integer(KEY_PROBLEM_DATA_COUNT);

         exprs_rhs.clear();
         measures.clear();

         for (size_t i = 0; i < num_configurations; i++) {
            prm->enter_subsection(KEY_PROBLEM_DATA_I + Utilities::int_to_string(i, 1));

            exprs_rhs.push_back(prm->get(KEY_PROBLEM_DATA_I_RHS));

            auto measure_desc = prm->get(KEY_PROBLEM_DATA_I_MEASURE);
            MeasureType my_measure_type;

            if (measure_desc == "Identical") {
               measures.push_back(Measure::identical);
               my_measure_type = MeasureType::discretized_function;
            } else if (measure_desc == "Grid") {
               measures.push_back(Measure::identical);
               my_measure_type = MeasureType::discretized_function;
            } else
               AssertThrow(false, ExcMessage("Unknown Measure: " + measure_desc));

            if (!i)
               measure_type = my_measure_type;

            AssertThrow(measure_type == my_measure_type,
                  ExcMessage("the resulting data types must be the same for all measurement configurations!"))

            GridPointMeasure<2>::declare_parameters(*prm);

            prm->leave_subsection();
         }
      }
      prm->leave_subsection();
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
         AssertThrow(false, ExcMessage("Unknown Method: " + smethod));
   }

   prm->leave_subsection();
}

void SettingsManager::log() {
   unsigned int prev_console = deallog.depth_console(100);
   unsigned int prev_file = deallog.depth_file(100);

   prm->log_parameters(deallog);

   deallog.depth_console(prev_console);
   deallog.depth_file(prev_file);
}

}
/* namespace wavepi */
