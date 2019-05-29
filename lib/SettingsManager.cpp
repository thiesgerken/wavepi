/*
 * SettingsManager.cpp
 *
 *  Created on: 22.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <base/DiscretizedFunction.h>
#include <base/Tuple.h>
#include <forward/WaveEquationBase.h>
#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/REGINN.h>
#include <measurements/ConvolutionMeasure.h>
#include <measurements/CubeBoundaryDistribution.h>
#include <measurements/GridDistribution.h>
#include <measurements/SensorDistribution.h>

#include <SettingsManager.h>

#include <tgmath.h>
#include <cstdio>
#include <utility>

namespace wavepi {
using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::measurements;

const std::string SettingsManager::KEY_GENERAL            = "general";
const std::string SettingsManager::KEY_GENERAL_DIMENSION  = "dimension";
const std::string SettingsManager::KEY_GENERAL_FE_DEGREE  = "finite element degree";
const std::string SettingsManager::KEY_GENERAL_QUAD_ORDER = "quadrature order";

const std::string SettingsManager::KEY_LOG                = "log";
const std::string SettingsManager::KEY_LOG_FILE           = "file";
const std::string SettingsManager::KEY_LOG_FILE_DEPTH     = "file depth";
const std::string SettingsManager::KEY_LOG_FILE_DEPTH_MPI = "file depth mpi";
const std::string SettingsManager::KEY_LOG_CONSOLE_DEPTH  = "console depth";

const std::string SettingsManager::KEY_MESH                    = "mesh";
const std::string SettingsManager::KEY_MESH_END_TIME           = "end time";
const std::string SettingsManager::KEY_MESH_INITIAL_REFINES    = "initial refines";
const std::string SettingsManager::KEY_MESH_INITIAL_TIME_STEPS = "initial time steps";
const std::string SettingsManager::KEY_MESH_SHAPE              = "shape";
const std::string SettingsManager::KEY_MESH_SHAPE_GENERATOR    = "generator name";
const std::string SettingsManager::KEY_MESH_SHAPE_OPTIONS      = "options";

const std::string SettingsManager::KEY_PROBLEM                      = "problem";
const std::string SettingsManager::KEY_PROBLEM_TYPE                 = "type";
const std::string SettingsManager::KEY_PROBLEM_TRANSFORM            = "transform";
const std::string SettingsManager::KEY_PROBLEM_NORM_DOMAIN          = "norm of domain";
const std::string SettingsManager::KEY_PROBLEM_NORM_CODOMAIN        = "norm of codomain";
const std::string SettingsManager::KEY_PROBLEM_NORM_DOMAIN_P_ENABLE = "domain lp wrapping";
const std::string SettingsManager::KEY_PROBLEM_NORM_DOMAIN_P        = "domain p";

const std::string SettingsManager::KEY_PROBLEM_NORM_H1L2ALPHA         = "H1L2 alpha";
const std::string SettingsManager::KEY_PROBLEM_NORM_H2L2ALPHA         = "H2L2 alpha";
const std::string SettingsManager::KEY_PROBLEM_NORM_H2L2BETA          = "H2L2 beta";
const std::string SettingsManager::KEY_PROBLEM_NORM_H1H1ALPHA         = "H1H1 alpha";
const std::string SettingsManager::KEY_PROBLEM_NORM_H1H1GAMMA         = "H1H1 gamma";
const std::string SettingsManager::KEY_PROBLEM_NORM_H2L2PLUSL2H1ALPHA = "H2L2+L2H1 alpha";
const std::string SettingsManager::KEY_PROBLEM_NORM_H2L2PLUSL2H1BETA  = "H2L2+L2H1 beta";
const std::string SettingsManager::KEY_PROBLEM_NORM_H2L2PLUSL2H1GAMMA = "H2L2+L2H1 gamma";

const std::string SettingsManager::KEY_PROBLEM_EPSILON           = "epsilon";
const std::string SettingsManager::KEY_PROBLEM_CONSTANTS         = "constants";
const std::string SettingsManager::KEY_PROBLEM_GUESS             = "initial guess";
const std::string SettingsManager::KEY_PROBLEM_PARAM_RHO         = "parameter rho";
const std::string SettingsManager::KEY_PROBLEM_PARAM_RHO_DYNAMIC = "parameter rho dynamic";
const std::string SettingsManager::KEY_PROBLEM_PARAM_Q           = "parameter q";
const std::string SettingsManager::KEY_PROBLEM_PARAM_C           = "parameter c";
const std::string SettingsManager::KEY_PROBLEM_PARAM_NU          = "parameter nu";
const std::string SettingsManager::KEY_PROBLEM_PARAM_BACKGROUND  = "background parameter";
const std::string SettingsManager::KEY_PROBLEM_SHAPE_SCALE       = "shape scaling";

const std::string SettingsManager::KEY_PROBLEM_DATA                       = "data";
const std::string SettingsManager::KEY_PROBLEM_DATA_ADDITIONAL_REFINES    = "additional refines";
const std::string SettingsManager::KEY_PROBLEM_DATA_ADDITIONAL_DEGREE     = "additional fe degrees";
const std::string SettingsManager::KEY_PROBLEM_DATA_COUNT                 = "number of right hand sides";
const std::string SettingsManager::KEY_PROBLEM_DATA_RHS                   = "right hand sides";
const std::string SettingsManager::KEY_PROBLEM_DATA_CONFIG                = "configurations";
const std::string SettingsManager::KEY_PROBLEM_DATA_I                     = "config ";  // + number
const std::string SettingsManager::KEY_PROBLEM_DATA_I_MEASURE             = "measure";
const std::string SettingsManager::KEY_PROBLEM_DATA_I_MASK                = "mask";
const std::string SettingsManager::KEY_PROBLEM_DATA_I_SENSOR_DISTRIBUTION = "sensor distribution";

const std::string SettingsManager::KEY_INVERSION        = "inversion";
const std::string SettingsManager::KEY_INVERSION_TAU    = "tau";
const std::string SettingsManager::KEY_INVERSION_METHOD = "method";

void SettingsManager::declare_parameters(std::shared_ptr<ParameterHandler> prm) {
  OutputProgressListener<2, Tuple<DiscretizedFunction<2>>>::declare_parameters(*prm);
  WatchdogProgressListener<Tuple<DiscretizedFunction<2>>, Tuple<DiscretizedFunction<2>>>::declare_parameters(*prm);
  StatOutputProgressListener<Tuple<DiscretizedFunction<2>>, Tuple<DiscretizedFunction<2>>>::declare_parameters(*prm);
  AbstractEquation<2>::declare_parameters(*prm);
  BoundCheckProgressListener<2, Tuple<DiscretizedFunction<2>>, DiscretizedFunction<2>>::declare_parameters(*prm);
  LogTransform<2>::declare_parameters(*prm);
  ArtanhTransform<2>::declare_parameters(*prm);

  prm->enter_subsection(KEY_GENERAL);
  {
    prm->declare_entry(KEY_GENERAL_DIMENSION, "2", Patterns::Integer(1, 3), "problem dimension");
    prm->declare_entry(KEY_GENERAL_FE_DEGREE, "1", Patterns::Integer(1, 4),
                       "polynomial degree of finite elements. Note that bound "
                       "checking is currently only implemented for linear "
                       "elements.");
    prm->declare_entry(KEY_GENERAL_QUAD_ORDER, "3", Patterns::Integer(1, 20),
                       "order of quadrature (QGauss, exact in polynomials of "
                       "degree ≤ 2n-1, use at least finite element degree + "
                       "1) ");
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_LOG);
  {
    prm->declare_entry(KEY_LOG_FILE, "wavepi.log", Patterns::FileName(Patterns::FileName::output), "external log file");
    prm->declare_entry(KEY_LOG_FILE_DEPTH, "3", Patterns::Integer(0), "depth for the log file (root process)");
    prm->declare_entry(KEY_LOG_FILE_DEPTH_MPI, "3", Patterns::Integer(0), "depth for the log file (other processes)");
    prm->declare_entry(KEY_LOG_CONSOLE_DEPTH, "2", Patterns::Integer(0), "depth for stdout (root process)");
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_MESH);
  {
    prm->declare_entry(KEY_MESH_END_TIME, "6.28318530718", Patterns::Double(0), "time horizon T");
    prm->declare_entry(KEY_MESH_INITIAL_REFINES, "6", Patterns::Integer(0), "refines of the (initial) spatial grid");
    prm->declare_entry(KEY_MESH_INITIAL_TIME_STEPS, "256", Patterns::Integer(2), "(initial) number of time steps");

    prm->enter_subsection(KEY_MESH_SHAPE);
    {
      prm->declare_entry(KEY_MESH_SHAPE_GENERATOR, "hyper_cube",
                         Patterns::Selection("hyper_cube|hyper_L|hyper_ball|cheese"),
                         "generator for the triangulation");
      prm->declare_entry(KEY_MESH_SHAPE_OPTIONS, "left=-1.0, right=1.0", Patterns::Anything(),
                         "options for the generator, in the form `var1=value1, var2=value2, "
                         "...`.\n Available options: left, right for hyper_cube and hyper_L, "
                         "center_{x,y,z} and radius for hyper_cube, scale for cheese.");
    }
    prm->leave_subsection();
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_PROBLEM);
  {
    prm->declare_entry(KEY_PROBLEM_TYPE, "rho", Patterns::Selection("c|nu|rho|rho_constant|q"),
                       "parameter that is reconstructed");

    prm->declare_entry(KEY_PROBLEM_TRANSFORM, "Identity", Patterns::Selection("Identity|Log|Artanh"),
                       "transformation to apply to the parameter (e.g. to get rid of constraints)");

    prm->declare_entry(KEY_PROBLEM_NORM_DOMAIN, "L2L2",
                       Patterns::Selection("L2L2|H1L2|H2L2|H1H1|H2L2PlusL2H1|Coefficients"),
                       "norm to use for parameters (incl. the reconstruction)");
    prm->declare_entry(KEY_PROBLEM_NORM_CODOMAIN, "L2L2",
                       Patterns::Selection("L2L2|H1L2|H2L2|H1H1|H2L2PlusL2H1|Coefficients"),
                       "Set the norm to use for fields. Be aware that this has "
                       "to match the norm that the measurements expect its "
                       "inputs to have.");

    prm->declare_entry(KEY_PROBLEM_NORM_DOMAIN_P, "2.0", Patterns::Double(1),
                       "index p for the l^p-norm. See also " + KEY_PROBLEM_NORM_DOMAIN_P_ENABLE + ".");
    prm->declare_entry(KEY_PROBLEM_NORM_DOMAIN_P_ENABLE, "false", Patterns::Bool(),
                       "Wrap the norm of the parameter space inside a l^p-norm");

    prm->declare_entry(KEY_PROBLEM_NORM_H1L2ALPHA, "0.5", Patterns::Double(0),
                       "Factor α in front of derivative term of H¹([0,T], L²(Ω)) dot product");

    prm->declare_entry(KEY_PROBLEM_NORM_H2L2ALPHA, "0.5", Patterns::Double(0),
                       "Factor α in front of first derivative term of "
                       "H²([0,T], L²(Ω)) dot product");
    prm->declare_entry(KEY_PROBLEM_NORM_H2L2BETA, "0.25", Patterns::Double(0),
                       "Factor β in front of second derivative term of "
                       "H²([0,T], L²(Ω)) dot product");

    prm->declare_entry(KEY_PROBLEM_NORM_H1H1ALPHA, "0.5", Patterns::Double(0),
                       "Factor α in front of time derivative term of H¹([0,T], H¹(Ω)) dot product");
    prm->declare_entry(KEY_PROBLEM_NORM_H1H1GAMMA, "0.5", Patterns::Double(0),
                       "Factor ɣ in front of gradient term of H¹([0,T], H¹(Ω)) dot product");

    prm->declare_entry(KEY_PROBLEM_NORM_H2L2PLUSL2H1ALPHA, "0.5", Patterns::Double(0),
                       "Factor α in front of time derivative term of H²([0,T], L²(Ω)) ∩ L²([0,T], H¹(Ω)) dot product");
    prm->declare_entry(KEY_PROBLEM_NORM_H2L2PLUSL2H1BETA, "0.25", Patterns::Double(0),
                       "Factor β in front of second time derivative term of "
                       "H²([0,T], L²(Ω)) ∩ L²([0,T], H¹(Ω)) dot product");
    prm->declare_entry(KEY_PROBLEM_NORM_H2L2PLUSL2H1GAMMA, "0.5", Patterns::Double(0),
                       "Factor ɣ in front of gradient term of H²([0,T], L²(Ω)) ∩ L²([0,T], H¹(Ω)) dot product");

    prm->declare_entry(KEY_PROBLEM_EPSILON, "1e-2", Patterns::Double(0, 1), "relative noise level ε");

    prm->declare_entry(KEY_PROBLEM_CONSTANTS, "", Patterns::Anything(),
                       "constants for the function declarations,\nin the form "
                       "`var1=value1, var2=value2, ...`.");

    prm->declare_entry(KEY_PROBLEM_GUESS, "0.0", Patterns::Anything(), "initial guess");

    prm->declare_entry(KEY_PROBLEM_PARAM_RHO, "1.0", Patterns::Anything(), "parameter ρ");
    prm->declare_entry(KEY_PROBLEM_PARAM_RHO_DYNAMIC, "true", Patterns::Bool(),
                       "parameter ρ time-dependent? (performance)");
    prm->declare_entry(KEY_PROBLEM_PARAM_Q, "0.0", Patterns::Anything(), "parameter q");
    prm->declare_entry(KEY_PROBLEM_PARAM_C, "2.0", Patterns::Anything(), "parameter c");
    prm->declare_entry(KEY_PROBLEM_PARAM_NU, "0.0", Patterns::Anything(), "parameter ν");
    prm->declare_entry(KEY_PROBLEM_PARAM_BACKGROUND, "0.0", Patterns::Anything(),
                       "background parameter (added to all arguments of the forward operator, including exact data)");
    prm->declare_entry(KEY_PROBLEM_SHAPE_SCALE, "1.0", Patterns::Double(0),
                       "scaling parameter applied to all shapes that are used");

    prm->enter_subsection(KEY_PROBLEM_DATA);
    {
      prm->declare_entry(
          KEY_PROBLEM_DATA_ADDITIONAL_REFINES, "0", Patterns::Integer(0),
          "Additional global refines to perform in space and time for data synthesis (to avoid inverse crime)");
      prm->declare_entry(KEY_PROBLEM_DATA_ADDITIONAL_DEGREE, "0", Patterns::Integer(0),
                         "Additional finite element degrees to use for data synthesis (to avoid inverse crime)");

      prm->declare_entry(KEY_PROBLEM_DATA_COUNT, "1", Patterns::Integer(1), "Number of right hand sides");

      prm->declare_entry(KEY_PROBLEM_DATA_CONFIG, "0", Patterns::Anything(),
                         "configuration to use for which right hand side, "
                         "separated by semicolons");
      prm->declare_entry(KEY_PROBLEM_DATA_RHS, "if(t<3.14, if(norm{x|y|z} < 0.1, sin(2*t), 0.0), 0.0)",
                         Patterns::Anything(), "right hand sides, separated by semicolons");

      for (size_t i = 0; i < num_configurations; i++) {
        prm->enter_subsection(KEY_PROBLEM_DATA_I + Utilities::int_to_string(i, 1));

        ConvolutionMeasure<2>::declare_parameters(*prm);
        GridDistribution<2>::declare_parameters(*prm);
        CubeBoundaryDistribution<2>::declare_parameters(*prm);

        prm->declare_entry(KEY_PROBLEM_DATA_I_MEASURE, "Field",
                           Patterns::Selection("Field|MaskedField|Convolution|Delta"), "type of measurements");

        prm->declare_entry(KEY_PROBLEM_DATA_I_SENSOR_DISTRIBUTION, "Grid", Patterns::Selection("Grid|CubeBoundary"),
                           "in case of simulated sensors, their location on the mesh");

        prm->declare_entry(KEY_PROBLEM_DATA_I_MASK, "1", Patterns::Anything(),
                           "in case of MaskedField, the mask function");

        prm->leave_subsection();
      }
    }
    prm->leave_subsection();
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_INVERSION);
  {
    prm->declare_entry(KEY_INVERSION_METHOD, "REGINN", Patterns::Selection("REGINN|NonlinearLandweber"),
                       "solver for the inverse problem");
    prm->declare_entry(KEY_INVERSION_TAU, "2", Patterns::Double(0), "parameter τ for discrepancy principle");

    REGINN<DiscretizedFunction<2>, DiscretizedFunction<2>, Function<2>>::declare_parameters(*prm);
    NonlinearLandweber<DiscretizedFunction<2>, DiscretizedFunction<2>, Function<2>>::declare_parameters(*prm);
  }
  prm->leave_subsection();
}

void SettingsManager::get_parameters(std::shared_ptr<ParameterHandler> prm) {
  this->prm = prm;
  AssertThrow(prm, ExcInternalError());

  prm->enter_subsection(KEY_LOG);
  {
    log_file           = prm->get(KEY_LOG_FILE);
    log_file_depth     = prm->get_integer(KEY_LOG_FILE_DEPTH);
    log_file_depth_mpi = prm->get_integer(KEY_LOG_FILE_DEPTH_MPI);
    log_console_depth  = prm->get_integer(KEY_LOG_CONSOLE_DEPTH);
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_GENERAL);
  {
    fe_degree  = prm->get_integer(KEY_GENERAL_FE_DEGREE);
    quad_order = prm->get_integer(KEY_GENERAL_QUAD_ORDER);
    dimension  = prm->get_integer(KEY_GENERAL_DIMENSION);
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_MESH);
  {
    end_time           = prm->get_double(KEY_MESH_END_TIME);
    initial_refines    = prm->get_integer(KEY_MESH_INITIAL_REFINES);
    initial_time_steps = prm->get_integer(KEY_MESH_INITIAL_TIME_STEPS);

    dt = end_time / (initial_time_steps - 1);
    times.clear();
    times.resize(initial_time_steps);

    for (size_t i = 0; i < initial_time_steps; i++)
      times[i] = i * dt;

    prm->enter_subsection(KEY_MESH_SHAPE);
    {
      std::string generator = prm->get(KEY_MESH_SHAPE_GENERATOR);

      std::string options_list                = prm->get(KEY_MESH_SHAPE_OPTIONS);
      std::vector<std::string> options_listed = Utilities::split_string_list(options_list, ',');

      shape_options.clear();

      for (size_t i = 0; i < options_listed.size(); ++i) {
        std::vector<std::string> this_c = Utilities::split_string_list(options_listed[i], '=');
        AssertThrow(this_c.size() == 2, ExcMessage("Could not parse generator options"));
        double tmp;
        AssertThrow(std::sscanf(this_c[1].c_str(), "%lf", &tmp), ExcMessage("Could not parse generator options"));
        shape_options[this_c[0]] = tmp;
      }

      if (generator == "hyper_cube") {
        if (!shape_options.count("left")) shape_options.emplace("left", -5.0);
        if (!shape_options.count("right")) shape_options.emplace("right", 5.0);

        shape = MeshShape::hyper_cube;
      } else if (generator == "hyper_L") {
        if (!shape_options.count("left")) shape_options.emplace("left", -5.0);
        if (!shape_options.count("right")) shape_options.emplace("right", 5.0);

        shape = MeshShape::hyper_l;
      } else if (generator == "hyper_ball") {
        if (!shape_options.count("center_x")) shape_options.emplace("center_x", 0.0);
        if (!shape_options.count("center_y")) shape_options.emplace("center_y", 0.0);
        if (!shape_options.count("center_z")) shape_options.emplace("center_z", 0.0);
        if (!shape_options.count("radius")) shape_options.emplace("radius", 1.0);

        shape = MeshShape::hyper_ball;
      } else if (generator == "cheese") {
        if (!shape_options.count("scale")) shape_options.emplace("scale", 1.0);

        AssertThrow(dimension > 1, ExcMessage("cheese only makes sense for dim > 1."));
        shape = MeshShape::cheese;
      } else {
        AssertThrow(false, ExcMessage("Unknown mesh generator:" + generator));
      }
    }
    prm->leave_subsection();
  }
  prm->leave_subsection();

  prm->enter_subsection(KEY_PROBLEM);
  {
    epsilon = prm->get_double(KEY_PROBLEM_EPSILON);

    std::string norm_domain_s = prm->get(KEY_PROBLEM_NORM_DOMAIN);

    if (norm_domain_s == "L2L2")
      norm_domain = NormType::l2l2;
    else if (norm_domain_s == "H1L2")
      norm_domain = NormType::h1l2;
    else if (norm_domain_s == "H2L2")
      norm_domain = NormType::h2l2;
    else if (norm_domain_s == "H1H1")
      norm_domain = NormType::h1h1;
    else if (norm_domain_s == "H2L2PlusL2H1")
      norm_domain = NormType::h2l2plusl2h1;
    else if (norm_domain_s == "Coefficients")
      norm_domain = NormType::vector;
    else
      AssertThrow(false, ExcMessage("Cannot parse norm of domain"));

    std::string norm_codomain_s = prm->get(KEY_PROBLEM_NORM_CODOMAIN);

    if (norm_codomain_s == "L2L2")
      norm_codomain = NormType::l2l2;
    else if (norm_codomain_s == "H1L2")
      norm_codomain = NormType::h1l2;
    else if (norm_codomain_s == "H2L2")
      norm_codomain = NormType::h2l2;
    else if (norm_codomain_s == "H1H1")
      norm_codomain = NormType::h1h1;
    else if (norm_codomain_s == "H2L2PlusL2H1")
      norm_codomain = NormType::h2l2plusl2h1;
    else if (norm_codomain_s == "Coefficients")
      norm_codomain = NormType::vector;
    else
      AssertThrow(false, ExcMessage("Cannot parse norm of codomain"));

    norm_domain_enable_wrapping = prm->get_bool(KEY_PROBLEM_NORM_DOMAIN_P_ENABLE);
    norm_domain_p               = prm->get_double(KEY_PROBLEM_NORM_DOMAIN_P);

    norm_h1l2_alpha = prm->get_double(KEY_PROBLEM_NORM_H1L2ALPHA);

    norm_h2l2_alpha = prm->get_double(KEY_PROBLEM_NORM_H2L2ALPHA);
    norm_h2l2_beta  = prm->get_double(KEY_PROBLEM_NORM_H2L2BETA);

    norm_h1h1_alpha = prm->get_double(KEY_PROBLEM_NORM_H1H1ALPHA);
    norm_h1h1_gamma = prm->get_double(KEY_PROBLEM_NORM_H1H1GAMMA);

    norm_h2l2plusl2h1_alpha = prm->get_double(KEY_PROBLEM_NORM_H2L2PLUSL2H1ALPHA);
    norm_h2l2plusl2h1_beta  = prm->get_double(KEY_PROBLEM_NORM_H2L2PLUSL2H1BETA);
    norm_h2l2plusl2h1_gamma = prm->get_double(KEY_PROBLEM_NORM_H2L2PLUSL2H1GAMMA);

    std::string transform_s = prm->get(KEY_PROBLEM_TRANSFORM);

    if (transform_s == "Identity")
      transform = TransformType::identity;
    else if (transform_s == "Log")
      transform = TransformType::log;
    else if (transform_s == "Artanh")
      transform = TransformType::artanh;
    else
      AssertThrow(false, ExcMessage("Cannot parse transform type"));

    std::string problem = prm->get(KEY_PROBLEM_TYPE);

    if (problem == "rho")
      problem_type = ProblemType::rho;
    else if (problem == "q")
      problem_type = ProblemType::q;
    else if (problem == "nu")
      problem_type = ProblemType::nu;
    else if (problem == "c")
      problem_type = ProblemType::c;
    else if (problem == "rho_constant")
      problem_type = ProblemType::rho_constant;
    else
      AssertThrow(false, ExcMessage("unknown problem type"));

    std::string constants_list            = prm->get(KEY_PROBLEM_CONSTANTS);
    std::vector<std::string> const_listed = Utilities::split_string_list(constants_list, ',');

    constants_for_exprs.clear();

    for (size_t i = 0; i < const_listed.size(); ++i) {
      std::vector<std::string> this_c = Utilities::split_string_list(const_listed[i], '=');
      AssertThrow(this_c.size() == 2, ExcMessage("Invalid format"));
      double tmp;
      AssertThrow(std::sscanf(this_c[1].c_str(), "%lf", &tmp), ExcMessage("Double number?"));
      constants_for_exprs[this_c[0]] = tmp;
    }

    expr_initial_guess    = prm->get(KEY_PROBLEM_GUESS);
    expr_param_rho        = prm->get(KEY_PROBLEM_PARAM_RHO);
    expr_param_nu         = prm->get(KEY_PROBLEM_PARAM_NU);
    expr_param_c          = prm->get(KEY_PROBLEM_PARAM_C);
    expr_param_q          = prm->get(KEY_PROBLEM_PARAM_Q);
    expr_param_background = prm->get(KEY_PROBLEM_PARAM_BACKGROUND);

    shape_scale = prm->get_double(KEY_PROBLEM_SHAPE_SCALE);
    rho_dynamic = prm->get_bool(KEY_PROBLEM_PARAM_RHO_DYNAMIC);

    prm->enter_subsection(KEY_PROBLEM_DATA);
    {
      num_rhs   = prm->get_integer(KEY_PROBLEM_DATA_COUNT);
      exprs_rhs = Utilities::split_string_list(prm->get(KEY_PROBLEM_DATA_RHS), ';');

      synthesis_additional_refines    = prm->get_integer(KEY_PROBLEM_DATA_ADDITIONAL_REFINES);
      synthesis_additional_fe_degrees = prm->get_integer(KEY_PROBLEM_DATA_ADDITIONAL_DEGREE);

      configs.clear();
      measures.clear();
      sensor_distributions.clear();
      expr_masks.clear();

      try {
        for (auto is : Utilities::split_string_list(prm->get(KEY_PROBLEM_DATA_CONFIG), ';')) {
          size_t ci = std::stoi(is);
          AssertThrow(ci < num_configurations, ExcIndexRange(ci, 0, num_configurations));
          configs.push_back(ci);
        }
      } catch (std::invalid_argument &e) {
        AssertThrow(false, ExcMessage("could not parse configurations: " + std::string(e.what())));
      }

      AssertThrow(num_rhs == exprs_rhs.size(), ExcMessage("number of right hand sides != number of expressions given"));
      AssertThrow(num_rhs == configs.size(),
                  ExcMessage("number of right hand sides != number of configurations given"));

      std::vector<MeasureType> measure_types;

      for (size_t i = 0; i < num_configurations; i++) {
        prm->enter_subsection(KEY_PROBLEM_DATA_I + Utilities::int_to_string(i, 1));

        auto measure_desc = prm->get(KEY_PROBLEM_DATA_I_MEASURE);
        MeasureType my_measure_type;

        if (measure_desc == "Field") {
          measures.push_back(Measure::field);
          my_measure_type = MeasureType::discretized_function;
        } else if (measure_desc == "MaskedField") {
          measures.push_back(Measure::masked_field);
          my_measure_type = MeasureType::discretized_function;
        } else if (measure_desc == "Convolution") {
          measures.push_back(Measure::convolution);
          my_measure_type = MeasureType::vector;
        } else if (measure_desc == "Delta") {
          measures.push_back(Measure::delta);
          my_measure_type = MeasureType::vector;
        } else {
          AssertThrow(false, ExcMessage("Unknown Measure: " + measure_desc));
        }

        measure_types.push_back(my_measure_type);

        auto distrib_desc = prm->get(KEY_PROBLEM_DATA_I_SENSOR_DISTRIBUTION);

        if (distrib_desc == "Grid") {
          sensor_distributions.push_back(SettingsManager::SensorDistribution::grid);
        } else if (distrib_desc == "CubeBoundary") {
          sensor_distributions.push_back(SettingsManager::SensorDistribution::cube_boundary);
        } else {
          AssertThrow(false, ExcMessage("Unknown sensor distribution: " + distrib_desc));
        }

        expr_masks.push_back(prm->get(KEY_PROBLEM_DATA_I_MASK));
        prm->leave_subsection();
      }

      for (size_t i = 0; i < num_rhs; i++) {
        auto my_measure_type = measure_types[configs[i]];

        if (!i) measure_type = my_measure_type;

        AssertThrow(measure_type == my_measure_type, ExcMessage("the resulting data types must be the same for "
                                                                "all used measurement configurations!"))
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
      method = NonlinearMethod::reginn;
    else if (smethod == "NonlinearLandweber")
      method = NonlinearMethod::nonlinear_landweber;
    else
      AssertThrow(false, ExcMessage("Unknown Method: " + smethod));
  }

  prm->leave_subsection();
}

void SettingsManager::log_parameters() {
#ifdef WAVEPI_MPI
  size_t mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
#else
  size_t mpi_rank = 0;
#endif

  unsigned int prev_console = mpi_rank > 0 ? 0 : deallog.depth_console(100);
  unsigned int prev_file    = deallog.depth_file(100);

  prm->log_parameters(deallog);

  deallog.depth_console(prev_console);
  deallog.depth_file(prev_file);
}

}  // namespace wavepi
