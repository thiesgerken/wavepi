/*
 * SettingsManager.h
 *
 *  Created on: 22.08.2017
 *      Author: thies
 */

#ifndef LIB_SETTINGSMANAGER_H_
#define LIB_SETTINGSMANAGER_H_

#include <deal.II/base/parameter_handler.h>

#include <base/Norm.h>

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace wavepi {
using namespace dealii;
using namespace wavepi::base;

/**
 * Manages settings for `wavepi.cpp` and class `WavePI`.
 */
class SettingsManager {
 public:
  static const std::string KEY_GENERAL;
  static const std::string KEY_GENERAL_DIMENSION;
  static const std::string KEY_GENERAL_FE_DEGREE;
  static const std::string KEY_GENERAL_QUAD_ORDER;

  static const std::string KEY_LOG;
  static const std::string KEY_LOG_FILE;
  static const std::string KEY_LOG_FILE_DEPTH;
  static const std::string KEY_LOG_FILE_DEPTH_MPI;
  static const std::string KEY_LOG_CONSOLE_DEPTH;

  static const std::string KEY_MESH;
  static const std::string KEY_MESH_END_TIME;
  static const std::string KEY_MESH_INITIAL_REFINES;
  static const std::string KEY_MESH_INITIAL_TIME_STEPS;
  static const std::string KEY_MESH_SHAPE;
  static const std::string KEY_MESH_SHAPE_GENERATOR;
  static const std::string KEY_MESH_SHAPE_OPTIONS;

  static const std::string KEY_PROBLEM;
  static const std::string KEY_PROBLEM_TYPE;
  static const std::string KEY_PROBLEM_TRANSFORM;
  static const std::string KEY_PROBLEM_NORM_DOMAIN;
  static const std::string KEY_PROBLEM_NORM_DOMAIN_P_ENABLE;
  static const std::string KEY_PROBLEM_NORM_DOMAIN_P;
  static const std::string KEY_PROBLEM_NORM_CODOMAIN;
  static const std::string KEY_PROBLEM_NORM_H1L2ALPHA;
  static const std::string KEY_PROBLEM_NORM_H2L2ALPHA;
  static const std::string KEY_PROBLEM_NORM_H2L2BETA;
  static const std::string KEY_PROBLEM_NORM_H1H1ALPHA;
  static const std::string KEY_PROBLEM_NORM_H1H1GAMMA;

  static const std::string KEY_PROBLEM_NORM_H2L2PLUSL2H1ALPHA;
  static const std::string KEY_PROBLEM_NORM_H2L2PLUSL2H1BETA;
  static const std::string KEY_PROBLEM_NORM_H2L2PLUSL2H1GAMMA;

  static const std::string KEY_PROBLEM_EPSILON;
  static const std::string KEY_PROBLEM_CONSTANTS;
  static const std::string KEY_PROBLEM_GUESS;
  static const std::string KEY_PROBLEM_PARAM_RHO;
  static const std::string KEY_PROBLEM_PARAM_RHO_DYNAMIC;
  static const std::string KEY_PROBLEM_PARAM_Q;
  static const std::string KEY_PROBLEM_PARAM_C;
  static const std::string KEY_PROBLEM_PARAM_NU;
  static const std::string KEY_PROBLEM_PARAM_BACKGROUND;
  static const std::string KEY_PROBLEM_SHAPE_SCALE;

  static const std::string KEY_INVERSION;
  static const std::string KEY_INVERSION_METHOD;
  static const std::string KEY_INVERSION_TAU;

  static const std::string KEY_PROBLEM_DATA;
  static const std::string KEY_PROBLEM_DATA_ADDITIONAL_REFINES;
  static const std::string KEY_PROBLEM_DATA_ADDITIONAL_DEGREE;
  static const std::string KEY_PROBLEM_DATA_COUNT;
  static const std::string KEY_PROBLEM_DATA_RHS;
  static const std::string KEY_PROBLEM_DATA_CONFIG;
  static const std::string KEY_PROBLEM_DATA_I;
  static const std::string KEY_PROBLEM_DATA_I_MEASURE;
  static const std::string KEY_PROBLEM_DATA_I_MASK;
  static const std::string KEY_PROBLEM_DATA_I_SENSOR_DISTRIBUTION;

  /**
   * possible problems
   */
  enum class ProblemType { q, rho, c, nu, rho_constant };

  /**
   * possible nonlinear methods
   */
  enum class NonlinearMethod { reginn, nonlinear_landweber };

  /**
   * possible mesh shapes
   */
  enum class MeshShape { hyper_cube, hyper_l, hyper_ball, cheese };

  /**
   * possible measurement operators
   */
  enum class Measure { field, masked_field, convolution, delta };

  /**
   * possible sensor distributions
   */
  enum class SensorDistribution { grid, cube_boundary };

  /**
   * possible measurement types
   */
  enum class MeasureType { discretized_function, vector };

  /**
   * possible transforms
   */
  enum class TransformType { identity, log };

  /**
   * possible norms
   */
  enum class NormType { vector, l2l2, h1l2, h2l2, h1h1, h2l2plusl2h1 };

  std::shared_ptr<ParameterHandler> prm;

  std::string log_file;
  int log_file_depth;
  int log_file_depth_mpi;
  int log_console_depth;

  int dimension;
  int fe_degree;
  int quad_order;

  double end_time;
  int initial_refines;
  size_t initial_time_steps;

  double dt;
  std::vector<double> times;

  double epsilon;
  double tau;
  ProblemType problem_type;
  NonlinearMethod method;
  TransformType transform;
  NormType norm_domain;
  NormType norm_codomain;

  double norm_domain_p;
  bool norm_domain_enable_wrapping;

  double norm_h1l2_alpha;

  double norm_h2l2_alpha;
  double norm_h2l2_beta;

  double norm_h1h1_alpha;
  double norm_h1h1_gamma;

  double norm_h2l2plusl2h1_alpha;
  double norm_h2l2plusl2h1_beta;
  double norm_h2l2plusl2h1_gamma;

  std::map<std::string, double> constants_for_exprs;
  std::string expr_initial_guess;
  std::string expr_param_q;
  std::string expr_param_nu;
  std::string expr_param_rho;
  std::string expr_param_c;
  std::string expr_param_background;
  double shape_scale;
  bool rho_dynamic;

  size_t num_rhs;
  std::vector<std::string> exprs_rhs;
  std::vector<size_t> configs;

  size_t synthesis_additional_refines;
  size_t synthesis_additional_fe_degrees;

  static const size_t num_configurations = 2;
  std::vector<Measure> measures;
  std::vector<SensorDistribution> sensor_distributions;
  std::vector<std::string> expr_masks;
  MeasureType measure_type;

  std::map<std::string, double> shape_options;
  MeshShape shape;

  /**
   * Declare all available Parameters.
   */
  static void declare_parameters(std::shared_ptr<ParameterHandler> prm);

  /**
   * Read general parameters and those for class WavePI from `prm`. `prm` is also stored for later access,
   * i.e. to construct instances of classes that are needed elsewhere.
   */
  void get_parameters(std::shared_ptr<ParameterHandler> prm);

  /**
   * Write all parameters to `deallog`.
   */
  void log_parameters();
};

} /* namespace wavepi */

#endif /* LIB_SETTINGSMANAGER_H_ */
