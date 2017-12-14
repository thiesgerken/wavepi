/*
 * SettingsManager.h
 *
 *  Created on: 22.08.2017
 *      Author: thies
 */

#ifndef LIB_SETTINGSMANAGER_H_
#define LIB_SETTINGSMANAGER_H_

#include <deal.II/base/parameter_handler.h>

#include <stddef.h>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace wavepi {
using namespace dealii;

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
      static const std::string KEY_PROBLEM_EPSILON;
      static const std::string KEY_PROBLEM_CONSTANTS;
      static const std::string KEY_PROBLEM_GUESS;
      static const std::string KEY_PROBLEM_PARAM_A;
      static const std::string KEY_PROBLEM_PARAM_Q;
      static const std::string KEY_PROBLEM_PARAM_C;
      static const std::string KEY_PROBLEM_PARAM_NU;

      static const std::string KEY_INVERSION;
      static const std::string KEY_INVERSION_METHOD;
      static const std::string KEY_INVERSION_TAU;

      static const std::string KEY_PROBLEM_DATA;
      static const std::string KEY_PROBLEM_DATA_COUNT;
      static const std::string KEY_PROBLEM_DATA_RHS;
      static const std::string KEY_PROBLEM_DATA_CONFIG;
      static const std::string KEY_PROBLEM_DATA_I;
      static const std::string KEY_PROBLEM_DATA_I_MEASURE;

      /**
       * possible problems
       */
      enum class ProblemType {
         L2Q = 1, L2A = 2, L2C = 3, L2Nu = 4
      };

      /**
       * possible nonlinear methods
       */
      enum class NonlinearMethod {
         REGINN = 1, NonlinearLandweber = 2
      };

      /**
       * possible mesh shapes
       */
      enum class MeshShape {
         hyper_cube = 1, hyper_L = 2, hyper_ball = 3, cheese = 4
      };

      /**
       * possible measurement operators
       */
      enum class Measure {
         identical = 1, grid = 2
      };

      /**
       * possible measurement types
       */
      enum class MeasureType {
         discretized_function = 1, vector = 2
      };

      std::shared_ptr<ParameterHandler> prm;

      std::string log_file;
      int log_file_depth;
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

      std::map<std::string, double> constants_for_exprs;
      std::string expr_initial_guess;
      std::string expr_param_q;
      std::string expr_param_nu;
      std::string expr_param_a;
      std::string expr_param_c;

      size_t num_rhs;
      std::vector<std::string> exprs_rhs;
      std::vector<size_t> configs;

      static const size_t num_configurations = 2;
      std::vector<Measure> measures;
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