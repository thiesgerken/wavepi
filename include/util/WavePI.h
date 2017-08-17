/*
 * WavePI.h
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_WAVEPI_H_
#define INCLUDE_UTIL_WAVEPI_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>
#include <deal.II/base/function_parser.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>

#include <inversion/NonlinearProblem.h>

#include <memory>
#include <string>

namespace wavepi {
namespace util {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
class WavePI {
   public:
      static const std::string KEY_GENERAL;
      static const std::string KEY_DIMENSION;
      static const std::string KEY_FE_DEGREE;
      static const std::string KEY_QUAD_ORDER;

      static const std::string KEY_MESH;
      static const std::string KEY_END_TIME;
      static const std::string KEY_INITIAL_REFINES;
      static const std::string KEY_INITIAL_TIME_STEPS;

      static const std::string KEY_PROBLEM;
      static const std::string KEY_PROBLEM_TYPE;
      static const std::string KEY_PROBLEM_EPSILON;
      static const std::string KEY_PROBLEM_CONSTANTS;
      static const std::string KEY_PROBLEM_NUM_RHS;
      static const std::string KEY_PROBLEM_RHS;
      static const std::string KEY_PROBLEM_GUESS;
      static const std::string KEY_PROBLEM_PARAM_A;
      static const std::string KEY_PROBLEM_PARAM_Q;
      static const std::string KEY_PROBLEM_PARAM_C;
      static const std::string KEY_PROBLEM_PARAM_NU;
      static const std::string KEY_INVERSION;
      static const std::string KEY_INVERSION_METHOD;
      static const std::string KEY_INVERSION_TAU;

      static void declare_parameters(ParameterHandler &prm);

      WavePI(std::shared_ptr<ParameterHandler> prm);

      void run();

      void initialize_mesh();
      void initialize_problem();
      void generate_data();

   private:

      // possible problems
      enum class ProblemType {
         L2Q = 1, L2A = 2, L2C = 3, L2Nu = 4
      };

      // possible nonlinear methods
      enum class NonlinearMethod {
         REGINN = 1, NonlinearLandweber = 2
      };

      std::shared_ptr<ParameterHandler> prm;

      int fe_degree;
      int quad_order;

      double end_time;
      int initial_refines;
      int initial_time_steps;

      double epsilon;
      double tau;
      ProblemType problem_type;
      NonlinearMethod method;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<WaveEquation<dim>> wave_eq;

      using Param = DiscretizedFunction<dim>;
      using Sol = DiscretizedFunction<dim>;

      std::shared_ptr<NonlinearProblem<Param, Sol>> problem;

      std::shared_ptr<Function<dim>> param_exact_cont;
      std::shared_ptr<Param> param_exact;

      std::shared_ptr<FunctionParser<dim>> initial_guess;
      std::shared_ptr<FunctionParser<dim>> param_q;
      std::shared_ptr<FunctionParser<dim>> param_nu;
      std::shared_ptr<FunctionParser<dim>> param_a;
      std::shared_ptr<FunctionParser<dim>> param_c;
      std::shared_ptr<FunctionParser<dim>> rhs;

      std::shared_ptr<Sol> data; // noisy data
};

} /* namespace util */
} /* namespace wavepi */

#endif /* INCLUDE_UTIL_WAVEPI_H_ */
