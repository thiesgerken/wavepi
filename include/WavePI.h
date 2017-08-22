/*
 * WavePI.h
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_WAVEPI_H_
#define INCLUDE_WAVEPI_H_

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>

#include <inversion/NonlinearProblem.h>

#include <SettingsManager.h>

#include <util/MacroFunctionParser.h>

#include <memory>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::util;

template<int dim>
class WavePI {
   public:
      WavePI(std::shared_ptr<SettingsManager> cfg);

      void run();

      void initialize_mesh();
      void initialize_problem();
      void generate_data();

   private:

      std::shared_ptr<SettingsManager> cfg;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<WaveEquation<dim>> wave_eq;

      using Param = DiscretizedFunction<dim>;
      using Sol = DiscretizedFunction<dim>;
      using Exact = Function<dim>;

      std::shared_ptr<NonlinearProblem<Param, Sol>> problem;

      std::shared_ptr<Function<dim>> param_exact;

      std::shared_ptr<MacroFunctionParser<dim>> initial_guess;
      std::shared_ptr<MacroFunctionParser<dim>> param_q;
      std::shared_ptr<MacroFunctionParser<dim>> param_nu;
      std::shared_ptr<MacroFunctionParser<dim>> param_a;
      std::shared_ptr<MacroFunctionParser<dim>> param_c;
      std::shared_ptr<MacroFunctionParser<dim>> rhs;

      std::shared_ptr<Sol> data; // noisy data

      /**
       * `Point` constructor from three values, neglecting those that are not needed.
       */
      static Point<dim> make_point(double x, double y, double z);
};

} /* namespace wavepi */

#endif /* INCLUDE_UTIL_WAVEPI_H_ */
