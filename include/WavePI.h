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

#include <problems/WaveProblem.h>

#include <measurements/Measure.h>
#include <SettingsManager.h>

#include <util/MacroFunctionParser.h>
#include <util/Tuple.h>

#include <memory>
#include <vector>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::util;
using namespace wavepi::measurements;
using namespace wavepi::problems;

template<int dim, typename Meas>
class WavePI {
      using Param = DiscretizedFunction<dim>;
      using Sol = DiscretizedFunction<dim>;
      using Exact = Function<dim>;

   public:
      /**
       * @param cfg Settings to use
       * @param measures Measures to use for the right hand sides, due to templating this class cannot instantiate them itself
       */
      WavePI(std::shared_ptr<SettingsManager> cfg);

      void run();

   private:
      std::shared_ptr<SettingsManager> cfg;
      std::vector<std::shared_ptr<Measure<Param, Meas>>> measures;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<WaveEquation<dim>> wave_eq;

      std::shared_ptr<WaveProblem<dim, Meas>> problem;

      std::shared_ptr<Function<dim>> param_exact;

      std::shared_ptr<MacroFunctionParser<dim>> initial_guess;
      std::shared_ptr<MacroFunctionParser<dim>> param_q;
      std::shared_ptr<MacroFunctionParser<dim>> param_nu;
      std::shared_ptr<MacroFunctionParser<dim>> param_a;
      std::shared_ptr<MacroFunctionParser<dim>> param_c;

      std::vector<std::shared_ptr<Function<dim>>> pulses;

      std::shared_ptr<Tuple<Meas>> data; // noisy data

      /**
       * `Point` constructor from three values, neglecting those that are not needed.
       */
      static Point<dim> make_point(double x, double y, double z);

      std::shared_ptr<Measure<Param, Meas>> get_measure(size_t config_idx);

      void initialize_mesh();
      void initialize_problem();
      void generate_data();
};

} /* namespace wavepi */

#endif /* INCLUDE_UTIL_WAVEPI_H_ */
