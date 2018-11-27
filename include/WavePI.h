/*
 * WavePI.h
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_WAVEPI_H_
#define INCLUDE_WAVEPI_H_

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <base/DiscretizedFunction.h>
#include <base/MacroFunctionParser.h>
#include <base/SpaceTimeMesh.h>
#include <base/Tuple.h>
#include <forward/WaveEquation.h>
#include <measurements/Measure.h>
#include <problems/WaveProblem.h>

#include <SettingsManager.h>

#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::measurements;
using namespace wavepi::problems;

template <int dim, typename Meas>
class WavePI {
  using Param = DiscretizedFunction<dim>;
  using Sol   = DiscretizedFunction<dim>;
  using Exact = Function<dim>;

 public:
  /**
   * @param cfg Settings to use
   * @param measures Measures to use for the right hand sides, due to templating this class cannot instantiate them
   * itself
   */
  WavePI(std::shared_ptr<SettingsManager> cfg);

  /**
   * discretize the initial guess, initialize the inversion scheme and run it
   */
  void run();

 private:
  std::shared_ptr<SettingsManager> cfg;
  std::vector<std::shared_ptr<Measure<Param, Meas>>> measures;

  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain;

  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_vector;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_l2l2;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_h1l2;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_h2l2;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_h1h1;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_h2l2plusl2h1;

  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  std::shared_ptr<WaveEquation<dim>> wave_eq;

  std::shared_ptr<WaveProblem<dim, Meas>> problem;
  std::shared_ptr<Transformation<dim>> transform;

  std::shared_ptr<LightFunction<dim>> param_exact;
  std::shared_ptr<LightFunction<dim>> param_exact_untransformed;

  std::shared_ptr<MacroFunctionParser<dim>> initial_guess;
  std::shared_ptr<MacroFunctionParser<dim>> param_q;
  std::shared_ptr<MacroFunctionParser<dim>> param_nu;
  std::shared_ptr<MacroFunctionParser<dim>> param_rho;
  std::shared_ptr<MacroFunctionParser<dim>> param_c;

  std::shared_ptr<MacroFunctionParser<dim>> param_background;

  std::vector<std::shared_ptr<Function<dim>>> pulses;

  std::shared_ptr<Tuple<Meas>> data;  // noisy data

  /**
   * convenience `Point` constructor from three values, neglecting those that are not needed.
   */
  static Point<dim> make_point(double x, double y, double z);

  /**
   * ignore the additional time steps and interpolate at original time steps (they must exist!) to new mesh
   */
  std::shared_ptr<Tuple<DiscretizedFunction<dim>>> interpolate_field(
      std::shared_ptr<SpaceTimeMesh<dim>> target_mesh, std::shared_ptr<Tuple<DiscretizedFunction<dim>>> data) const;

  std::shared_ptr<Measure<Param, Meas>> get_measure(size_t config_idx, std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                                    std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm);

  /**
   * create a mesh based on `cfg` and return it.
   *
   * @param additional_refines additional global refines to use in space and time domain
   * @param additional_fe_degrees number to add on `cfg->fe_degree`
   */
  std::shared_ptr<SpaceTimeMesh<dim>> initialize_mesh(size_t additional_refines    = 0,
                                                      size_t additional_fe_degrees = 0) const;

  /**
   * initializes `wave_eq`, `problem`, `measures`, `pulses`, `param_exact`, `param_exact_untransformed` and `transform`.
   * Has to be called after `initialize_mesh` was used to create `this->mesh`.
   */
  void initialize_problem();

  /**
   * initializes `data`. Has to be called after `initialize_problem`.
   */
  void synthesize_data();

  /**
   * modifies `data` and interpolates it onto the given mesh.
   * This is only necessary if the field itself is used, otherwise this function does nothing.
   */
  void interpolate_data(std::shared_ptr<SpaceTimeMesh<dim>> target_mesh);

  void log_error(DiscretizedFunction<dim>& reconstruction, std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm);
  void log_error_initial(DiscretizedFunction<dim>& reconstruction_minus_initial,
                         std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm,
                         DiscretizedFunction<dim>& exact_minus_initial);
};

} /* namespace wavepi */

#endif /* INCLUDE_UTIL_WAVEPI_H_ */
