/*
 * ConvolutionMeasure.h
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_DELTAMEASURE_H_
#define INCLUDE_MEASUREMENTS_DELTAMEASURE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include <base/DiscretizedFunction.h>
#include <base/LightFunction.h>
#include <base/SpaceTimeMesh.h>
#include <measurements/Measure.h>
#include <measurements/SensorDistribution.h>
#include <measurements/SensorValues.h>

#include <list>
#include <memory>
#include <utility>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::base;

/**
 * Point measurements, implemented by using the nearest vertices to the respective point
 */
template <int dim>
class DeltaMeasure : public Measure<DiscretizedFunction<dim>, SensorValues<dim>> {
 public:
  virtual ~DeltaMeasure() = default;

  /**
   * @param points Points in space and time (last dimension is time) where you want those point measurements.
   * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
   * @param delta_scale_space Desired support radius in space
   * @param delta_scale_time Desired support radius in time
   */
  DeltaMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh, std::shared_ptr<SensorDistribution<dim>> points, Norm norm);

  static void declare_parameters(ParameterHandler& prm);
  void get_parameters(ParameterHandler& prm);

  virtual SensorValues<dim> zero() override;

  virtual SensorValues<dim> evaluate(const DiscretizedFunction<dim>& field) override;

  /**
   * Adjoint, discretized on the mesh last used for evaluate
   */
  virtual DiscretizedFunction<dim> adjoint(const SensorValues<dim>& measurements) override;

 private:
  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  std::shared_ptr<SensorDistribution<dim>> sensor_distribution;
  Norm norm;
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_DELTAMEASURE_H_ */
