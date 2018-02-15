/*
 * PointMeasure.h
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_POINTMEASURE_H_
#define INCLUDE_MEASUREMENTS_POINTMEASURE_H_

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
#include <base/SpaceTimeGrid.h>
#include <base/SpaceTimeMesh.h>
#include <measurements/Measure.h>

#include <list>
#include <memory>
#include <utility>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::base;

/**
 * Point measurements, implemented as scalar product between the given field and a delta-approximating function.
 */
template <int dim>
class PointMeasure : public Measure<DiscretizedFunction<dim>, MeasuredValues<dim>> {
 public:
  virtual ~PointMeasure() = default;

  /**
   * @param points Points in space and time (last dimension is time) where you want those point measurements.
   * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
   * @param delta_scale_space Desired support radius in space
   * @param delta_scale_time Desired support radius in time
   */
  PointMeasure(std::shared_ptr<SpaceTimeGrid<dim>> points, std::shared_ptr<LightFunction<dim>> delta_shape,
               double delta_scale_space, double delta_scale_time);

  /**
   * Does not initialize most of the values, you have to use get_parameters afterwards and use `set_measurement_points`.
   * *
   */
  PointMeasure();

  static void declare_parameters(ParameterHandler& prm);
  void get_parameters(ParameterHandler& prm);

  virtual MeasuredValues<dim> evaluate(const DiscretizedFunction<dim>& field);

  /**
   * Adjoint, discretized on the mesh last used for evaluate
   */
  virtual DiscretizedFunction<dim> adjoint(const MeasuredValues<dim>& measurements);

  const std::shared_ptr<SpaceTimeGrid<dim>>& get_measurement_points() const { return measurement_points; }

  void set_measurement_points(const std::shared_ptr<SpaceTimeGrid<dim>>& measurement_points) {
    this->measurement_points = measurement_points;
  }

  class HatShape : public LightFunction<dim> {
   public:
    virtual ~HatShape() = default;

    virtual double evaluate(const Point<dim + 1>& p) const {
      double nrm = 0.0;
      for (size_t i = 0; i < dim; i++)
        nrm += p[i] * p[i];

      return std::max(1 - sqrt(nrm), 0.0) * std::max(1 - p[dim], 0.0);
    }
  };

  class ConstShape : public LightFunction<dim> {
   public:
    virtual ~ConstShape() = default;

    virtual double evaluate(const Point<dim + 1>& p) const {
      double nrm = 0.0;
      for (size_t i = 0; i < dim; i++)
        nrm += p[i] * p[i];

      return nrm <= 1.0 && p[dim] <= 1.0 && p[dim] >= -1.0 ? 1.0 : 0.0;
    }
  };

  class LightFunctionWrapper : public Function<dim> {
   public:
    virtual ~LightFunctionWrapper() = default;
    LightFunctionWrapper(std::shared_ptr<LightFunction<dim>> base, double delta_scale_space, double delta_scale_time)
        : base(base), delta_scale_space(delta_scale_space), delta_scale_time(delta_scale_time) {}

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const {
      Assert(component == 0, ExcInternalError());
      Assert(base, ExcInternalError());

      Point<dim + 1> p1;

      p1(dim) = (this->get_time() - offset(dim)) / delta_scale_time;

      for (size_t d = 0; d < dim; d++)
        p1(d) = (p(d) - offset(d)) / delta_scale_space;

      return base->evaluate(p1);
    }

    void set_offset(const Point<dim + 1>& offset) { this->offset = offset; }

   private:
    std::shared_ptr<LightFunction<dim>> base;
    double delta_scale_space;
    double delta_scale_time;

    Point<dim + 1> offset;
  };

 protected:
  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  std::shared_ptr<SpaceTimeGrid<dim>> measurement_points;
  std::shared_ptr<LightFunction<dim>> delta_shape;

  double delta_scale_space;
  double delta_scale_time;

  /**
   * collect jobs so that we have to go through the mesh only once
   * for each time a list of sensor numbers and factors
   */
  std::vector<std::vector<std::pair<size_t, double>>> compute_jobs() const;
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_POINTMEASURE_H_ */
