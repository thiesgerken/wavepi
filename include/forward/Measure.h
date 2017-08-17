/*
 * Measurement.h
 *
 *  Created on: 17.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_FORWARD_MEASURE_H_
#define INCLUDE_FORWARD_MEASURE_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>
#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <memory>
#include <vector>

namespace wavepi {
namespace forward {

using namespace dealii;

/**
 * Interface for linear Measurements. `Measurement` will most likely be `std::vector<double>`.
 * The type `Sol` is the type of the continuous things this class measures.
 */
template<typename Sol, typename Measurement>
class Measure {
   public:

      virtual ~Measure() = default;

      /**
       * Compute a measurement.
       */
      virtual Measurement evaluate(const Sol& field) = 0;

      /**
       * Compute the adjoint of what `evaluate` does.
       * This function should ask `Sol` and `Measurement` for the required spaces.
       */
      virtual Sol adjoint(const Measurement& measurements) = 0;

};

/**
 * No real measurement, just returns the argument.
 * This way one can still reconstruct from the whole field even when the rest of the code expects measurement operators.
 */
template<typename Sol>
class IdenticalMeasure: public Measure<Sol, Sol> {
   public:

      virtual ~IdenticalMeasure() = default;

      virtual Sol evaluate(const Sol& field) {
         return field;
      }

      virtual Sol adjoint(const Sol& measurements) {
         return measurements;
      }

};

/**
 * Point measurements, implemented as scalar product between the given field and a delta-approximating function.
 */
template<int dim>
class PointMeasure: public Measure<DiscretizedFunction<dim>, std::vector<double>> {
   public:

      virtual ~PointMeasure() = default;

      /**
       * @param solution_mesh Mesh that the functions that are measured live on, needed for `adjoint`.
       * @param points Points in space and time (last dimension is time) where you want those point measurements.
       */
      PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
            const std::vector<Point<dim + 1>>& points);

      virtual std::vector<double> evaluate(const DiscretizedFunction<dim>& field);

      virtual DiscretizedFunction<dim> adjoint(const std::vector<double>& measurements);
   protected:
      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::vector<Point<dim + 1>> measurement_points;

};

/**
 * Point measurements on a grid (in space and time),
 * implemented as scalar product between the given field and a delta-approximating function.
 */
template<int dim>
class GridPointMeasure: public PointMeasure<dim> {
   public:
      virtual ~GridPointMeasure() = default;

      /**
       * Construct a grid for point measurements, given a temporal and spatial coordinates.
       * The resulting grid will have `times.size() * prod_{i=1}^{dim} spatial_points[i].size()` measurement points.
       *
       * @param solution_mesh Mesh that the functions that are measured live on, needed for `adjoint`.
       * @param times Points in time where you want those point measurements.
       * @param spatial_points a vector with `dim` entries, each containing the coordinates for measurment points in that dimension.
       */
      GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh, const std::vector<double> &times,
            const std::vector<std::vector<double>> &spatial_points);

   private:
      static std::vector<Point<dim + 1>> make_grid(const std::vector<double> &times,
            const std::vector<std::vector<double>> &spatial_points);
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_MEASURE_H_ */
