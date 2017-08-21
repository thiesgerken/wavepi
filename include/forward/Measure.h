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
#include <deal.II/base/parameter_handler.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>

#include <util/RadialParsedFunction.h>

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
       * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
       * @param delta_scale_space Desired support radius in space
       * @param delta_scale_time Desired support radius in time
       */
      PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
            const std::vector<Point<dim + 1>>& points, std::shared_ptr<Function<dim>> delta_shape,
            double delta_scale_space, double delta_scale_time);

      /**
       * Does not initialize most of the values, you have to use get_parameters afterwards and use `set_measurement_points`.
       *
       * @param solution_mesh Mesh that the functions that are measured live on, needed for `adjoint`.
       */
      PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh);

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

      virtual std::vector<double> evaluate(const DiscretizedFunction<dim>& field);

      virtual DiscretizedFunction<dim> adjoint(const std::vector<double>& measurements);

      const std::vector<Point<dim + 1> >& get_measurement_points() const {
         return measurement_points;
      }

      void set_measurement_points(const std::vector<Point<dim + 1> >& measurement_points) {
         this->measurement_points = measurement_points;
      }

   protected:
      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::vector<Point<dim + 1>> measurement_points;

      std::shared_ptr<Function<dim>> delta_shape;
      double delta_scale_space;
      double delta_scale_time;

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
       * @param spatial_points a vector with `dim` entries, each containing the coordinates for measurement points in that dimension.
       * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
       * @param delta_scale_space Desired support radius in space.
       * @param delta_scale_time Desired support radius in time.
       */
      GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh, const std::vector<double> &times,
            const std::vector<std::vector<double>> &spatial_points,
            std::shared_ptr<Function<dim>> delta_shape, double delta_scale_space, double delta_scale_time);

      /**
       * Does not initialize most of the values, you have to use get_parameters afterwards.
       *
       * @param solution_mesh Mesh that the functions that are measured live on, needed for `adjoint`.
       */
      GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh);

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);


   private:
      static std::vector<Point<dim + 1>> make_grid(const std::vector<double> &times,
            const std::vector<std::vector<double>> &spatial_points);

      static std::vector<double> make_points(const std::string description, bool is_time = false);

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_MEASURE_H_ */
