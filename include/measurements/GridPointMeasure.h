/*
 * GridPointMeasure.h
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_GRIDPOINTMEASURE_H_
#define INCLUDE_MEASUREMENTS_GRIDPOINTMEASURE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

#include <measurements/PointMeasure.h>

#include <memory>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::util;

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
       * @param times Points in time where you want those point measurements.
       * @param spatial_points a vector with `dim` entries, each containing the coordinates for measurement points in that dimension.
       * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
       * @param delta_scale_space Desired support radius in space.
       * @param delta_scale_time Desired support radius in time.
       */
      GridPointMeasure(const std::vector<double> &times,
            const std::vector<std::vector<double>> &spatial_points,
            std::shared_ptr<Function<dim+1>> delta_shape, double delta_scale_space, double delta_scale_time);

      /**
       * Does not initialize most of the values, you have to use `get_parameters` afterwards.
       *
       */
      GridPointMeasure();

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

   private:

      static std::vector<double> make_points(const std::string description, bool is_time = false);

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_GRIDPOINTMEASURE_H_ */
