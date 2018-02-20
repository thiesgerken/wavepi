/*
 * SensorDistribution.h
 *
 *  Created on: 20.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_SENSORDISTRIBUTION_H_
#define INCLUDE_MEASUREMENTS_SENSORDISTRIBUTION_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <stddef.h>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {
using namespace dealii;

/**
 * class that supplies locations of sensors in space and time
 */
template <int dim>
class SensorDistribution {
 public:
  virtual ~SensorDistribution() = default;

  /**
   * @param times times used in this grid in ascending order.
   * @param spatial points at each time step (same length as `times`, inner vector may be differently sized)
   */
  SensorDistribution(const std::vector<double>& times, const std::vector<std::vector<Point<dim>>>& points);

  SensorDistribution() = default;

  inline size_t size() const { return space_time_points.size(); }

  inline const Point<dim + 1>& operator[](const size_t i) const {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return space_time_points[i];
  }

  inline const std::vector<double>& get_times() const { return times; }

  inline const std::vector<Point<dim + 1>>& get_space_time_points() const { return space_time_points; }

  inline const std::vector<Point<dim>>& get_points(size_t time_index) const {
    Assert(time_index < times.size(), ExcIndexRange(time_index, 0, times.size()));
    return points[time_index];
  }

  virtual void write_pvd(const std::vector<double>& values __attribute((unused)),
                         std::string path __attribute((unused)), std::string filename __attribute((unused)),
                         std::string name __attribute((unused))) {
    // TODO: write_pvd for unstructured grids
    AssertThrow(false, ExcNotImplemented());
  }

 protected:
  std::vector<double> times;
  std::vector<std::vector<Point<dim>>> points;
  std::vector<Point<dim + 1>> space_time_points;  // waste of memory, but easier access

  void update_points(const std::vector<double>& times, const std::vector<std::vector<Point<dim>>>& points);
};

/**
 * Point measurements on a grid (in space and time),
 * implemented as scalar product between the given field and a delta-approximating function.
 */
template <int dim>
class GridDistribution : public SensorDistribution<dim> {
 public:
  virtual ~GridDistribution() = default;

  /**
   * Construct a grid for point measurements, given a temporal and spatial coordinates.
   * The resulting grid will have `times.size() * prod_{i=1}^{dim} spatial_points[i].size()` measurement points.
   *
   * @param times Points in time where you want those point measurements.
   * @param points_per_dim a vector with `dim` entries, each containing the coordinates for measurement points in that
   * dimension.
   */
  GridDistribution(const std::vector<double>& times, const std::vector<std::vector<double>>& points_per_dim);

  /**
   * Empty Grid
   */
  GridDistribution() = default;

  /**
   * Declare parameters used by `GridDistribution(ParameterHandler&)`
   */
  static void declare_parameters(ParameterHandler& prm);

  /**
   * Initialize using a settings file
   */
  void get_parameters(ParameterHandler& prm);

  virtual void write_pvd(const std::vector<double>& values, std::string path, std::string filename,
                         std::string name) override;

 private:
  // grid extents
  std::vector<std::vector<double>> points_per_dim;

  void update_grid(const std::vector<double>& times, const std::vector<std::vector<double>>& points_per_dim);
  static std::vector<double> parse_description(const std::string description, bool is_time = false);
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_SENSORDISTRIBUTION_H_ */
