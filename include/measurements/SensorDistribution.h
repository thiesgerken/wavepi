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
  SensorDistribution(const std::vector<double>& times, const std::vector<std::vector<Point<dim>>>& points_per_time);

  SensorDistribution() = default;

  /**
   * total number of points
   */
  inline size_t size() const { return space_time_points.size(); }

  /**
   * access every point, ordering given by `times()`.
   */
  inline const Point<dim + 1>& operator[](const size_t i) const {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return space_time_points[i];
  }

  /**
   * get times that are used in measurement points
   */
  inline const std::vector<double>& get_times() const { return times; }

  /**
   * get points that belong to one time (-index)
   */
  inline const std::vector<Point<dim>>& get_points_per_time(size_t time_index) const {
    Assert(time_index < times.size(), ExcIndexRange(time_index, 0, times.size()));
    return points_per_time[time_index];
  }

  virtual bool times_per_point_available() const { return !times.size() || points.size(); }

  virtual size_t index_times_per_point(size_t point_index __attribute((unused)),
                                       size_t time_index __attribute((unused))) {
    AssertThrow(times_per_point_available(), ExcMessage("!times_per_point_available()"));
    AssertThrow(false, ExcMessage("index_times_per_point() not implemented although times_per_point_available()"));
    return 0;
  }

  /**
   * get points that are used in measurement points. Might not be supported for every distribution, use
   * `times_per_point_available` before.
   */
  inline const std::vector<Point<dim>>& get_points() const {
    Assert(times_per_point_available(), ExcMessage("!times_per_point_available()"));
    return points;
  }

  /**
   * get times that belong to one point (-index). Might not be supported for every distribution, use
   * `times_per_point_available` before.
   */
  inline const std::vector<double>& get_times_per_point(size_t point_index) const {
    Assert(times_per_point_available(), ExcMessage("!times_per_point_available()"));
    Assert(point_index < points.size(), ExcIndexRange(point_index, 0, points.size()));
    return times_per_point[point_index];
  }

  virtual void write_pvd(const std::vector<double>& values __attribute((unused)),
                         std::string path __attribute((unused)), std::string filename __attribute((unused)),
                         std::string name __attribute((unused))) {
    // TODO: write_pvd for unstructured grids
    AssertThrow(false, ExcNotImplemented());
  }

 protected:
  // waste of memory, but easier access: different data structures depending on what the user wants to do

  std::vector<double> times;
  std::vector<std::vector<Point<dim>>> points_per_time;

  std::vector<Point<dim>> points;
  std::vector<std::vector<double>> times_per_point;

  std::vector<Point<dim + 1>> space_time_points;

  void update_points(const std::vector<double>& times, const std::vector<std::vector<Point<dim>>>& points_per_time);
};

/**
 * distribution on a grid in time and space
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

  virtual size_t index_times_per_point(size_t point_index, size_t time_index) override;

 private:
  // grid extents
  std::vector<std::vector<double>> points_per_dim;

  void update_grid(const std::vector<double>& times, const std::vector<std::vector<double>>& points_per_dim);
  static std::vector<double> parse_description(const std::string description, bool is_time = false);
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_SENSORDISTRIBUTION_H_ */
