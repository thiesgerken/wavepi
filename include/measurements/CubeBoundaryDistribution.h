/*
 * CubeBoundaryDistribution.h
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#ifndef LIB_MEASUREMENTS_CUBEBOUNDARYDISTRIBUTION_H_
#define LIB_MEASUREMENTS_CUBEBOUNDARYDISTRIBUTION_H_

#include <measurements/SensorDistribution.h>

namespace wavepi {
namespace measurements {

/**
 * distributes sensors at the boundary of a hyper-cube
 */
template <int dim>
class CubeBoundaryDistribution : public SensorDistribution<dim> {
 public:
  virtual ~CubeBoundaryDistribution() = default;

  /**
   * Construct a grid for point measurements, given a temporal and spatial coordinates.
   * The resulting grid will have `times.size() * prod_{i=1}^{dim} spatial_points[i].size()` measurement points.
   *
   * @param times Points in time where you want those point measurements.
   * @param points_per_dim a vector with `dim` entries, each containing the coordinates for measurement points in that
   * dimension.
   */
  CubeBoundaryDistribution(const std::vector<double>& times, const std::vector<std::vector<double>>& points_per_dim);

  /**
   * Empty Grid
   */
  CubeBoundaryDistribution() = default;

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
  // grid description
  std::vector<std::vector<double>> points_per_dim;

  void update_grid(const std::vector<double>& times, const std::vector<std::vector<double>>& points_per_dim);
  static std::vector<double> parse_description(const std::string description, bool is_time = false);
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* LIB_MEASUREMENTS_CUBEBOUNDARYDISTRIBUTION_H_ */
