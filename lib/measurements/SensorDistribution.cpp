/*
 * SensorDistribution.cpp
 *
 *  Created on: 20.02.2018
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <measurements/SensorDistribution.h>

#include <cstdio>
#include <fstream>
#include <iostream>

namespace wavepi {
namespace measurements {
using namespace dealii;

template <int dim>
SensorDistribution<dim>::SensorDistribution(const std::vector<double> &times,
                                            const std::vector<std::vector<Point<dim>>> &points_per_time) {
  update_points(times, points_per_time);
}

template <int dim>
void SensorDistribution<dim>::update_points(const std::vector<double> &times,
                                            const std::vector<std::vector<Point<dim>>> &points_per_time) {
  AssertThrow(times.size() == points_per_time.size(), ExcDimensionMismatch(times.size(), points_per_time.size()));
  size_t expected_size = 0;

  for (auto x : points_per_time)
    expected_size += x.size();

  space_time_points.clear();
  space_time_points.reserve(expected_size);

  for (size_t i = 0; i < times.size(); i++)
    for (auto x : points_per_time[i]) {
      Point<dim + 1> pt;

      for (size_t d = 0; d < dim; d++)
        pt[d] = x[d];

      pt[dim] = times[i];

      space_time_points.push_back(pt);
    }

  this->times           = times;
  this->points_per_time = points_per_time;

  this->points.clear();
  this->times_per_point.clear();
}

template class SensorDistribution<1>;
template class SensorDistribution<2>;
template class SensorDistribution<3>;

} /* namespace measurements */
} /* namespace wavepi */
