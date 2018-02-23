/*
 * CubeBoundaryDistribution.cpp
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <measurements/CubeBoundaryDistribution.h>
#include <measurements/GridDistribution.h>
#include <measurements/SensorDistribution.h>

#include <cstdio>
#include <fstream>
#include <iostream>

namespace wavepi {
namespace measurements {
using namespace dealii;

template <>
void CubeBoundaryDistribution<1>::update_grid(const std::vector<double> &times,
                                              const std::vector<std::vector<double>> &points_per_dim) {
  Assert(points_per_dim.size() == 1, ExcInternalError());

  this->points_per_dim = points_per_dim;

  size_t nX = points_per_dim[0].size();

  std::vector<Point<1>> points_each_time;
  points_each_time.reserve(2);

  points_each_time.emplace_back(points_per_dim[0][0]);
  points_each_time.emplace_back(points_per_dim[0][nX - 1]);

  std::vector<std::vector<Point<1>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(points_each_time.size());

  for (size_t ix = 0; ix < points_each_time.size(); ix++)
    times_per_point.push_back(times);
}

template <>
void CubeBoundaryDistribution<2>::update_grid(const std::vector<double> &times,
                                              const std::vector<std::vector<double>> &points_per_dim) {
  Assert(points_per_dim.size() == 2, ExcInternalError());

  this->points_per_dim = points_per_dim;

  size_t nX = points_per_dim[0].size();
  size_t nY = points_per_dim[1].size();

  std::vector<Point<2>> points_each_time;
  points_each_time.reserve(2 * (nX + nY));

  for (size_t ix = 0; ix < nX; ix++)
    for (size_t iy = 0; iy < nY; iy++)
      if (ix == 0 || ix == nX - 1 || iy == 0 || iy == nY - 1)
        points_each_time.emplace_back(points_per_dim[0][ix], points_per_dim[1][iy]);

  std::vector<std::vector<Point<2>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(points_each_time.size());

  for (size_t ix = 0; ix < points_each_time.size(); ix++)
    times_per_point.push_back(times);
}

template <>
void CubeBoundaryDistribution<3>::update_grid(const std::vector<double> &times,
                                              const std::vector<std::vector<double>> &points_per_dim) {
  Assert(points_per_dim.size() == 3, ExcInternalError());

  this->points_per_dim = points_per_dim;

  size_t nX = points_per_dim[0].size();
  size_t nY = points_per_dim[1].size();
  size_t nZ = points_per_dim[2].size();

  std::vector<Point<3>> points_each_time;
  points_each_time.reserve(2 * (nX * nY + nX * nZ + nY * nZ));

  for (size_t ix = 0; ix < nX; ix++)
    for (size_t iy = 0; iy < nY; iy++)
      for (size_t iz = 0; iz < nZ; iz++)
        if (ix == 0 || ix == nX - 1 || iy == 0 || iy == nY - 1 || iz == 0 || iz == nZ - 1)
          points_each_time.emplace_back(points_per_dim[0][ix], points_per_dim[1][iy], points_per_dim[2][iz]);

  std::vector<std::vector<Point<3>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(points_each_time.size());

  for (size_t ix = 0; ix < points_each_time.size(); ix++)
    times_per_point.push_back(times);
}

template <int dim>
CubeBoundaryDistribution<dim>::CubeBoundaryDistribution(const std::vector<double> &times,
                                                        const std::vector<std::vector<double>> &points_per_dim) {
  update_grid(times, points_per_dim);
}

template <int dim>
void CubeBoundaryDistribution<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("CubeBoundaryDistribution");
  {
    prm.declare_entry("points x", "-0.9:10:0.9", Patterns::Anything(),
                      "points for the grid in x-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are inclusive, so they define the margin to the boundary of the cube.");
    prm.declare_entry("points y", "-0.9:10:0.9", Patterns::Anything(),
                      "points for the grid in y-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are inclusive, so they define the margin to the boundary of the cube.");
    prm.declare_entry("points z", "-0.9:10:0.9", Patterns::Anything(),
                      "points for the grid in z-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are inclusive, so they define the margin to the boundary of the cube.");
    prm.declare_entry("points t", "0:10:6", Patterns::Anything(),
                      "points for the grid in time. Format: '[lower bound]:[n_points]:[ub]'. Upper bound is inclusive, "
                      "lower bound is exclusive iff it equals 0.0.");
  }
  prm.leave_subsection();
}

template <int dim>
void CubeBoundaryDistribution<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("CubeBoundaryDistribution");
  {
    AssertThrow(0 <= dim && dim <= 3, ExcInternalError());

    std::vector<std::vector<double>> spatial_points;
    spatial_points.emplace_back(parse_description(prm.get("points x")));
    if (dim > 1) spatial_points.emplace_back(parse_description(prm.get("points y")));
    if (dim > 2) spatial_points.emplace_back(parse_description(prm.get("points z")));

    auto temporal_points = parse_description(prm.get("points t"), true);
    update_grid(temporal_points, spatial_points);
  }
  prm.leave_subsection();
}

template <int dim>
std::vector<double> CubeBoundaryDistribution<dim>::parse_description(const std::string description, bool is_time) {
  double lb, ub;
  size_t nb;

  AssertThrow(std::sscanf(description.c_str(), " %lf : %zu : %lf ", &lb, &nb, &ub) == 3,
              ExcMessage("Could not parse points"));
  AssertThrow((is_time && nb > 1 && lb < ub && lb >= 0.0) || (!is_time && nb >= 1 && lb <= ub),
              ExcMessage("Illegal interval spec: " + description));

  std::vector<double> points(nb);

  if (lb == 0.0 && is_time)  // lb excl, ub incl
    for (size_t i = 0; i < nb; i++)
      points[i] = lb + (i + 1) * (ub - lb) / nb;
  else
    // lb excl, ub excl
    for (size_t i = 0; i < nb; i++)
      points[i] = lb + (i + 1) * (ub - lb) / (nb + 1);

  return points;
}

template <int dim>
size_t CubeBoundaryDistribution<dim>::index_times_per_point(size_t point_index, size_t time_index) {
  return time_index * this->points_per_time[0].size() + point_index;
}

template <int dim>
void CubeBoundaryDistribution<dim>::write_pvd(const std::vector<double> &values, std::string path, std::string filename,
                                              std::string name) {
  GridDistribution<dim> grid(this->times, points_per_dim);

  auto grid_values = spread_to_grid(values);
  grid.write_pvd(grid_values, path, filename, name);
}

template <>
std::vector<double> CubeBoundaryDistribution<1>::spread_to_grid(const std::vector<double> &values) {
  size_t nX = points_per_dim[0].size();

  std::vector<double> grid_values;
  grid_values.reserve(nX);
  size_t idx = 0;

  for (size_t it = 0; it < times.size(); it++)
    for (size_t ix = 0; ix < nX; ix++)
      if (ix == 0 || ix == nX - 1)
        grid_values.push_back(values[idx++]);
      else
        grid_values.push_back(0.0);

  return grid_values;
}

template <>
std::vector<double> CubeBoundaryDistribution<2>::spread_to_grid(const std::vector<double> &values) {
  size_t nX = points_per_dim[0].size();
  size_t nY = points_per_dim[1].size();

  std::vector<double> grid_values;
  grid_values.reserve(nX * nY);
  size_t idx = 0;

  for (size_t it = 0; it < times.size(); it++)
    for (size_t ix = 0; ix < nX; ix++)
      for (size_t iy = 0; iy < nY; iy++)
        if (ix == 0 || ix == nX - 1 || iy == 0 || iy == nY - 1)
          grid_values.push_back(values[idx++]);
        else
          grid_values.push_back(0.0);

  return grid_values;
}

template <>
std::vector<double> CubeBoundaryDistribution<3>::spread_to_grid(const std::vector<double> &values) {
  size_t nX = points_per_dim[0].size();
  size_t nY = points_per_dim[1].size();
  size_t nZ = points_per_dim[2].size();

  std::vector<double> grid_values;
  grid_values.reserve(nX * nY * nZ);
  size_t idx = 0;

  for (size_t it = 0; it < times.size(); it++)
    for (size_t ix = 0; ix < nX; ix++)
      for (size_t iy = 0; iy < nY; iy++)
        for (size_t iz = 0; iz < nZ; iz++)
          if (ix == 0 || ix == nX - 1 || iy == 0 || iy == nY - 1 || iz == 0 || iz == nZ - 1)
            grid_values.push_back(values[idx++]);
          else
            grid_values.push_back(0.0);

  return grid_values;
}

template class CubeBoundaryDistribution<1>;
template class CubeBoundaryDistribution<2>;
template class CubeBoundaryDistribution<3>;

} /* namespace measurements */
} /* namespace wavepi */
