/*
 * CubeBoundaryDistribution.cpp
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <measurements/CubeBoundaryDistribution.h>
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
  // TODO
  AssertThrow(false, ExcNotImplemented());

  this->points_per_dim = points_per_dim;

  Assert(points_per_dim.size() == 1, ExcInternalError());
  size_t nb_points = points_per_dim[0].size();

  std::vector<Point<1>> points_each_time(nb_points);

  for (size_t ix = 0; ix < points_per_dim[0].size(); ix++)
    points_each_time[ix] = Point<1>(points_per_dim[0][ix]);

  std::vector<std::vector<Point<1>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(nb_points);

  for (size_t ix = 0; ix < nb_points; ix++)
    times_per_point.push_back(times);
}

template <>
void CubeBoundaryDistribution<2>::update_grid(const std::vector<double> &times,
                                              const std::vector<std::vector<double>> &points_per_dim) {
  // TODO
  AssertThrow(false, ExcNotImplemented());

  this->points_per_dim = points_per_dim;

  Assert(points_per_dim.size() == 2, ExcInternalError());
  size_t nb_points = points_per_dim[0].size() * points_per_dim[1].size();

  std::vector<Point<2>> points_each_time(nb_points);

  for (size_t ix = 0; ix < points_per_dim[0].size(); ix++)
    for (size_t iy = 0; iy < points_per_dim[1].size(); iy++)
      points_each_time[ix * points_per_dim[1].size() + iy] = Point<2>(points_per_dim[0][ix], points_per_dim[1][iy]);

  std::vector<std::vector<Point<2>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(nb_points);

  for (size_t ix = 0; ix < nb_points; ix++)
    times_per_point.push_back(times);
}

template <>
void CubeBoundaryDistribution<3>::update_grid(const std::vector<double> &times,
                                              const std::vector<std::vector<double>> &points_per_dim) {
  // TODO
  AssertThrow(false, ExcNotImplemented());

  this->points_per_dim = points_per_dim;

  Assert(points_per_dim.size() == 3, ExcInternalError());
  size_t nb_points = points_per_dim[0].size() * points_per_dim[1].size() * points_per_dim[2].size();

  std::vector<Point<3>> points_each_time(nb_points);

  for (size_t ix = 0; ix < points_per_dim[0].size(); ix++)
    for (size_t iy = 0; iy < points_per_dim[1].size(); iy++)
      for (size_t iz = 0; iz < points_per_dim[2].size(); iz++)
        points_each_time[ix * points_per_dim[1].size() * points_per_dim[2].size() + iy * points_per_dim[2].size() +
                         iz] = Point<3>(points_per_dim[0][ix], points_per_dim[1][iy], points_per_dim[2][iz]);

  std::vector<std::vector<Point<3>>> points_per_time(times.size());

  for (size_t i = 0; i < times.size(); i++)
    points_per_time[i] = points_each_time;

  update_points(times, points_per_time);

  // update times_per_point and points now (cleared by update_points!)
  points = points_each_time;
  times_per_point.reserve(nb_points);

  for (size_t ix = 0; ix < nb_points; ix++)
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
    // TODO
    AssertThrow(false, ExcNotImplemented());
  }
  prm.leave_subsection();
}

template <int dim>
void CubeBoundaryDistribution<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("CubeBoundaryDistribution");
  {
    AssertThrow(0 <= dim && dim <= 3, ExcInternalError());

    // TODO
    AssertThrow(false, ExcNotImplemented());

    // need
    //   * margin(s) to the boundary
    //   * number of points per dimension
    //   * cube size
    // could also use the notation of GridDistribution? ...
  }
  prm.leave_subsection();
}

template <int dim>
size_t CubeBoundaryDistribution<dim>::index_times_per_point(size_t point_index, size_t time_index) {
  // TODO
  // return time_index * this->points_per_time[0].size() + point_index;
  AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void CubeBoundaryDistribution<dim>::write_pvd(const std::vector<double> &values, std::string path, std::string filename,
                                              std::string name) {
  // TODO
  AssertThrow(false, ExcNotImplemented());
}

template class CubeBoundaryDistribution<1>;
template class CubeBoundaryDistribution<2>;
template class CubeBoundaryDistribution<3>;

} /* namespace measurements */
} /* namespace wavepi */
