/*
 * SpaceTimeGrid.cpp
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#include <util/SpaceTimeGrid.h>

namespace wavepi {
namespace util {

template <int dim>
SpaceTimeGrid<dim>::SpaceTimeGrid(const std::vector<double> &times, const std::vector<std::vector<Point<dim>>> &points)
    : times(times), points(points), grid_extents() {
  AssertThrow(times.size() == points.size(), ExcDimensionMismatch(times.size(), points.size()));
  make_space_time_points();
}

template <int dim>
void SpaceTimeGrid<dim>::make_space_time_points() {
  size_t size_cache = 0;

  for (auto x : points) size_cache += x.size();

  space_time_points.clear();
  space_time_points.reserve(size_cache);

  for (size_t i = 0; i < times.size(); i++)
    for (auto x : points[i]) {
      Point<dim + 1> pt;

      for (size_t d = 0; d < dim; d++) pt[d] = x[d];

      pt[dim] = times[i];

      space_time_points.push_back(pt);
    }
}

template <>
SpaceTimeGrid<1>::SpaceTimeGrid(const std::vector<double> &times,
                                const std::vector<std::vector<double>> &spatial_points) {
  Assert(spatial_points.size() == 1, ExcInternalError());
  this->times      = times;
  size_t nb_points = spatial_points[0].size();

  std::vector<Point<1>> points_per_time(nb_points);

  for (size_t ix = 0; ix < spatial_points[0].size(); ix++) points_per_time[ix] = Point<1>(spatial_points[0][ix]);

  this->points = std::vector<std::vector<Point<1>>>(times.size());

  for (size_t i = 0; i < times.size(); i++) points[i] = points_per_time;

  this->grid_extents = spatial_points;
  make_space_time_points();
}

template <>
SpaceTimeGrid<2>::SpaceTimeGrid(const std::vector<double> &times,
                                const std::vector<std::vector<double>> &spatial_points) {
  Assert(spatial_points.size() == 2, ExcInternalError());
  this->times      = times;
  size_t nb_points = spatial_points[0].size() * spatial_points[1].size();

  std::vector<Point<2>> points_per_time(nb_points);

  for (size_t ix = 0; ix < spatial_points[0].size(); ix++)
    for (size_t iy = 0; iy < spatial_points[1].size(); iy++)
      points_per_time[ix * spatial_points[1].size() + iy] = Point<2>(spatial_points[0][ix], spatial_points[1][iy]);

  this->points = std::vector<std::vector<Point<2>>>(times.size());

  for (size_t i = 0; i < times.size(); i++) points[i] = points_per_time;

  this->grid_extents = spatial_points;
  make_space_time_points();
}

template <>
SpaceTimeGrid<3>::SpaceTimeGrid(const std::vector<double> &times,
                                const std::vector<std::vector<double>> &spatial_points) {
  Assert(spatial_points.size() == 3, ExcInternalError());
  this->times      = times;
  size_t nb_points = spatial_points[0].size() * spatial_points[1].size() * spatial_points[2].size();

  std::vector<Point<3>> points_per_time(nb_points);

  for (size_t ix = 0; ix < spatial_points[0].size(); ix++)
    for (size_t iy = 0; iy < spatial_points[1].size(); iy++)
      for (size_t iz = 0; iz < spatial_points[2].size(); iz++)
        points_per_time[ix * spatial_points[1].size() * spatial_points[2].size() + iy * spatial_points[2].size() + iz] =
            Point<3>(spatial_points[0][ix], spatial_points[1][iy], spatial_points[2][iz]);

  this->points = std::vector<std::vector<Point<3>>>(times.size());

  for (size_t i = 0; i < times.size(); i++) points[i] = points_per_time;

  this->grid_extents = spatial_points;
  make_space_time_points();
}

template class SpaceTimeGrid<1>;
template class SpaceTimeGrid<2>;
template class SpaceTimeGrid<3>;

} /* namespace util */
} /* namespace wavepi */
