/*
 * GridDistribution.cpp
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/utilities.h>

#include <measurements/GridDistribution.h>
#include <measurements/SensorDistribution.h>

#include <cstdio>
#include <fstream>
#include <iostream>

namespace wavepi {
namespace measurements {
using namespace dealii;

template <>
void GridDistribution<1>::update_grid(const std::vector<double> &times,
                                      const std::vector<std::vector<double>> &points_per_dim) {
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
void GridDistribution<2>::update_grid(const std::vector<double> &times,
                                      const std::vector<std::vector<double>> &points_per_dim) {
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
void GridDistribution<3>::update_grid(const std::vector<double> &times,
                                      const std::vector<std::vector<double>> &points_per_dim) {
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
GridDistribution<dim>::GridDistribution(const std::vector<double> &times,
                                        const std::vector<std::vector<double>> &points_per_dim) {
  update_grid(times, points_per_dim);
}

template <int dim>
void GridDistribution<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("GridDistribution");
  {
    prm.declare_entry("points x", "-1:10:1", Patterns::Anything(),
                      "points for the grid in x-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are exclusive.");
    prm.declare_entry("points y", "-1:10:1", Patterns::Anything(),
                      "points for the grid in y-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are exclusive.");
    prm.declare_entry("points z", "-1:10:1", Patterns::Anything(),
                      "points for the grid in z-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound "
                      "are exclusive.");
    prm.declare_entry("points t", "0:10:6", Patterns::Anything(),
                      "points for the grid in time. Format: '[lower bound]:[n_points]:[ub]'. Upper bound is inclusive, "
                      "lower bound is exclusive iff it equals 0.0.");
  }
  prm.leave_subsection();
}

template <int dim>
void GridDistribution<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("GridDistribution");
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
std::vector<double> GridDistribution<dim>::parse_description(const std::string description, bool is_time) {
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
  else if (lb > 0.0 && is_time)  // lb incl, ub incl
    for (size_t i = 0; i < nb; i++)
      points[i] = lb + (i + 1) * (ub - lb) / (nb - 1);
  else
    // lb excl, ub excl
    for (size_t i = 0; i < nb; i++)
      points[i] = lb + (i + 1) * (ub - lb) / (nb + 1);

  return points;
}

template <int dim>
size_t GridDistribution<dim>::index_times_per_point(size_t point_index, size_t time_index) {
  return time_index * this->points_per_time[0].size() + point_index;
}

template <int dim>
void GridDistribution<dim>::write_pvd(const std::vector<double> &values, std::string path, std::string filename,
                                      std::string name) {
  size_t extentX           = points_per_dim[0].size() - 1;
  size_t extentY           = dim > 1 ? points_per_dim[1].size() - 1 : 0;
  size_t extentZ           = dim > 2 ? points_per_dim[2].size() - 1 : 0;
  size_t nb_spatial_points = this->points_per_time[0].size();

  std::ofstream fpvd(path + filename + ".pvd", std::ios::out | std::ios::trunc);
  fpvd.precision(8);
  fpvd << std::fixed << "<?xml version=\"1.0\"?>" << std::endl
       << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl
       << "<Collection>";

  for (size_t ti = 0; ti < this->times.size(); ti++) {
    fpvd << "<DataSet timestep=\"" << this->times[ti] << "\" group=\"\" part=\"0\" file=\"" << filename << "-"
         << Utilities::to_string(ti, 4) << ".vts\"/>" << std::endl;

    std::ofstream fvts(path + filename + "-" + Utilities::to_string(ti, 4) + ".vts", std::ios::out | std::ios::trunc);

    fvts << std::fixed
         << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
         << std::endl
         << "<StructuredGrid WholeExtent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">" << std::endl
         << "<Piece Extent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">" << std::endl
         << "<PointData Scalars=\"" << name << "\">" << std::endl
         << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">" << std::endl;

    fvts.precision(16);
    fvts << std::scientific;

    for (size_t i = 0; i < nb_spatial_points; i++)
      fvts << values[nb_spatial_points * ti + i] << " ";

    fvts << std::endl
         << "</DataArray>" << std::endl
         << "</PointData>" << std::endl
         << "<CellData></CellData>" << std::endl
         << "<Points>" << std::endl
         << "<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;

    fvts.precision(6);
    fvts << std::fixed;

    for (auto p : this->points_per_time[ti])
      fvts << p(0) << " " << (dim > 1 ? p(1) : 0.0) << " " << (dim > 2 ? p(2) : 0.0) << " ";

    fvts << std::endl
         << "</DataArray>" << std::endl
         << "</Points>" << std::endl
         << "</Piece>" << std::endl
         << "</StructuredGrid>" << std::endl
         << "</VTKFile>" << std::endl;
    fvts.close();
  }

  fpvd << "</Collection>" << std::endl << "</VTKFile>" << std::endl;
  fpvd.close();
}

template class GridDistribution<1>;
template class GridDistribution<2>;
template class GridDistribution<3>;

} /* namespace measurements */
} /* namespace wavepi */
