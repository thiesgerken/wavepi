/*
 * Measure.cpp
 *
 *  Created on: 17.08.2017
 *      Author: thies
 */

#include <forward/Measure.h>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim> PointMeasure<dim>::PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
      const std::vector<Point<dim + 1>>& points)
      : mesh(solution_mesh), measurement_points(points) {
}

template<int dim> std::vector<double> PointMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
   // TODO: Implement, probably also needs class members for shape of delta approx and its size.
   AssertThrow(false, ExcNotImplemented());
}

template<int dim> DiscretizedFunction<dim> PointMeasure<dim>::adjoint(
      const std::vector<double>& measurements) {
   // TODO: Implement
   AssertThrow(false, ExcNotImplemented());
}

template<int dim>
GridPointMeasure<dim>::GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
      const std::vector<double> &times, const std::vector<std::vector<double>> &spatial_points)
      : PointMeasure<dim>(solution_mesh, make_grid(times, spatial_points)) {
}

template<>
std::vector<Point<2>> GridPointMeasure<1>::make_grid(const std::vector<double> &times,
      const std::vector<std::vector<double>> &spatial_points) {
   Assert(spatial_points.size() == 1, ExcInternalError());
   size_t nb_points = times.size();

   for (auto coords : spatial_points)
      nb_points *= coords.size();

   std::vector<Point<2>> points(nb_points);

   for (size_t i = 0; i < times.size(); i++)
      for (size_t ix = 0; ix < spatial_points[0].size(); ix++)
         points[i * spatial_points[0].size() + ix] = Point<2>(spatial_points[0][ix], times[i]);

   return points;
}

template<>
std::vector<Point<3>> GridPointMeasure<2>::make_grid(const std::vector<double> &times,
      const std::vector<std::vector<double>> &spatial_points) {
   Assert(spatial_points.size() == 2, ExcInternalError());
   size_t nb_points = times.size();

   for (auto coords : spatial_points)
      nb_points *= coords.size();

   std::vector<Point<3>> points(nb_points);

   for (size_t i = 0; i < times.size(); i++)
      for (size_t ix = 0; ix < spatial_points[0].size(); ix++)
         for (size_t iy = 0; iy < spatial_points[1].size(); iy++)
            points[i * spatial_points[0].size() * spatial_points[1].size() + ix * spatial_points[1].size()
                  + iy] = Point<3>(spatial_points[0][ix], spatial_points[1][iy], times[i]);

   return points;
}

template<>
std::vector<Point<4>> GridPointMeasure<3>::make_grid(const std::vector<double> &times,
      const std::vector<std::vector<double>> &spatial_points) {
   Assert(spatial_points.size() == 3, ExcInternalError());
   size_t nb_points = times.size();

   for (auto coords : spatial_points)
      nb_points *= coords.size();

   std::vector<Point<4>> points(nb_points);

   for (size_t i = 0; i < times.size(); i++)
      for (size_t ix = 0; ix < spatial_points[0].size(); ix++)
         for (size_t iy = 0; iy < spatial_points[1].size(); iy++)
            for (size_t iz = 0; iz < spatial_points[2].size(); iz++) {
               Point<4> pt;

               pt[0] = spatial_points[0][ix];
               pt[1] = spatial_points[1][iy];
               pt[2] = spatial_points[2][iz];
               pt[3] = times[i];

               points[i * spatial_points[0].size() * spatial_points[1].size() * spatial_points[2].size()
                     + ix * spatial_points[1].size() * spatial_points[2].size()
                     + iy * spatial_points[2].size() + iz] = pt;
            }

   return points;
}

template class GridPointMeasure<1> ;
template class GridPointMeasure<2> ;
template class GridPointMeasure<3> ;

} /* namespace forward */
} /* namespace wavepi */
