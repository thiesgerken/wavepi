/*
 * Measure.cpp
 *
 *  Created on: 17.08.2017
 *      Author: thies
 */

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <forward/Measure.h>
#include <stddef.h>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim> PointMeasure<dim>::PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
      const std::vector<Point<dim + 1>>& points, std::shared_ptr<Function<dim>> delta_shape,
      double delta_scale_space, double delta_scale_time)
      : mesh(solution_mesh), measurement_points(points), delta_shape(delta_shape), delta_scale_space(
            delta_scale_space), delta_scale_time(delta_scale_time) {
}

template<int dim> PointMeasure<dim>::PointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh)
      : mesh(solution_mesh), measurement_points(), delta_shape(), delta_scale_space(0.0), delta_scale_time(
            0.0) {
}

template<int dim> std::vector<double> PointMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
   AssertThrow(
         mesh && delta_shape && delta_scale_space > 0 && delta_scale_time > 0
               && measurement_points.size() > 0, ExcNotInitialized());

   // TODO: Implement, probably also needs class members for shape of delta approx and its size.
   AssertThrow(false, ExcNotImplemented());
}

template<int dim> DiscretizedFunction<dim> PointMeasure<dim>::adjoint(
      const std::vector<double>& measurements) {
   AssertThrow(
         mesh && delta_shape && delta_scale_space > 0 && delta_scale_time > 0
               && measurement_points.size() > 0, ExcNotInitialized());

   // TODO: Implement
   AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void PointMeasure<dim>::declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection("PointMeasure");
   {
      AssertThrow(false, ExcNotImplemented());
      //  prm.declare_entry("tol", "0.7", Patterns::Double(0, 1), "rel. tolerance");
   }
   prm.leave_subsection();
}

template<int dim>
void PointMeasure<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("PointMeasure");
   {
      AssertThrow(false, ExcNotImplemented());
//   tol = prm.get_double("tol");
   }
   prm.leave_subsection();
}

template<int dim>
GridPointMeasure<dim>::GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh,
      const std::vector<double> &times, const std::vector<std::vector<double>> &spatial_points,
      std::shared_ptr<Function<dim>> delta_shape, double delta_scale_space, double delta_scale_time)
      : PointMeasure<dim>(solution_mesh, make_grid(times, spatial_points), delta_shape, delta_scale_space,
            delta_scale_time) {
}

template<int dim>
GridPointMeasure<dim>::GridPointMeasure(std::shared_ptr<SpaceTimeMesh<dim>> solution_mesh)
      : PointMeasure<dim>(solution_mesh) {
}

template<int dim>
void GridPointMeasure<dim>::declare_parameters(ParameterHandler &prm) {
   prm.enter_subsection("GridPointMeasure");
   {
      AssertThrow(false, ExcNotImplemented());

      PointMeasure<dim>::declare_parameters(prm);
         //   prm.declare_entry("tol", "0.7", Patterns::Double(0, 1), "rel. tolerance");
   }
   prm.leave_subsection();
}

template<int dim>
void GridPointMeasure<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("GridPointMeasure");
   {
      AssertThrow(false, ExcNotImplemented());

      PointMeasure<dim>::get_parameters(prm);

      // tol = prm.get_double("tol");
   }
   prm.leave_subsection();
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
