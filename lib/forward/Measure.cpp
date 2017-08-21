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
#include <cstdio>
#include <string>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::util;

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
      prm.declare_entry("radius space", "0.1", Patterns::Double(0),
            "scaling of shape function in spatial variables");
      prm.declare_entry("radius time", "0.1", Patterns::Double(0),
            "scaling of shape function in time variable");
      prm.declare_entry("shape", "if(r < 1, sqrt(3)*(1-r), 0) * if(t < 1, sqrt(3)*(1-t), 0)",
            Patterns::Anything(),
            "shape of the delta approximating function, operating on variables r and t. has to have support in [0,1]^2.");
   }
   prm.leave_subsection();
}

template<int dim>
void PointMeasure<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("PointMeasure");
   {
      delta_shape = std::make_shared<RadialParsedFunction<dim>>(prm.get("shape"));

      delta_scale_space = prm.get_double("radius space");
      delta_scale_time = prm.get_double("radius time");

      AssertThrow(delta_scale_space * delta_scale_time > 0.0,
            ExcMessage("sensor radii in time and space have to be positive!"));
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
      prm.declare_entry("points x", "-1:10:1", Patterns::Anything(),
            "points for the grid in x-direction. Format: '[lower bound]:[number of points]:[upper bound]'.\n Lower bound and upper bound are exclusive.");
      prm.declare_entry("points y", "-1:10:1", Patterns::Anything(),
            "points for the grid in y-direction. Format: '[lower bound]:[number of points]:[upper bound]'.\n Lower bound and upper bound are exclusive.");
      prm.declare_entry("points z", "-1:10:1", Patterns::Anything(),
            "points for the grid in z-direction. Format: '[lower bound]:[number of points]:[upper bound]'.\n Lower bound and upper bound are exclusive.");
      prm.declare_entry("points t", "-1:10:1", Patterns::Anything(),
            "points for the grid in time. Format: '[lower bound]:[number of points]:[upper bound]'.\n Upper bound is inclusive, lower bound is exclusive iff it equals 0.0.");

      PointMeasure<dim>::declare_parameters(prm);
   }
   prm.leave_subsection();
}

template<int dim>
void GridPointMeasure<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("GridPointMeasure");
   {
      Assert(dim <= 3, ExcInternalError());

      std::vector<std::vector<double>> spatial_points;
      spatial_points.emplace_back(make_points(prm.get("points x")));

      if (dim > 1)
         spatial_points.emplace_back(make_points(prm.get("points y")));

      if (dim > 2)
         spatial_points.emplace_back(make_points(prm.get("points z")));

      auto temporal_points = make_points(prm.get("points t"), true);
      this->set_measurement_points(make_grid(temporal_points, spatial_points));
      PointMeasure<dim>::get_parameters(prm);
   }
   prm.leave_subsection();
}

template<int dim>
std::vector<double> GridPointMeasure<dim>::make_points(const std::string description, bool is_time) {
   double lb, ub;
   size_t nb;

   AssertThrow(std::sscanf(description.c_str(), "%*[ ]%lf:%*[ ]%zu:%*[ ]%lf%*[ ]", &lb, &nb, &ub),
         ExcMessage("Could not parse points"));
   AssertThrow((is_time && nb > 1 && lb < ub && lb >= 0.0) || (!is_time && nb >= 1 && lb <= ub),
         ExcMessage("Illegal interval spec"));

   std::vector<double> points(nb);

   if (lb == 0.0 && is_time) // lb excl, ub incl
      for (size_t i = 0; i < nb; i++)
         points[i] = lb + (i + 1) * (ub - lb) / nb;
   else if (lb > 0.0 && is_time) // lb incl, ub incl
      for (size_t i = 0; i < nb; i++)
         points[i] = lb + (i + 1) * (ub - lb) / (nb - 1);
   else
      // lb excl, ub excl
      for (size_t i = 0; i < nb; i++)
         points[i] = lb + (i + 1) * (ub - lb) / (nb + 1);

   return points;
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
