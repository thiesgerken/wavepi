/*
 * Measure.cpp
 *
 *  Created on: 17.08.2017
 *      Author: thies
 */

#include <deal.II/base/function.h>

#include <measurements/Measure.h>

#include <stddef.h>
#include <cstdio>
#include <string>

namespace wavepi {
namespace measurements {
using namespace dealii;
using namespace wavepi::util;
using namespace wavepi::forward;

template<int dim> PointMeasure<dim>::PointMeasure(  std::shared_ptr<SpaceTimeGrid<dim>> points, std::shared_ptr<Function<dim>> delta_shape,
      double delta_scale_space, double delta_scale_time)
      :  mesh(), measurement_points(points), delta_shape(delta_shape), delta_scale_space(
            delta_scale_space), delta_scale_time(delta_scale_time) {
}

template<int dim> PointMeasure<dim>::PointMeasure()
      : mesh(), measurement_points(), delta_shape(), delta_scale_space(0.0), delta_scale_time(
            0.0) {
}

template<int dim> MeasuredValues<dim> PointMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
   AssertThrow(          delta_shape && delta_scale_space > 0 && delta_scale_time > 0
         && measurement_points  && measurement_points->size() > 0, ExcNotInitialized());
   this->mesh = field.get_mesh();

   // TODO: Implement, probably also needs class members for shape of delta approx and its size.
   AssertThrow(false, ExcNotImplemented());
}

template<int dim> DiscretizedFunction<dim> PointMeasure<dim>::adjoint(
      const MeasuredValues<dim>& measurements) {
   AssertThrow(
         mesh && delta_shape && delta_scale_space > 0 && delta_scale_time > 0
               && measurement_points && measurement_points->size() > 0, ExcNotInitialized());

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
GridPointMeasure<dim>::GridPointMeasure(
      const std::vector<double> &times, const std::vector<std::vector<double>> &spatial_points,
      std::shared_ptr<Function<dim>> delta_shape, double delta_scale_space, double delta_scale_time)
      : PointMeasure<dim>(SpaceTimeGrid<dim>::make_grid(times, spatial_points), delta_shape, delta_scale_space,
            delta_scale_time) {
}

template<int dim>
GridPointMeasure<dim>::GridPointMeasure()
      : PointMeasure<dim>() {
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
      this->set_measurement_points(SpaceTimeGrid<dim>::make_grid(temporal_points, spatial_points));
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

template class GridPointMeasure<1> ;
template class GridPointMeasure<2> ;
template class GridPointMeasure<3> ;

} /* namespace forward */
} /* namespace wavepi */
