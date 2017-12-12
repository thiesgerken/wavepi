/*
 * GridPointMeasure.cpp
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#include <deal.II/base/function.h>

#include <measurements/GridPointMeasure.h>

#include <stddef.h>
#include <cstdio>
#include <string>

namespace wavepi {
namespace measurements {
using namespace dealii;
using namespace wavepi::util;
using namespace wavepi::forward;

template<int dim>
GridPointMeasure<dim>::GridPointMeasure(const std::vector<double> &times,
      const std::vector<std::vector<double>> &spatial_points, std::shared_ptr<LightFunction<dim>> delta_shape,
      double delta_scale_space, double delta_scale_time)
      : PointMeasure<dim>(std::make_shared<SpaceTimeGrid<dim>>(times, spatial_points), delta_shape,
            delta_scale_space, delta_scale_time) {
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
            "points for the grid in x-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound are exclusive.");
      prm.declare_entry("points y", "-1:10:1", Patterns::Anything(),
            "points for the grid in y-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound are exclusive.");
      prm.declare_entry("points z", "-1:10:1", Patterns::Anything(),
            "points for the grid in z-direction. Format: '[lb]:[n_points]:[ub]'. Lower bound and upper bound are exclusive.");
      prm.declare_entry("points t", "0:10:6", Patterns::Anything(),
            "points for the grid in time. Format: '[lower bound]:[n_points]:[ub]'. Upper bound is inclusive, lower bound is exclusive iff it equals 0.0.");

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
      this->set_measurement_points(std::make_shared<SpaceTimeGrid<dim>>(temporal_points, spatial_points));
      PointMeasure<dim>::get_parameters(prm);
   }
   prm.leave_subsection();
}

template<int dim>
std::vector<double> GridPointMeasure<dim>::make_points(const std::string description, bool is_time) {
   double lb, ub;
   size_t nb;

   AssertThrow(std::sscanf(description.c_str(), " %lf : %zu : %lf ", &lb, &nb, &ub) == 3,
         ExcMessage("Could not parse points"));
   AssertThrow((is_time && nb > 1 && lb < ub && lb >= 0.0) || (!is_time && nb >= 1 && lb <= ub),
         ExcMessage("Illegal interval spec: " + description));

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

