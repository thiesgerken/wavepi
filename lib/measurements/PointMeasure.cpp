/*
 * PointMeasure.cpp
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

#include <measurements/PointMeasure.h>

#include <stddef.h>
#include <list>
#include <utility>

namespace wavepi {
namespace measurements {
using namespace dealii;
using namespace wavepi::util;
using namespace wavepi::forward;

template<int dim>
PointMeasure<dim>::AssemblyScratchData::AssemblyScratchData(const FiniteElement<dim> &fe,
      const Quadrature<dim> &quad)
      : fe_values(fe, quad, update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
PointMeasure<dim>::AssemblyScratchData::AssemblyScratchData(const AssemblyScratchData &scratch_data)
      : fe_values(scratch_data.fe_values.get_fe(), scratch_data.fe_values.get_quadrature(),
            update_values | update_quadrature_points | update_JxW_values) {
}

template<int dim>
void PointMeasure<dim>::copy_local_to_global(const std::vector<std::pair<size_t, double>> &jobs,
      MeasuredValues<dim> &dest, const AssemblyCopyData &copy_data) const {
   for (size_t i = 0; i < jobs.size(); ++i)
      dest[jobs[i].first] += copy_data.cell_values[i];
}

template<int dim>
void PointMeasure<dim>::local_add_contributions(const std::vector<std::pair<size_t, double>> &jobs,
      const Vector<double> &u, double time, const typename DoFHandler<dim>::active_cell_iterator &cell,
      AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data) const {
   const unsigned int dofs_per_cell = scratch_data.fe_values.get_fe().dofs_per_cell;
   const unsigned int n_q_points = scratch_data.fe_values.get_quadrature().size();

   copy_data.cell_values.resize(jobs.size());
   scratch_data.fe_values.reinit(cell);

   std::vector<types::global_dof_index> local_dof_indices;
   cell->get_dof_indices(local_dof_indices);

   // TODO: Scale p2 in time and space and values s.t. l2 norm = 1 if it was before
     AssertThrow(false, ExcNotImplemented());

   for (size_t i = 0; i < jobs.size(); ++i) {
      Point<dim + 1> p = (*measurement_points)[jobs[i].first];
      p(dim) = time - p(dim);

      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point) {
         Point<dim + 1> p2(p);
         for (size_t d = 0; d < dim; d++)
            p2(d) = scratch_data.fe_values.quadrature_point(q_point)(d) - p2(d);

         const double val_delta = delta_shape->value(p2);

         for (unsigned int k = 0; k < dofs_per_cell; ++k)
            copy_data.cell_values[i] += jobs[i].second * val_delta * u[local_dof_indices[k]]
                  * scratch_data.fe_values.shape_value(k, q_point) * scratch_data.fe_values.JxW(q_point);
      }
   }
}

template<int dim> PointMeasure<dim>::PointMeasure(std::shared_ptr<SpaceTimeGrid<dim>> points,
      std::shared_ptr<Function<dim + 1>> delta_shape, double delta_scale_space, double delta_scale_time)
      : mesh(), measurement_points(points), delta_shape(delta_shape), delta_scale_space(delta_scale_space), delta_scale_time(
            delta_scale_time) {
}

template<int dim> PointMeasure<dim>::PointMeasure()
      : mesh(), measurement_points(), delta_shape(), delta_scale_space(0.0), delta_scale_time(0.0) {
}

template<int dim> MeasuredValues<dim> PointMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
   AssertThrow(
         delta_shape && delta_scale_space > 0 && delta_scale_time > 0 && measurement_points
               && measurement_points->size() > 0, ExcNotInitialized());
   this->mesh = field.get_mesh();

   // collect jobs so that we have to go through the mesh only once
   // for each time a list of sensor numbers and factors
   std::vector<std::vector<std::pair<size_t, double>>> jobs(mesh->length());

   size_t sensor_offset = 0;
   for (size_t mti = 0; mti < measurement_points->get_times().size(); mti++) {
      double mtime = measurement_points->get_times()[mti];

      for (size_t ti = 0; ti < mesh->get_times().size(); ti++) {
         double t = mesh->get_time(ti);

         if (t < mtime - delta_scale_time)
            continue;
         if (t > mtime + delta_scale_time)
            break;

         double factor = 0.0;

         if (ti > 0)
            factor += (t - mesh->get_time(ti - 1)) / 2;

         if (ti + 1 < mesh->get_times().size())
            factor += (mesh->get_time(ti + 1) - t) / 2;

         for (size_t msi = 0; msi < measurement_points->get_points()[mti].size(); msi++)
            jobs[ti].emplace_back(msi + sensor_offset, factor);
      }

      sensor_offset += measurement_points->get_points()[mti].size();
   }

   MeasuredValues<dim> res(measurement_points);

   for (size_t ji = 0; ji < jobs.size(); ji++) {
      auto dof = field.get_mesh()->get_dof_handler(ji);

      WorkStream::run(dof->begin_active(), dof->end(),
            std::bind(&PointMeasure<dim>::local_add_contributions, *this, std::ref(jobs[ji]),
                  std::ref(field.get_function_coefficient(ji)), mesh->get_time(ji), std::placeholders::_1,
                  std::placeholders::_2, std::placeholders::_3),
            std::bind(&PointMeasure<dim>::copy_local_to_global, *this, std::ref(jobs[ji]), std::ref(res),
                  std::placeholders::_1), AssemblyScratchData(dof->get_fe(), mesh->get_quadrature()),
            AssemblyCopyData());
   }

   return res;
}

template<int dim> DiscretizedFunction<dim> PointMeasure<dim>::adjoint(
      const MeasuredValues<dim>& measurements) {
   AssertThrow(
         mesh && delta_shape && delta_scale_space > 0 && delta_scale_time > 0 && measurement_points
               && measurement_points->size() > 0, ExcNotInitialized());

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
      prm.declare_entry("shape", "max(sqrt(3)*(1-norm{x|y|z}), 0) * if(t < 1, sqrt(3)*(1-t), 0)",
            Patterns::Anything(),
            "shape of the delta approximating function. Has to have support in [-1,1]^{dim+1}.");
   }
   prm.leave_subsection();
}

template<int dim>
void PointMeasure<dim>::get_parameters(ParameterHandler &prm) {
   prm.enter_subsection("PointMeasure");
   {
      std::map<std::string, double> constants;
      delta_shape = std::make_shared<MacroFunctionParser<dim + 1>>(prm.get("shape"), constants, true);

      delta_scale_space = prm.get_double("radius space");
      delta_scale_time = prm.get_double("radius time");

      AssertThrow(delta_scale_space * delta_scale_time > 0.0,
            ExcMessage("sensor radii in time and space have to be positive!"));
   }
   prm.leave_subsection();
}

template class PointMeasure<1> ;
template class PointMeasure<2> ;
template class PointMeasure<3> ;

} /* namespace forward */
} /* namespace wavepi */

