/*
 * ConvolutionMeasure.cpp
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#include <base/Norm.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <measurements/ConvolutionMeasure.h>
#include <measurements/SensorValues.h>
#include <stddef.h>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::base;

template <int dim>
ConvolutionMeasure<dim>::ConvolutionMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                            std::shared_ptr<SensorDistribution<dim>> points, Norm norm,
                                            std::shared_ptr<LightFunction<dim>> delta_shape, double delta_scale_space,
                                            double delta_scale_time)
    : mesh(mesh),
      sensor_distribution(points),
      norm(norm),
      delta_shape(delta_shape),
      delta_scale_space(delta_scale_space),
      delta_scale_time(delta_scale_time) {
  AssertThrow(mesh && delta_shape, ExcNotInitialized());
}

template <int dim>
ConvolutionMeasure<dim>::ConvolutionMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                            std::shared_ptr<SensorDistribution<dim>> points, Norm norm)
    : mesh(mesh),
      sensor_distribution(points),
      norm(norm),
      delta_shape(),
      delta_scale_space(0.0),
      delta_scale_time(0.0) {
  AssertThrow(mesh, ExcNotInitialized());
}

template <int dim>
std::vector<std::vector<std::pair<size_t, double>>> ConvolutionMeasure<dim>::compute_jobs() const {
  std::vector<std::vector<std::pair<size_t, double>>> jobs(mesh->length());

  size_t sensor_offset = 0;
  for (size_t mti = 0; mti < sensor_distribution->get_times().size(); mti++) {
    double mtime = sensor_distribution->get_times()[mti];

    for (size_t ti = 0; ti < mesh->get_times().size(); ti++) {
      double t = mesh->get_time(ti);

      if (t < mtime - delta_scale_time) continue;
      if (t > mtime + delta_scale_time) break;

      double factor = 0.0;

      if (ti > 0) factor += (t - mesh->get_time(ti - 1)) / 2;

      if (ti + 1 < mesh->get_times().size()) factor += (mesh->get_time(ti + 1) - t) / 2;

      // norm correction
      factor *= delta_scale_time * pow(delta_scale_space, dim);

      for (size_t msi = 0; msi < sensor_distribution->get_points_per_time(mti).size(); msi++)
        jobs[ti].emplace_back(msi + sensor_offset, factor);
    }

    sensor_offset += sensor_distribution->get_points_per_time(mti).size();
  }

  return jobs;
}

template <int dim>
SensorValues<dim> ConvolutionMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
  AssertThrow(delta_shape && delta_scale_space > 0 && delta_scale_time > 0, ExcNotInitialized());
  AssertThrow(sensor_distribution && sensor_distribution->size(), ExcNotInitialized());
  AssertThrow(mesh == field.get_mesh(), ExcMessage("ConvolutionMeasure called with different meshes"));
  AssertThrow(norm == field.get_norm(), ExcMessage("ConvolutionMeasure called with different norms"));

  LightFunctionWrapper wrapper(delta_shape, delta_scale_space, delta_scale_time);
  SensorValues<dim> res(sensor_distribution);

  auto jobs = compute_jobs();

  for (size_t ji = 0; ji < jobs.size(); ji++) {
    auto dof = field.get_mesh()->get_dof_handler(ji);
    Vector<double> interp_shape(dof->n_dofs());

    wrapper.set_time(mesh->get_time(ji));

    for (size_t k = 0; k < jobs[ji].size(); k++) {
      wrapper.set_offset((*sensor_distribution)[jobs[ji][k].first]);

      interp_shape = 0.0;
      VectorTools::interpolate(*dof, wrapper, interp_shape);
      mesh->get_constraint_matrix(ji)->distribute(interp_shape);

      res[jobs[ji][k].first] +=
          jobs[ji][k].second * mesh->get_mass_matrix(ji)->matrix_scalar_product(interp_shape, field[ji]);
    }
  }

  return res;
}

template <int dim>
SensorValues<dim> ConvolutionMeasure<dim>::zero() {
  return SensorValues<dim>(sensor_distribution);
}

template <int dim>
DiscretizedFunction<dim> ConvolutionMeasure<dim>::adjoint(const SensorValues<dim>& measurements) {
  AssertThrow(delta_shape && delta_scale_space > 0 && delta_scale_time > 0, ExcNotInitialized());
  AssertThrow(mesh && sensor_distribution && sensor_distribution->size(), ExcNotInitialized());

  DiscretizedFunction<dim> res(mesh);
  auto jobs = compute_jobs();

  LightFunctionWrapper wrapper(delta_shape, delta_scale_space, delta_scale_time);

  for (size_t ji = 0; ji < jobs.size(); ji++) {
    auto dof_handler = mesh->get_dof_handler(ji);
    wrapper.set_time(mesh->get_time(ji));
    Vector<double> tmp(dof_handler->n_dofs());

    for (size_t i = 0; i < jobs[ji].size(); i++) {
      size_t sensor_idx = jobs[ji][i].first;

      wrapper.set_offset((*sensor_distribution)[sensor_idx]);

      tmp = 0.0;

      // interpolate makes sense if the forward measurement operator also uses the interpolation.
      VectorTools::interpolate(*dof_handler, wrapper, tmp);
      mesh->get_constraint_matrix(ji)->distribute(tmp);

      res[ji].add(jobs[ji][i].second * measurements[sensor_idx], tmp);
    }
  }

  res.set_norm(norm);

  // res has coefficients in there, not dot products.
  res.dot_mult_mass_and_transform_inverse();

  return res;
}

template <int dim>
void ConvolutionMeasure<dim>::declare_parameters(ParameterHandler& prm) {
  prm.enter_subsection("ConvolutionMeasure");
  {
    prm.declare_entry("radius space", "0.2", Patterns::Double(0), "scaling of shape function in spatial variables");
    prm.declare_entry("radius time", "0.2", Patterns::Double(0), "scaling of shape function in time variable");
    prm.declare_entry("shape", "hat", Patterns::Selection("hat|constant"),
                      "shape of the delta approximating function. ");
  }
  prm.leave_subsection();
}

template <int dim>
void ConvolutionMeasure<dim>::get_parameters(ParameterHandler& prm) {
  prm.enter_subsection("ConvolutionMeasure");
  {
    auto shape_desc = prm.get("shape");

    if (shape_desc == "hat")
      delta_shape = std::make_shared<HatShape>();
    else if (shape_desc == "constant")
      delta_shape = std::make_shared<ConstShape>();
    else
      AssertThrow(false, ExcMessage("Unknown delta shape: " + shape_desc));

    delta_scale_space = prm.get_double("radius space");
    delta_scale_time  = prm.get_double("radius time");

    AssertThrow(delta_scale_space * delta_scale_time > 0.0,
                ExcMessage("sensor radii in time and space have to be positive!"));
  }
  prm.leave_subsection();
}

template class ConvolutionMeasure<1>;
template class ConvolutionMeasure<2>;
template class ConvolutionMeasure<3>;

}  // namespace measurements
} /* namespace wavepi */
