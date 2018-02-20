/*
 * DeltaMeasure.cpp
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#include <base/Norm.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <measurements/DeltaMeasure.h>
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
DeltaMeasure<dim>::DeltaMeasure(std::shared_ptr<SensorDistribution<dim>> points) : sensor_distribution(points) {}

template <int dim>
SensorValues<dim> DeltaMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
  AssertThrow(sensor_distribution && sensor_distribution->size(), ExcNotInitialized());
  this->mesh = field.get_mesh();

  SensorValues<dim> res(sensor_distribution);

  auto mapping = StaticMappingQ1<dim, spacedim>::mapping;
  std::pair<typename DoFHandler<dim, spacedim>::active_cell_iterator, Point<spacedim>> cell_point =
      GridTools::find_active_cell_around_point(mapping, dof_handler, p);

  Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));

  FEValues<dim, spacedim> fe_values(mapping, dof_handler.get_fe(), q, UpdateFlags(update_values));
  fe_values.reinit(cell_point.first);

  const unsigned int dofs_per_cell = dof_handler.get_fe().dofs_per_cell;

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  cell_point.first->get_dof_indices(local_dof_indices);

  for (unsigned int i = 0; i < dofs_per_cell; i++)
    rhs_vector(local_dof_indices[i]) = fe_values.shape_value(i, 0);

  // TODO
  AssertThrow(false, ExcNotImplemented());

  return res;
}

template <int dim>
DiscretizedFunction<dim> DeltaMeasure<dim>::adjoint(const SensorValues<dim>& measurements) {
  AssertThrow(mesh && sensor_distribution && sensor_distribution->size(), ExcNotInitialized());

  DiscretizedFunction<dim> res(mesh);

  // TODO
  AssertThrow(false, ExcNotImplemented());

  // indicate which norm we used for the adjoint
  res.set_norm(Norm::L2L2);
  res.dot_mult_mass_and_transform_inverse();

  return res;
}

template class DeltaMeasure<1>;
template class DeltaMeasure<2>;
template class DeltaMeasure<3>;

}  // namespace measurements
} /* namespace wavepi */
