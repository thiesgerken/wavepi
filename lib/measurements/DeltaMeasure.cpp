/*
 * DeltaMeasure.cpp
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_tools.h>

#include <base/ConstantMesh.h>
#include <base/Norm.h>
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
DeltaMeasure<dim>::DeltaMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                std::shared_ptr<SensorDistribution<dim>> points,
                                std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm)
    : mesh(mesh), sensor_distribution(points), norm(norm) {
  AssertThrow(mesh && norm, ExcNotInitialized());
}

template <int dim>
SensorValues<dim> DeltaMeasure<dim>::evaluate(const DiscretizedFunction<dim>& field) {
  AssertThrow(sensor_distribution && sensor_distribution->size(), ExcNotInitialized());
  AssertThrow(mesh == field.get_mesh(), ExcMessage("DeltaMeasure called with different meshes"));
  AssertThrow(*norm == *field.get_norm(), ExcMessage("DeltaMeasure called with different norms"));

  SensorValues<dim> res(sensor_distribution);
  auto mapping = StaticMappingQ1<dim>::mapping;

  if (std::dynamic_pointer_cast<ConstantMesh<dim>, SpaceTimeMesh<dim>>(mesh) &&
      sensor_distribution->times_per_point_available()) {
    // specialized implementation that is ordered by points, not time
    // (most expensive operation is find_active_cell_around_point(...))
    auto dof_handler = mesh->get_dof_handler(0);

    for (size_t mpi = 0; mpi < sensor_distribution->get_points().size(); mpi++) {
      auto pos = sensor_distribution->get_points()[mpi];

      std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> cell_point =
          GridTools::find_active_cell_around_point(mapping, *dof_handler, pos);

      Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
      FEValues<dim> fe_values(mapping, dof_handler->get_fe(), q, UpdateFlags(update_values));
      fe_values.reinit(cell_point.first);

      const unsigned int dofs_per_cell = dof_handler->get_fe().dofs_per_cell;

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell_point.first->get_dof_indices(local_dof_indices);

      for (size_t mti = 0; mti < sensor_distribution->get_times_per_point(mpi).size(); mti++) {
        double mtime = sensor_distribution->get_times_per_point(mpi)[mti];
        size_t ti    = mesh->nearest_time(mtime);

        // (φ_i, δ_p) = fe_values.shape_value(i, 0)
        // this loop calculates the dot product between field[ti] and δ_p directly
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          res[sensor_distribution->index_times_per_point(mpi, mti)] +=
              field[ti][local_dof_indices[i]] * fe_values.shape_value(i, 0);
      }
    }
  } else {
    size_t offset = 0;

    for (size_t mti = 0; mti < sensor_distribution->get_times().size(); mti++) {
      double mtime = sensor_distribution->get_times()[mti];

      size_t ti        = mesh->nearest_time(mtime);
      auto dof_handler = mesh->get_dof_handler(ti);

      for (size_t msi = 0; msi < sensor_distribution->get_points_per_time(mti).size(); msi++) {
        auto pos = sensor_distribution->get_points_per_time(mti)[msi];

        std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> cell_point =
            GridTools::find_active_cell_around_point(mapping, *dof_handler, pos);

        Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
        FEValues<dim> fe_values(mapping, dof_handler->get_fe(), q, UpdateFlags(update_values));
        fe_values.reinit(cell_point.first);

        const unsigned int dofs_per_cell = dof_handler->get_fe().dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell_point.first->get_dof_indices(local_dof_indices);

        // (φ_i, δ_p) = fe_values.shape_value(i, 0)
        // this loop calculates the dot product between field[ti] and δ_p directly
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          res[msi + offset] += field[ti][local_dof_indices[i]] * fe_values.shape_value(i, 0);
      }

      offset += sensor_distribution->get_points_per_time(mti).size();
    }
  }

  return res;
}

template <int dim>
SensorValues<dim> DeltaMeasure<dim>::zero() {
  return SensorValues<dim>(sensor_distribution);
}

template <int dim>
DiscretizedFunction<dim> DeltaMeasure<dim>::adjoint(const SensorValues<dim>& measurements) {
  AssertThrow(mesh && sensor_distribution && sensor_distribution->size(), ExcNotInitialized());

  DiscretizedFunction<dim> res(mesh);
  auto mapping  = StaticMappingQ1<dim>::mapping;
  size_t offset = 0;

  if (std::dynamic_pointer_cast<ConstantMesh<dim>, SpaceTimeMesh<dim>>(mesh) &&
      sensor_distribution->times_per_point_available()) {
    // specialized implementation that is ordered by points, not time
    // (most expensive operation is find_active_cell_around_point(...))
    auto dof_handler = mesh->get_dof_handler(0);

    for (size_t mpi = 0; mpi < sensor_distribution->get_points().size(); mpi++) {
      auto pos = sensor_distribution->get_points()[mpi];

      std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> cell_point =
          GridTools::find_active_cell_around_point(mapping, *dof_handler, pos);

      Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
      FEValues<dim> fe_values(mapping, dof_handler->get_fe(), q, UpdateFlags(update_values));
      fe_values.reinit(cell_point.first);

      const unsigned int dofs_per_cell = dof_handler->get_fe().dofs_per_cell;

      std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
      cell_point.first->get_dof_indices(local_dof_indices);

      for (size_t mti = 0; mti < sensor_distribution->get_times_per_point(mpi).size(); mti++) {
        double mtime = sensor_distribution->get_times_per_point(mpi)[mti];
        size_t ti    = mesh->nearest_time(mtime);

        // (φ_i, δ_p) = fe_values.shape_value(i, 0)
        // this loop calculates the dot product between field[ti] and δ_p directly
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          res[ti][local_dof_indices[i]] +=
              measurements[sensor_distribution->index_times_per_point(mpi, mti)] * fe_values.shape_value(i, 0);
      }
    }
  } else {
    for (size_t mti = 0; mti < sensor_distribution->get_times().size(); mti++) {
      double mtime = sensor_distribution->get_times()[mti];

      size_t ti        = mesh->nearest_time(mtime);
      auto dof_handler = mesh->get_dof_handler(ti);

      for (size_t msi = 0; msi < sensor_distribution->get_points_per_time(mti).size(); msi++) {
        auto pos = sensor_distribution->get_points_per_time(mti)[msi];

        std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim>> cell_point =
            GridTools::find_active_cell_around_point(mapping, *dof_handler, pos);

        Quadrature<dim> q(GeometryInfo<dim>::project_to_unit_cell(cell_point.second));
        FEValues<dim> fe_values(mapping, dof_handler->get_fe(), q, UpdateFlags(update_values));
        fe_values.reinit(cell_point.first);

        const unsigned int dofs_per_cell = dof_handler->get_fe().dofs_per_cell;

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        cell_point.first->get_dof_indices(local_dof_indices);

        // (φ_i, δ_p) = fe_values.shape_value(i, 0)
        // this loop calculates dot products, not coefficients.
        for (unsigned int i = 0; i < dofs_per_cell; i++)
          res[ti][local_dof_indices[i]] += measurements[msi + offset] * fe_values.shape_value(i, 0);
      }

      offset += sensor_distribution->get_points_per_time(mti).size();
    }
  }

  // indicate which norm we used for the adjoint
  res.set_norm(norm);
  res.dot_transform_inverse();

  return res;
}

template class DeltaMeasure<1>;
template class DeltaMeasure<2>;
template class DeltaMeasure<3>;

}  // namespace measurements
}  // namespace wavepi
