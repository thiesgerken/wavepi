/*
 * GridTools.cpp
 *
 *  Created on: 09.08.2017
 *      Author: thies
 */

#include <deal.II/base/geometry_info.h>
#include <util/GridTools.h>

namespace wavepi {
namespace util {
namespace GridTools {

template <int dim>
void set_all_boundary_ids(Triangulation<dim> &tria, int id) {
  typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(), endc = tria.end();
  for (; cell != endc; ++cell) {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
      if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(id);
    }
  }
}

template void set_all_boundary_ids(Triangulation<1> &tria, int id);
template void set_all_boundary_ids(Triangulation<2> &tria, int id);
template void set_all_boundary_ids(Triangulation<3> &tria, int id);

} /* namespace GridTools */
} /* namespace util */
} /* namespace wavepi */
