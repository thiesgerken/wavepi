/*
 * Util.cpp
 *
 *  Created on: 11.12.2017
 *      Author: thies
 */

#include <base/Util.h>
#include <deal.II/base/geometry_info.h>

namespace wavepi {
namespace base {

using namespace dealii;

std::string Util::replace(std::string const &in, std::map<std::string, std::string> const &subst) {
  std::ostringstream out;
  size_t pos = 0;
  for (;;) {
    size_t subst_pos = in.find("{{", pos);
    size_t end_pos   = in.find("}}", subst_pos);
    if (end_pos == std::string::npos) break;

    out.write(&*in.begin() + pos, subst_pos - pos);

    subst_pos += strlen("{{");
    std::map<std::string, std::string>::const_iterator subst_it = subst.find(in.substr(subst_pos, end_pos - subst_pos));

    AssertThrow(subst_it != subst.end(), ExcMessage("undefined substitution"));

    out << subst_it->second;
    pos = end_pos + strlen("}}");
  }
  out << in.substr(pos, std::string::npos);
  return out.str();
}

template <int dim>
void Util::set_all_boundary_ids(Triangulation<dim> &tria, int id) {
  typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(), endc = tria.end();
  for (; cell != endc; ++cell) {
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
      if (cell->face(face)->at_boundary()) cell->face(face)->set_boundary_id(id);
    }
  }
}

template void Util::set_all_boundary_ids(Triangulation<1> &tria, int id);
template void Util::set_all_boundary_ids(Triangulation<2> &tria, int id);
template void Util::set_all_boundary_ids(Triangulation<3> &tria, int id);

}  // namespace base
}  // namespace wavepi
