/*
 * GridTools.h
 *
 *  Created on: 09.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_GRIDTOOLS_H_
#define INCLUDE_UTIL_GRIDTOOLS_H_

#include <deal.II/grid/tria.h>

namespace wavepi {
namespace util {
namespace GridTools {
using namespace dealii;

template <int dim>
void set_all_boundary_ids(Triangulation<dim> &tria, int id);

} /* namespace GridTools */
} /* namespace util */
} /* namespace wavepi */

#endif /* INCLUDE_UTIL_GRIDTOOLS_H_ */
