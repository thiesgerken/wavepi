/*
 * Helpers.h
 *
 *  Created on: 11.12.2017
 *      Author: thies
 */

#ifndef INCLUDE_BASE_UTIL_H_
#define INCLUDE_BASE_UTIL_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/grid/tria.h>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <string>

namespace wavepi {
namespace base {

using namespace dealii;

class Util {
 public:
  static std::string replace(std::string const &in, std::map<std::string, std::string> const &subst);

  template <int dim>
  static void set_all_boundary_ids(Triangulation<dim> &tria, int id);

  static std::string format_duration(const double seconds);

 private:
  Util(){};
};

}  // namespace base
}  // namespace wavepi

#endif /* INCLUDE_BASE_UTIL_H_ */
