/*
 * Helpers.cpp
 *
 *  Created on: 11.12.2017
 *      Author: thies
 */

#include <util/Helpers.h>

namespace wavepi {
namespace util {

using namespace dealii;

std::string Helpers::replace(std::string const &in, std::map<std::string, std::string> const &subst) {
  std::ostringstream out;
  size_t pos = 0;
  for (;;) {
    size_t subst_pos = in.find("{{", pos);
    size_t end_pos   = in.find("}}", subst_pos);
    if (end_pos == std::string::npos) break;

    out.write(&*in.begin() + pos, subst_pos - pos);

    subst_pos += strlen("{{");
    std::map<std::string, std::string>::const_iterator subst_it = subst.find(in.substr(subst_pos, end_pos - subst_pos));

    AssertThrow(subst_it != subst.end(), ExcMessage("undefined substitution"))

            out
        << subst_it->second;
    pos = end_pos + strlen("}}");
  }
  out << in.substr(pos, std::string::npos);
  return out.str();
}

}  // namespace util
}  // namespace wavepi
