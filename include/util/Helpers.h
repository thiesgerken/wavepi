/*
 * Helpers.h
 *
 *  Created on: 11.12.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_HELPERS_H_
#define INCLUDE_UTIL_HELPERS_H_

#include <deal.II/base/exceptions.h>
#include <cstring>
#include <iostream>
#include <iterator>
#include <map>
#include <string>
#include <sstream>

namespace wavepi {
namespace util {

using namespace dealii;

class Helpers {
   public:

      static std::string replace(std::string const &in, std::map<std::string, std::string> const &subst);

   private:
      Helpers() {};
};


}
}


#endif /* INCLUDE_UTIL_HELPERS_H_ */
