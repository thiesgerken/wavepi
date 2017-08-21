/*
 * RadialParsedFunction.h
 *
 *  Created on: 21.08.2017
 *      Author: thies
 */

#ifndef LIB_UTIL_RADIALPARSEDFUNCTION_H_
#define LIB_UTIL_RADIALPARSEDFUNCTION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/function_parser.h>
#include <string>

namespace wavepi {
namespace util {

using namespace dealii;

/**
 * Function that is radially symmetric in time and space.
 */
template<int dim>
class RadialParsedFunction: public Function<dim> {
   public:
      virtual ~RadialParsedFunction() = default;

      RadialParsedFunction(std::string function_description);

      virtual double value(const Point<dim> & p, const unsigned int component = 0) const;

      virtual void set_time(double time);


   private:
      FunctionParser<1> base;
};

} /* namespace util */
} /* namespace wavepi */

#endif /* LIB_UTIL_RADIALPARSEDFUNCTION_H_ */
