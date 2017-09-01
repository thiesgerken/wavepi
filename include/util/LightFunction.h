/*
 * LightFunction.h
 *
 *  Created on: 01.09.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_LIGHTFUNCTION_H_
#define INCLUDE_UTIL_LIGHTFUNCTION_H_

#include <deal.II/base/point.h>

namespace wavepi {
namespace util {
using namespace dealii;

/**
 * no `set_time` s.t. evaluation can be thread safe
 */
template <int dim>
class LightFunction {
   public:
      virtual ~LightFunction() = default;

      /**
       * evaluate function
       *
       * @param p point in spacetime to evaluate at (last entry is time)
       */
      virtual double evaluate(const Point<dim+1> &p) const = 0;
};

} /* namespace util */
} /* namespace wavepi */

#endif /* INCLUDE_UTIL_LIGHTFUNCTION_H_ */
