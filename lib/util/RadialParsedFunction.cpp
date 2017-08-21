/*
 * RadialParsedFunction.cpp
 *
 *  Created on: 21.08.2017
 *      Author: thies
 */

#include <util/RadialParsedFunction.h>

namespace wavepi {
namespace util {
using namespace dealii;

template<int dim>
RadialParsedFunction<dim>::RadialParsedFunction(std::string function_description) {
   std::map<std::string, double> constants;
   base.initialize("r,t", function_description, constants, true);
}

template<int dim>
double RadialParsedFunction<dim>::value(const Point<dim> & p, const unsigned int component) const {
   Assert(component == 0, ExcInternalError());
   Point<1> nrm(p.norm());

   return base.value(nrm, 0);
}

template<int dim>
void RadialParsedFunction<dim>::set_time(double time) {
   Function<dim>::set_time(time);
   base.set_time(time);
}

template class RadialParsedFunction<1> ;
template class RadialParsedFunction<2> ;
template class RadialParsedFunction<3> ;

} /* namespace util */
} /* namespace wavepi */
