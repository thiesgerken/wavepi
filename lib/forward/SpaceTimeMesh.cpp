/*
 * SpaceTimeMesh.cpp
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <forward/SpaceTimeMesh.h>
#include <cmath>
#include <iostream>
#include <iterator>

namespace wavepi {
namespace forward {

template<int dim>
SpaceTimeMesh<dim>::SpaceTimeMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad)
      : times(times), fe(fe), quad(quad) {
   Assert(times.size() > 1, ExcInternalError());
   Assert(times[0] == 0, ExcInternalError());

   // quick test whether the times are increasing. (fails if they are reversed)
   Assert(times[1] > times[0], ExcInternalError());
}

template<int dim>
size_t SpaceTimeMesh<dim>::nearest_time(double time, size_t low, size_t up) const {
   Assert(low <= up, ExcInternalError()); // something went wrong

   if (low >= up) // low == up or sth went wrong
      return low;

   if (low + 1 == up) {
      if (std::abs(times[low] - time) <= std::abs(times[up] - time))
         return low;
      else
         return up;
   }

   size_t middle = (low + up) / 2;
   double val = times[middle];

   if (time > val)
      return nearest_time(time, middle, up);
   else if (time < val)
      return nearest_time(time, low, middle);
   else
      return middle;
}

template<int dim>
size_t SpaceTimeMesh<dim>::nearest_time(double time) const {
   Assert(times.size() > 0, ExcEmptyObject());

   if (times.size() == 1)
      return 0;
   else
      return nearest_time(time, 0, times.size() - 1);
}

template<int dim>
size_t SpaceTimeMesh<dim>::find_time(double time) const {
   size_t idx = nearest_time(time);
   AssertThrow(std::abs(time - times[idx]) < 1e-10,
         ExcMessage("time " + std::to_string(time) + " is not member of this mesh"));

   return idx;
}

template class SpaceTimeMesh<1> ;
template class SpaceTimeMesh<2> ;
template class SpaceTimeMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */

