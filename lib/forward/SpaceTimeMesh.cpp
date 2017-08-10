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
bool SpaceTimeMesh<dim>::near_enough(double time, size_t idx) const {
   Assert(idx >= 0 && idx < times.size(), ExcIndexRange(idx, 0, times.size()));

   if (times.size() == 1)
      return std::abs(times[idx] - time) < 1e-3;
   else if (idx > 0)
      return std::abs(times[idx] - time) < 1e-3 * std::abs(times[idx] - times[idx - 1]);
   else
      return std::abs(times[idx] - time) < 1e-3 * std::abs(times[idx + 1] - times[idx]);
}

template<int dim>
size_t SpaceTimeMesh<dim>::find_time(double time) const {
   size_t idx = find_nearest_time(time);

   if (!near_enough(time, idx)) {
      std::stringstream err;
      err << "requested time " << time << " not found, nearest is " << times[idx];
      Assert(false, ExcMessage(err.str()));
   }

   return idx;
}

template<int dim>
size_t SpaceTimeMesh<dim>::find_time(double time, size_t low, size_t up, bool increasing) const {
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
      if (increasing)
         return find_time(time, middle, up, increasing);
      else
         return find_time(time, low, middle, increasing);
   else if (time < val)
      if (increasing)
         return find_time(time, low, middle, increasing);
      else
         return find_time(time, middle, up, increasing);
   else
      return middle;
}

template<int dim>
size_t SpaceTimeMesh<dim>::find_nearest_time(double time) const {
   Assert(times.size() > 0, ExcEmptyObject());

   if (times.size() == 1)
      return 0;
   else
      return find_time(time, 0, times.size() - 1, times[1] - times[0] > 0);
}

template class SpaceTimeMesh<1> ;
template class SpaceTimeMesh<2> ;
template class SpaceTimeMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */

