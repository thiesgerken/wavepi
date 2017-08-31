/*
 * SpaceTimeGrid.h
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_SPACETIMEGRID_H_
#define INCLUDE_UTIL_SPACETIMEGRID_H_

#include <deal.II/base/point.h>
#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {
namespace util {
using namespace dealii;

template<int dim>
class SpaceTimeGrid {
   public:

      virtual ~SpaceTimeGrid() = default;

      /**
       * @param times times used in this grid in ascending order.
       * @param spatial points at each time step (same length as `times`, inner vector may be differently sized)
       */
      SpaceTimeGrid(const std::vector<double> &times, const std::vector<std::vector<Point<dim>>> &points);

      /**
       * Generate a structured grid.
       *
       * @param times the time points.
       * @param spatial_points nodal points in each dimension (has to be of length `dim`).
       */
      SpaceTimeGrid(const std::vector<double> &times, const std::vector<std::vector<double>> &spatial_points);

      size_t size() const {
         return space_time_points.size();
      }

      Point<dim + 1> operator[](size_t i) const {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return space_time_points[i];
      }

      /**
       * If this grid is structured (i.e. was created by `make_grid`), return the nodal points in each direction.
       * returns empty vector otherwise.
       */
      const std::vector<std::vector<double> >& get_grid_extents() const {
         return grid_extents;
      }

      const std::vector<double>& get_times() const {
         return times;
      }

      const std::vector<Point<dim + 1> >& get_space_time_points() const {
         return space_time_points;
      }

      const std::vector<std::vector<Point<dim> > >& get_points() const {
         return points;
      }

   private:
      std::vector<double> times;
      std::vector<std::vector<Point<dim>>> points;
      std::vector<Point<dim + 1>> space_time_points; // waste of memory, but easier access

      std::vector<std::vector<double>> grid_extents;
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_SPACETIMEGRID_H_ */
