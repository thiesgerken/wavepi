/*
 * SpaceTimeMesh.h
 *
 *  Created on: 12.07.2017
 *      Author: thies
 */

#ifndef FORWARD_SPACETIMEMESH_H_
#define FORWARD_SPACETIMEMESH_H_

#include <deal.II/lac/sparse_matrix.h>
#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class SpaceTimeMesh {
   public:
      SpaceTimeMesh(std::vector<double> times);
      virtual ~SpaceTimeMesh();

      inline const std::vector<double>& get_times() const {
         return times;
      }

      virtual std::shared_ptr<SparseMatrix<double>> get_mass_matrix(int time_index) = 0;

      size_t find_time(double time) const;
      size_t find_nearest_time(double time) const;
      bool near_enough(double time, size_t idx) const;

   protected:
      std::vector<double> times;

      size_t find_time(double time, size_t low, size_t up, bool increasing) const;

};

} /* namespace wavepi */
} /* namespace forward */

#endif /* LIB_FORWARD_SPACETIMEMESH_H_ */
