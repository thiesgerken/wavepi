/*
 * MaxIterChoice.h
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#ifndef LIB_INVERSION_MAXIMUMITERATIONCHOICE_H_
#define LIB_INVERSION_MAXIMUMITERATIONCHOICE_H_

#include <vector>

namespace wavepi {
namespace inversion {

class MaxIterChoice {
   public:
      virtual ~MaxIterChoice() = default;

      // reset iteration history
      void reset(double target_discrepancy, double initial_discrepancy);

      // calculate a tolerance from the history (and add it to the tolerance history)
      int get_max_iter();

      // add a new iteration to the history
      void add_iteration(double new_residual, int steps);

   protected:
      // previous values. Note that they will have the same size when calculate_max_iter is called.
      std::vector<int> previous_max_iters;
      std::vector<int> required_steps;
      std::vector<double> discrepancies;

      double target_discrepancy;
      double initial_discrepancy;

      // calculate a new maximum iteration count (possibly using the vectors above)
      virtual int calculate_max_iter() const = 0;
};

} /* namespace util */
} /* namespace wavepi */

#endif /* LIB_INVERSION_MAXIMUMITERATIONCHOICE_H_ */
