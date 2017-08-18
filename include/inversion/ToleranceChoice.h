/*
 * ToleranceChoice.h
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_TOLERANCECHOICE_H_
#define INCLUDE_INVERSION_TOLERANCECHOICE_H_

#include <vector>

namespace wavepi {
namespace inversion {

class ToleranceChoice {
   public:

      virtual ~ToleranceChoice() = default;

      // reset iteration history
      void reset(double target_discrepancy, double initial_discrepancy);

      // calculate a tolerance from the history (and add it to the tolerance history)
      double get_tolerance();

      // add a new iteration to the history
      void add_iteration(double new_residual, int steps);

   protected:
      // previous values. Note that they will have the same size when calculate_tolerance is called.
      std::vector<double> previous_tolerances;
      std::vector<int> required_steps;
      std::vector<double> discrepancies;

      double target_discrepancy;
      double initial_discrepancy;

      // calculate a new tolerance (possibly using the vectors above)
      virtual double calculate_tolerance() const = 0;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_TOLERANCECHOICE_H_ */
