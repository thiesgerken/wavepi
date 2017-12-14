/*
 * MaxIterChoice.cpp
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#include <inversion/MaxIterChoice.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/exceptions.h>

namespace wavepi {
namespace inversion {

using namespace dealii;

void MaxIterChoice::reset(double target_discrepancy, double initial_discrepancy) {
   previous_max_iters.clear();
   required_steps.clear();
   discrepancies.clear();

   this->target_discrepancy = target_discrepancy;
   this->initial_discrepancy = initial_discrepancy;
}

int MaxIterChoice::get_max_iter() {
   AssertThrow(discrepancies.size() == previous_max_iters.size(), ExcInternalError());

   int max_iter = calculate_max_iter();

   previous_max_iters.push_back(max_iter);
   return max_iter;
}

void MaxIterChoice::add_iteration(double new_discrepancy, int steps) {
   LogStream::Prefix p = LogStream::Prefix("ToleranceChoice");

   AssertThrow(discrepancies.size() == previous_max_iters.size() - 1, ExcInternalError());

   discrepancies.push_back(new_discrepancy);
   required_steps.push_back(steps);
}

} /* namespace inversion */
} /* namespace wavepi */
