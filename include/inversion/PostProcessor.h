/*
 * PostProcessor.h
 *
 *  Created on: 08.12.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_POSTPROCESSOR_H_
#define INCLUDE_INVERSION_POSTPROCESSOR_H_

#include <boost/filesystem/operations.hpp>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <forward/DiscretizedFunction.h>

#include <signal.h>
#include <stddef.h>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <memory>
#include <sstream>
#include <string>

namespace wavepi {
namespace inversion {
using namespace dealii;
using namespace wavepi::forward;

template<typename Param>
class PostProcessor {
   public:

      virtual ~PostProcessor() = default;

      virtual void post_process(int iteration_number, Param* current_estimate,
            double norm_current_estimate) = 0;

};

template<int dim>
class BoundEnforcingPostProcessor: public PostProcessor<DiscretizedFunction<dim>> {
   public:

      virtual ~BoundEnforcingPostProcessor() = default;

      BoundEnforcingPostProcessor(double lower_bound = -std::numeric_limits<double>::infinity(),
            double upper_bound = std::numeric_limits<double>::infinity())
            : lower_bound(lower_bound), upper_bound(upper_bound) {
      }

      BoundEnforcingPostProcessor(ParameterHandler &prm) {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("constraints");
         {
            prm.declare_entry("lower bound", "-1e300", Patterns::Double(),
                  "lower bound for reconstructed parameter. Not enforced if smaller than -1e100.");

            prm.declare_entry("upper bound", "1e300", Patterns::Double(),
                  "upper bound for reconstructed parameter. Not enforced if greater than +1e100.");
         }
         prm.leave_subsection();
      }

      void get_parameters(ParameterHandler &prm) {
         prm.enter_subsection("constraints");
         {
            lower_bound = prm.get_double("lower bound");
            upper_bound = prm.get_double("upper bound");
         }
         prm.leave_subsection();
      }

      virtual void post_process(int iteration_number __attribute__((unused)), DiscretizedFunction<dim>* current_estimate,
            double norm_current_estimate __attribute__((unused))) {
         if (lower_bound <= -1e100 && upper_bound >= 1e100)
            return;

         // these will not contain the min and max of estimate,
         // but rather something like max(lower_bound, min(estimate)) and min(upper_bound, max(estimate))
         double est_min = std::numeric_limits<double>::infinity();
         double est_max = -std::numeric_limits<double>::infinity();

         for (size_t i = 0; i < current_estimate->length(); i++)
            for (size_t j = 0; j < (*current_estimate)[i].size(); j++) {
               if (lower_bound > (*current_estimate)[i][j]) {
                  if (est_min > (*current_estimate)[i][j])
                     est_min = (*current_estimate)[i][j];

                  (*current_estimate)[i][j] = lower_bound;
               }

               if (upper_bound < (*current_estimate)[i][j]) {
                  if (est_max < (*current_estimate)[i][j])
                     est_max = (*current_estimate)[i][j];

                  (*current_estimate)[i][j] = upper_bound;
               }
            }

         if (est_min < lower_bound)
            deallog << "Adjusted lower bound of estimate from " << est_min << " to " << lower_bound
                  << std::endl;

         if (est_max > upper_bound)
            deallog << "Adjusted upper bound of estimate from " << est_max << " to " << upper_bound
                  << std::endl;
      }

      double get_lower_bound() const {
         return lower_bound;
      }

      void set_lower_bound(double lower_bound) {
         this->lower_bound = lower_bound;
      }

      double get_upper_bound() const {
         return upper_bound;
      }

      void set_upper_bound(double upper_bound) {
         this->upper_bound = upper_bound;
      }

   private:

      double lower_bound;
      double upper_bound;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif
