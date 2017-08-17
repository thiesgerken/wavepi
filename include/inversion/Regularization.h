/*
 * Regularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_REGULARIZATION_H_
#define INVERSION_REGULARIZATION_H_

#include <inversion/InversionProgress.h>

#include <climits>
#include <list>
#include <memory>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
// assumed to be some kind of iterative method
template<typename Param, typename Sol>
class Regularization {
   public:
      
      virtual ~Regularization() = default;

      // status_out: output for the last status
      // should call `progress` throughout the iteration (if the method is iterative)
      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<const Param> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol>> status_out) = 0;

      Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<const Param> exact_param) {
         return invert(data, target_discrepancy, exact_param, nullptr);
      }

      int get_max_iterations() const {
         return max_iterations;
      }

      void set_max_iterations(int max_iterations) {
         this->max_iterations = max_iterations;
      }

      bool get_abort_discrepancy_doubles() const {
         return abort_discrepancy_doubles;
      }

      void set_abort_discrepancy_doubles(bool abort_discrepancy_doubles) {
         this->abort_discrepancy_doubles = abort_discrepancy_doubles;
      }

      bool get_abort_increasing_discrepancy() const {
         return abort_increasing_discrepancy;
      }

      void set_abort_increasing_discrepancy(bool abort_increasing_discrepancy) {
         this->abort_increasing_discrepancy = abort_increasing_discrepancy;
      }

      void remove_listener(std::shared_ptr<InversionProgressListener<Param, Sol>> listener) {
         progress_listeners.remove(listener);
      }

      void add_listener(std::shared_ptr<InversionProgressListener<Param, Sol>> listener) {
         progress_listeners.push_back(listener);
      }

   protected:
      int max_iterations = INT_MAX;
      bool abort_increasing_discrepancy = false; // abort if the discrepancy is not decreasing anymore?
      bool abort_discrepancy_doubles = false; // abort if the discrepancy is higher than twice the start discrepancy?

      bool progress(InversionProgress<Param, Sol> state) {
         bool continue_iteration = true;

         for (auto listener : progress_listeners)
            continue_iteration &= listener->progress(state);

         return continue_iteration;
      }

   private:
      std::list<std::shared_ptr<InversionProgressListener<Param, Sol>>> progress_listeners;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_REGULARIZATION_H_ */
