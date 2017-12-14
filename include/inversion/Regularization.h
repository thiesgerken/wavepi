/*
 * Regularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_REGULARIZATION_H_
#define INVERSION_REGULARIZATION_H_

#include <inversion/InversionProgress.h>
#include <inversion/PostProcessor.h>

#include <climits>
#include <list>
#include <memory>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
// assumed to be some kind of iterative method
template<typename Param, typename Sol, typename Exact = Param>
class Regularization {
   public:

      virtual ~Regularization() = default;

      /**
       * Run the inversion algorithm.
       * Implementations of this method should call `progress` throughout the iteration (if the method is iterative).
       *
       * @param data Right hand side of the problem.
       * @param target_discrepancy Absolute discrepancy after which the inversion should be stopped.
       * @param exact_param The exact parameter, if known.
       * @param norm_exact Norm of the exact parameter, if known. Is used to output relative errors.
       * @param status_out Output for the last status.
       */
      virtual Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<Exact> exact_param,
            std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) = 0;

      Param invert(const Sol& data, double target_discrepancy, std::shared_ptr<Exact> exact_param) {
         return invert(data, target_discrepancy, exact_param, nullptr);
      }

      Param invert(const Sol& data, double target_discrepancy,
            std::shared_ptr<InversionProgress<Param, Sol, Exact>> status_out) {
         return invert(data, target_discrepancy, nullptr, status_out);
      }

      Param invert(const Sol& data, double target_discrepancy) {
         return invert(data, target_discrepancy, nullptr, nullptr);
      }

      void remove_listener(std::shared_ptr<InversionProgressListener<Param, Sol, Exact>> listener) {
         progress_listeners.remove(listener);
      }

      void add_listener(std::shared_ptr<InversionProgressListener<Param, Sol, Exact>> listener) {
         progress_listeners.push_back(listener);
      }

      void remove_post_processor(std::shared_ptr<PostProcessor<Param>> processor) {
         post_processors.remove(processor);
      }

      void add_post_processor(std::shared_ptr<PostProcessor<Param>> processor) {
         post_processors.push_back(processor);
      }

   protected:
      bool progress(InversionProgress<Param, Sol, Exact> state) {
         bool continue_iteration = true;

         for (auto listener : progress_listeners)
            continue_iteration &= listener->progress(state);

         return continue_iteration;
      }

      void post_process(int iteration_number, Param* current_estimate, double norm_current_estimate) {
         for (auto processor : post_processors)
            processor->post_process(iteration_number, current_estimate, norm_current_estimate);
      }

   private:
      std::list<std::shared_ptr<InversionProgressListener<Param, Sol, Exact>>> progress_listeners;
      std::list<std::shared_ptr<PostProcessor<Param>>> post_processors;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_REGULARIZATION_H_ */
