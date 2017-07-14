/*
 * InverseProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_INVERSEPROBLEM_H_
#define INVERSION_INVERSEPROBLEM_H_

#include <memory>

namespace wavepi {
namespace inversion {

template<typename Param, typename Sol>
struct InversionProgress {
      int iteration_number;

      const Param& current_estimate;
      double norm_current_estimate;

      const Sol& current_residual;
      double current_discrepancy;

      const Sol& data;
      double norm_data;

      std::shared_ptr<const Param> exact_param;
      double norm_exact_param; // might be <= 0 if !exact_param
      double current_error;

      InversionProgress(int iteration_number, const Param& current_estimate, double norm_current_estimate,
            const Sol& current_residual, double current_discrepancy, const Sol& data, double norm_data,
            std::shared_ptr<const Param> exact_param, double norm_exact_param)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), data(data), norm_data(norm_data), exact_param(exact_param), norm_exact_param(
                  norm_exact_param) {
         if (exact_param) {
            Param tmp(current_estimate);
            tmp -= *exact_param;
            current_error = tmp.norm();
         }
      }

      InversionProgress(int iteration_number, const Param& current_estimate, double norm_current_estimate,
            const Sol& current_residual, double current_discrepancy, const Sol& data, double norm_data)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), data(data), norm_data(norm_data), exact_param(), norm_exact_param(
                  -0.0), current_error(-0.0) {
      }

};

template<typename Param, typename Sol>
class InverseProblem {
   public:
      virtual ~InverseProblem() {
      }

      virtual Sol forward(const Param& f) = 0;

      // progress indicator that iterative methods can call
      // exact_param might be equal to null_ptr
      // default implementation does nothing
      // should return false, if you want the inversion to abort
      virtual bool progress(InversionProgress<Param, Sol> state __attribute((unused))) {
         return true;
      }

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSEPROBLEM_H_ */
