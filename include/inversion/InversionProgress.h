/*
 * InversionProgress.h
 *
 *  Created on: 03.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_INVERSIONPROGRESS_H_
#define INCLUDE_INVERSION_INVERSIONPROGRESS_H_

#include <deal.II/base/logstream.h>

#include <memory>
#include <string>

namespace wavepi {
namespace inversion {
using namespace dealii;

template<typename Param, typename Sol>
struct InversionProgress {
      int iteration_number;

      const Param* current_estimate;
      double norm_current_estimate;

      const Sol* current_residual;
      double current_discrepancy;

      const Sol* data;
      double norm_data;

      std::shared_ptr<const Param> exact_param;
      double norm_exact_param; // might be <= 0 if !exact_param
      double current_error;

      InversionProgress(int iteration_number, const Param* current_estimate, double norm_current_estimate,
            const Sol* current_residual, double current_discrepancy, const Sol* data, double norm_data,
            std::shared_ptr<const Param> exact_param, double norm_exact_param)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), data(data), norm_data(norm_data), exact_param(exact_param), norm_exact_param(
                  norm_exact_param) {
         if (exact_param) {
            Param tmp(*current_estimate);
            tmp -= *exact_param;
            current_error = tmp.norm();
         } else
            current_error = -0.0;
      }

      InversionProgress(int iteration_number, const Param* current_estimate, double norm_current_estimate,
            Sol* current_residual, double current_discrepancy, const Sol* data, double norm_data)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), data(data), norm_data(norm_data), exact_param(), norm_exact_param(
                  -0.0), current_error(-0.0) {
      }

      InversionProgress(const InversionProgress<Param, Sol>& o)
            : iteration_number(o.iteration_number), current_estimate(o.current_estimate), norm_current_estimate(
                  o.norm_current_estimate), current_residual(o.current_residual), current_discrepancy(
                  o.current_discrepancy), data(o.data), norm_data(o.norm_data), exact_param(o.exact_param), norm_exact_param(
                  o.norm_exact_param), current_error(o.current_error) {
      }

      InversionProgress<Param, Sol>& operator=(const InversionProgress<Param, Sol>& o) {
         iteration_number = o.iteration_number;
         current_estimate = o.current_estimate;
         norm_current_estimate = o.norm_current_estimate;
         current_residual = o.current_residual;
         current_discrepancy = o.current_discrepancy;
         data = o.data;
         norm_data = o.norm_data;
         exact_param = o.exact_param;
         norm_exact_param = o.norm_exact_param;
         current_error = o.current_error;

         return *this;
      }
};

template<typename Param, typename Sol>
class InversionProgressListener {
   public:
      virtual ~InversionProgressListener() {
      }
      ;

      // progress indicator that iterative methods can call
      // exact_param might be equal to null_ptr
      // should return false, if you want the inversion to abort
      virtual bool progress(InversionProgress<Param, Sol> state) = 0;

};

template<typename Param, typename Sol>
class GenericInversionProgressListener: public InversionProgressListener<Param, Sol> {
   public:
      virtual ~GenericInversionProgressListener() {
      }

      GenericInversionProgressListener(std::string counter_variable)
            : counter_variable(counter_variable) {
      }

      virtual bool progress(InversionProgress<Param, Sol> state) {
         deallog << counter_variable << "=" << state.iteration_number << ": rdisc="
               << state.current_discrepancy / state.norm_data;

         if (state.norm_exact_param > 0.0) {
            deallog << ", rnorm=" << state.norm_current_estimate / state.norm_exact_param << ", rerr="
                  << state.current_error / state.norm_exact_param;
         } else
            deallog << ", norm=" << state.norm_current_estimate;

         deallog << std::endl;
         return true;
      }

   private:
      std::string counter_variable;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_INVERSIONPROGRESS_H_ */
