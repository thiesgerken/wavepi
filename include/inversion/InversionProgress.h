/*
 * InversionProgress.h
 *
 *  Created on: 03.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_INVERSIONPROGRESS_H_
#define INCLUDE_INVERSION_INVERSIONPROGRESS_H_

#include <deal.II/base/logstream.h>

#include <iostream>
#include <memory>
#include <string>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include <forward/DiscretizedFunction.h>

#include <boost/filesystem.hpp>

namespace wavepi {
namespace inversion {
using namespace dealii;
using namespace wavepi::forward;

template<typename Param, typename Sol>
struct InversionProgress {
      int iteration_number;

      const Param* current_estimate;
      double norm_current_estimate;

      const Sol* current_residual;
      double current_discrepancy;
      double target_discrepancy;

      const Sol* data;
      double norm_data;

      std::shared_ptr<const Param> exact_param;
      double norm_exact_param; // might be <= 0 if !exact_param
      double current_error;

      bool finished;

      InversionProgress(int iteration_number, const Param* current_estimate, double norm_current_estimate,
            const Sol* current_residual, double current_discrepancy, double target_discrepancy,
            const Sol* data, double norm_data, std::shared_ptr<const Param> exact_param,
            double norm_exact_param, bool finished)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), target_discrepancy(target_discrepancy), data(data), norm_data(
                  norm_data), exact_param(exact_param), norm_exact_param(norm_exact_param) {
         if (exact_param) {
            Param tmp(*current_estimate);
            tmp -= *exact_param;
            current_error = tmp.norm();
         } else
            current_error = -0.0;

         this->finished = finished;
      }

      InversionProgress(int iteration_number, const Param* current_estimate, double norm_current_estimate,
            Sol* current_residual, double current_discrepancy, double target_discrepancy, const Sol* data,
            double norm_data, bool finished)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), target_discrepancy(target_discrepancy), data(data), norm_data(
                  norm_data), exact_param(), norm_exact_param(-0.0), current_error(-0.0), finished(finished) {
      }

      InversionProgress(const InversionProgress<Param, Sol>& o)
            : iteration_number(o.iteration_number), current_estimate(o.current_estimate), norm_current_estimate(
                  o.norm_current_estimate), current_residual(o.current_residual), current_discrepancy(
                  o.current_discrepancy), target_discrepancy(o.target_discrepancy), data(o.data), norm_data(
                  o.norm_data), exact_param(o.exact_param), norm_exact_param(o.norm_exact_param), current_error(
                  o.current_error), finished(o.finished) {
      }

      InversionProgress<Param, Sol>& operator=(const InversionProgress<Param, Sol>& o) {
         iteration_number = o.iteration_number;
         current_estimate = o.current_estimate;
         norm_current_estimate = o.norm_current_estimate;
         current_residual = o.current_residual;
         current_discrepancy = o.current_discrepancy;
         target_discrepancy = o.target_discrepancy;
         data = o.data;
         norm_data = o.norm_data;
         exact_param = o.exact_param;
         norm_exact_param = o.norm_exact_param;
         current_error = o.current_error;
         finished = o.finished;

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
         if (!state.finished) {
            deallog << counter_variable << "=" << state.iteration_number << ": rdisc="
                  << state.current_discrepancy / state.norm_data;

            if (state.norm_exact_param > 0.0) {
               deallog << ", rnorm=" << state.norm_current_estimate / state.norm_exact_param << ", rerr="
                     << state.current_error / state.norm_exact_param;
            } else
               deallog << ", norm=" << state.norm_current_estimate;
         } else if (state.current_discrepancy <= state.target_discrepancy)
            deallog << "Target discrepancy reached after " << state.iteration_number << " iteration"
                  << (state.iteration_number != 1 ? "s" : "") << ".";
         else
            deallog << "Unsuccessful termination after " << state.iteration_number << " iteration"
                  << (state.iteration_number != 1 ? "s" : "") << "!";

         deallog << std::endl;
         return true;
      }

   private:
      std::string counter_variable;
};

template<typename Param, typename Sol>
class CtrlCProgressListener: public InversionProgressListener<Param, Sol> {
   public:
      virtual ~CtrlCProgressListener() {
      }

      CtrlCProgressListener() {
         if (!handler_installed) {
            struct sigaction sigIntHandler;

            sigIntHandler.sa_handler = CtrlCProgressListener<Param, Sol>::sighandler;
            sigemptyset(&sigIntHandler.sa_mask);
            sigIntHandler.sa_flags = 0;

            sigaction(SIGINT, &sigIntHandler, NULL);
            handler_installed = true;
         }
      }

      virtual bool progress(InversionProgress<Param, Sol> state __attribute((unused))) {
         return !abort;
      }

   private:
      static bool abort;
      static bool handler_installed;

      static void sighandler(int s) {
         if (abort) {
            printf("\nCaught signal %d. Issuing a hard-abort.\n", s);

            AssertThrow(false, ExcMessage("Aborting computation due to signal"))
         }

         else {
            printf("\nCaught signal %d. Issuing a soft-abort.\n", s);

            abort = true;
         }
      }

};

template<typename Param, typename Sol>
bool CtrlCProgressListener<Param, Sol>::abort = false;

template<typename Param, typename Sol>
bool CtrlCProgressListener<Param, Sol>::handler_installed = false;

template<int dim>
class OutputProgressListener: public InversionProgressListener<DiscretizedFunction<dim>,
      DiscretizedFunction<dim>> {
   public:
      virtual ~OutputProgressListener() {
      }

      OutputProgressListener(int interval)
            : interval(interval) {
      }

      virtual bool progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state) {
         LogStream::Prefix p = LogStream::Prefix("Output");
         std::map<std::string, std::string> subs;
         subs["i"] = Utilities::int_to_string(state.iteration_number, 4);

         std::string dest = replace(destination_prefix, subs);

         if (save_exact && state.iteration_number == 0 && state.exact_param) {
            std::string filename = dest + replace(filename_exact, subs);

            boost::filesystem::create_directories(dest);
            state.exact_param->write_pvd(filename, "exact_param");
         }

         if ((interval > 0 && state.iteration_number % interval == 0) || (save_last && state.finished)) {
            if (save_residual) {
               std::string filename = dest + replace(filename_residual, subs);

               boost::filesystem::create_directories(dest);
               state.current_residual->write_pvd(filename, "exact_param");
            }

            if (save_estimate) {
               std::string filename = dest + replace(filename_estimate, subs);

               boost::filesystem::create_directories(dest);
               state.current_estimate->write_pvd(filename, "estimate");
            }
         }

         return true;
      }

      const std::string& get_destination_prefix() const {
         return destination_prefix;
      }

      void set_destination_prefix(const std::string& destination_prefix = "./step-{{i}}/") {
         this->destination_prefix = destination_prefix;
      }

      const std::string& get_filename_estimate() const {
         return filename_estimate;
      }

      void set_filename_estimate(const std::string& filename_estimate = "estimate") {
         this->filename_estimate = filename_estimate;
      }

      const std::string& get_filename_exact() const {
         return filename_exact;
      }

      void set_filename_exact(const std::string& filename_exact = "param") {
         this->filename_exact = filename_exact;
      }

      const std::string& get_filename_residual() const {
         return filename_residual;
      }

      void set_filename_residual(const std::string& filename_residual = "residual") {
         this->filename_residual = filename_residual;
      }

      int get_interval() const {
         return interval;
      }

      void set_interval(int interval) {
         this->interval = interval;
      }

      bool is_save_estimate() const {
         return save_estimate;
      }

      void set_save_estimate(bool save_estimate = true) {
         this->save_estimate = save_estimate;
      }

      bool is_save_exact() const {
         return save_exact;
      }

      void set_save_exact(bool save_exact = true) {
         this->save_exact = save_exact;
      }

      bool is_save_last() const {
         return save_last;
      }

      void set_save_last(bool save_last = true) {
         this->save_last = save_last;
      }

      bool is_save_residual() const {
         return save_residual;
      }

      void set_save_residual(bool save_residual = true) {
         this->save_residual = save_residual;
      }

   private:

      // all of these have support for expansion of {{i}} for the iteration index.
      // destination_prefix is created if necessary.
      // destination_prefix has to end with a slash!
      std::string filename_estimate = "estimate";
      std::string filename_residual = "residual";
      std::string filename_exact = "param";
      std::string destination_prefix = "./step-{{i}}/";

      bool save_estimate = true;
      bool save_residual = true;
      bool save_exact = true; // (once)

      // save the last one
      bool save_last = true;

      // interval <= 0 -> no interval-based output
      int interval;

      std::string replace(std::string const &in, std::map<std::string, std::string> const &subst) {
         std::ostringstream out;
         size_t pos = 0;
         for (;;) {
            size_t subst_pos = in.find("{{", pos);
            size_t end_pos = in.find("}}", subst_pos);
            if (end_pos == std::string::npos)
               break;

            out.write(&*in.begin() + pos, subst_pos - pos);

            subst_pos += strlen("{{");
            std::map<std::string, std::string>::const_iterator subst_it = subst.find(
                  in.substr(subst_pos, end_pos - subst_pos));

            AssertThrow(subst_it != subst.end(), ExcMessage("undefined substitution"))

            out << subst_it->second;
            pos = end_pos + strlen("}}");
         }
         out << in.substr(pos, std::string::npos);
         return out.str();
      }
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_INVERSIONPROGRESS_H_ */
