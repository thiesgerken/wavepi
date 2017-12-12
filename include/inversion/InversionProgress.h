/*
 * InversionProgress.h
 *
 *  Created on: 03.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_INVERSIONPROGRESS_H_
#define INCLUDE_INVERSION_INVERSIONPROGRESS_H_

#include <boost/filesystem/operations.hpp>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/parameter_handler.h>

#include <forward/DiscretizedFunction.h>
#include <util/Helpers.h>

#include <signal.h>
#include <stddef.h>
#include <cstdio>
#include <fstream>
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
using namespace wavepi::util;

template<typename Param, typename Sol, typename Exact = Param>
struct InversionProgress {
      int iteration_number;

      const Param* current_estimate;
      double norm_current_estimate;

      const Sol* current_residual;
      double current_discrepancy;
      double target_discrepancy;

      const Sol* data;
      double norm_data;

      std::shared_ptr<Exact> exact_param;
      double norm_exact_param; // might be <= 0 if !exact_param
      double current_error;

      bool finished;

      InversionProgress(int iteration_number, const Param* current_estimate, double norm_current_estimate,
            const Sol* current_residual, double current_discrepancy, double target_discrepancy,
            const Sol* data, double norm_data, std::shared_ptr<Exact> exact_param, bool finished)
            : iteration_number(iteration_number), current_estimate(current_estimate), norm_current_estimate(
                  norm_current_estimate), current_residual(current_residual), current_discrepancy(
                  current_discrepancy), target_discrepancy(target_discrepancy), data(data), norm_data(
                  norm_data), exact_param(exact_param) {
         if (exact_param)
            current_error = current_estimate->absolute_error(*exact_param, &norm_exact_param);
         else {
            current_error = -0.0;
            norm_exact_param = -0.0;
         }

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

template<typename Param, typename Sol, typename Exact = Param>
class InversionProgressListener {
   public:

      virtual ~InversionProgressListener() = default;

      // progress indicator that iterative methods can call
      // exact_param might be equal to null_ptr
      // should return false, if you want the inversion to abort
      virtual bool progress(InversionProgress<Param, Sol, Exact> state) = 0;

};

template<typename Param, typename Sol, typename Exact = Param>
class GenericInversionProgressListener: public InversionProgressListener<Param, Sol, Exact> {
   public:

      virtual ~GenericInversionProgressListener() = default;

      GenericInversionProgressListener(std::string counter_variable)
            : counter_variable(counter_variable) {
      }

      virtual bool progress(InversionProgress<Param, Sol, Exact> state) {
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

template<typename Param, typename Sol, typename Exact = Param>
class CtrlCProgressListener: public InversionProgressListener<Param, Sol, Exact> {
   public:

      virtual ~CtrlCProgressListener() = default;

      CtrlCProgressListener() {
         if (!handler_installed) {
            struct sigaction sigIntHandler;

            sigIntHandler.sa_handler = CtrlCProgressListener<Param, Sol, Exact>::sighandler;
            sigemptyset(&sigIntHandler.sa_mask);
            sigIntHandler.sa_flags = 0;

            sigaction(SIGINT, &sigIntHandler, NULL);
            handler_installed = true;
         }
      }

      virtual bool progress(InversionProgress<Param, Sol, Exact> state __attribute((unused))) {
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

template<typename Param, typename Sol, typename Exact>
bool CtrlCProgressListener<Param, Sol, Exact>::abort = false;

template<typename Param, typename Sol, typename Exact>
bool CtrlCProgressListener<Param, Sol, Exact>::handler_installed = false;

template<int dim, typename Meas>
class OutputProgressListener: public InversionProgressListener<DiscretizedFunction<dim>, Meas, Function<dim>> {
   public:

      virtual ~OutputProgressListener() = default;

      OutputProgressListener(int interval)
            : interval(interval) {
      }

      OutputProgressListener(ParameterHandler &prm) {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("output");
         {
            prm.declare_entry("interval", "10", Patterns::Integer(0),
                  "output every n iterations, or never if n == 0.");
            prm.declare_entry("last", "true", Patterns::Bool(), "output the last iteration before exit");

            prm.declare_entry("data", "true", Patterns::Bool(),
                  "output the problem's right hand side on the first iteration");
            prm.declare_entry("exact", "true", Patterns::Bool(),
                  "output the problem's exact solution on the first iteration\n(if available and discretized on first grid)");

            prm.declare_entry("estimate", "true", Patterns::Bool(), "output the current estimate");
            prm.declare_entry("residual", "true", Patterns::Bool(), "output the current residual");

            prm.declare_entry("destination", "./step-{{i}}/", Patterns::DirectoryName(),
                  "output path for step {{i}}; has to end with a slash");
         }
         prm.leave_subsection();
      }

      void get_parameters(ParameterHandler &prm) {
         prm.enter_subsection("output");
         {
            interval = prm.get_integer("interval");
            save_last = prm.get_bool("last");

            save_data = prm.get_bool("data");
            save_exact = prm.get_bool("exact");

            save_estimate = prm.get_bool("estimate");
            save_residual = prm.get_bool("residual");

            destination_prefix = prm.get("destination");
         }
         prm.leave_subsection();
      }

      virtual bool progress(InversionProgress<DiscretizedFunction<dim>, Meas, Function<dim>> state) {
         std::map<std::string, std::string> subs;
         subs["i"] = Utilities::int_to_string(state.iteration_number, 4);

         std::string dest = Helpers::replace(destination_prefix, subs);

         if (save_exact && state.iteration_number == 0 && state.exact_param) {
            std::string filename = Helpers::replace(filename_exact, subs);

            boost::filesystem::create_directories(dest);
            deallog << "Saving exact parameter in " << dest << std::endl;
            LogStream::Prefix p = LogStream::Prefix("Output");

            DiscretizedFunction<dim> exact_disc(state.current_estimate->get_mesh(), *state.exact_param);
            exact_disc.write_pvd(dest, filename, "param");
         }

         if (save_data && state.iteration_number == 0) {
            std::string filename = Helpers::replace(filename_data, subs);

            boost::filesystem::create_directories(dest);
            deallog << "Saving data in " << dest << std::endl;
            LogStream::Prefix p = LogStream::Prefix("Output");
            state.data->write_pvd(dest, filename, "data");
         }

         if ((interval > 0 && state.iteration_number % interval == 0) || (save_last && state.finished)) {
            if (save_residual) {
               std::string filename = Helpers::replace(filename_residual, subs);

               boost::filesystem::create_directories(dest);
               deallog << "Saving current residual in " << dest << std::endl;
               LogStream::Prefix p = LogStream::Prefix("Output");
               state.current_residual->write_pvd(dest, filename, "residual");
            }

            if (save_estimate) {
               std::string filename = Helpers::replace(filename_estimate, subs);

               boost::filesystem::create_directories(dest);
               deallog << "Saving current estimate in " << dest << std::endl;
               LogStream::Prefix p = LogStream::Prefix("Output");
               state.current_estimate->write_pvd(dest, filename, "estimate");
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

      bool is_save_data() const {
         return save_data;
      }

      void set_save_data(bool save_data = true) {
         this->save_data = save_data;
      }

   private:

      // all of these have support for expansion of {{i}} for the iteration index.
      // destination_prefix is created if necessary.
      // destination_prefix has to end with a slash!
      std::string filename_estimate = "estimate";
      std::string filename_residual = "residual";
      std::string filename_exact = "param";
      std::string filename_data = "data";
      std::string destination_prefix = "./step-{{i}}/";

      bool save_estimate = true;
      bool save_residual = true;
      bool save_exact = true; // (once)
      bool save_data = true; // (once)

      // save the last one
      bool save_last = true;

      // interval <= 0 -> no interval-based output
      int interval;
};

template<int dim, typename Meas>
class BoundCheckProgressListener: public InversionProgressListener<DiscretizedFunction<dim>, Meas,
      Function<dim>> {
   public:

      virtual ~BoundCheckProgressListener() = default;

      BoundCheckProgressListener(double lower_bound = -std::numeric_limits<double>::infinity(),
            double upper_bound = std::numeric_limits<double>::infinity())
            : lower_bound(lower_bound), upper_bound(upper_bound) {
      }

      BoundCheckProgressListener(ParameterHandler &prm) {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("constraints");
         {
            prm.declare_entry("lower bound", "-1e300", Patterns::Double(),
                  "lower bound for reconstructed parameter");

            prm.declare_entry("upper bound", "1e300", Patterns::Double(),
                  "upper bound for reconstructed parameter");
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

      virtual bool progress(InversionProgress<DiscretizedFunction<dim>, Meas, Function<dim>> state) {
         double est_min = std::numeric_limits<double>::infinity();
         double est_max = -std::numeric_limits<double>::infinity();

         for (size_t i = 0; i < state.current_estimate->length(); i++)
            for (size_t j = 0; j < (*state.current_estimate)[i].size(); j++) {

               if (est_min > (*state.current_estimate)[i][j])
                  est_min = (*state.current_estimate)[i][j];

               if (est_max < (*state.current_estimate)[i][j])
                  est_max = (*state.current_estimate)[i][j];
            }

         deallog << est_min << " <= estimate <= " << est_max << std::endl;

         if (est_min < lower_bound)
            deallog << "constraint estimate >= " << lower_bound << " violated" << std::endl;

         if (est_max > upper_bound)
            deallog << "constraint estimate <= " << upper_bound << " violated" << std::endl;

         return est_min >= lower_bound && est_max <= upper_bound;
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

template<typename Param, typename Sol, typename Exact = Param>
class StatOutputProgressListener: public InversionProgressListener<Param, Sol, Exact> {
   public:

      virtual ~StatOutputProgressListener() = default;

      StatOutputProgressListener(std::string file_prefix)
            : file_prefix(file_prefix) {
      }

      StatOutputProgressListener(ParameterHandler &prm) {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("output");
         {
            prm.declare_entry("statistics", "history", Patterns::Anything(),
                  "Output file for rdisc, (r)norm and (r)error (if available) of estimate. If not empty, this will result in [file].csv and [file].svg.");
         }
         prm.leave_subsection();
      }

      virtual void get_parameters(ParameterHandler &prm) {
         prm.enter_subsection("output");
         {
            file_prefix = prm.get("statistics");
         }
         prm.leave_subsection();
      }

      virtual bool progress(InversionProgress<Param, Sol, Exact> state) {
         if (!file_prefix.size() || state.finished)
            return true;

         std::ofstream csv_file(file_prefix + ".csv", std::ios::out | std::ios::app);

         if (!csv_file)
            return true;

         if (csv_file.tellp() == 0) {
            int num_cols;

            if (state.norm_exact_param > 0) {
               csv_file << "Iteration,rel. Discrepancy,rel. Norm,rel. Error" << std::endl;
               num_cols = 4;
            } else {
               csv_file << "Iteration,rel. Discrepancy,Norm" << std::endl;
               num_cols = 3;
            }

            std::ofstream gplot_file(file_prefix + ".gplot", std::ios::out | std::ios::trunc);
            gplot_file << "set xlabel 'Iteration'" << std::endl;
            gplot_file << "set grid" << std::endl;
            gplot_file << "set term png size 1200,500" << std::endl;
            gplot_file << "set output '" << file_prefix << ".png'" << std::endl;
            gplot_file << "set datafile separator ','" << std::endl;
            gplot_file << "set key outside" << std::endl;

            gplot_file << "plot for [col=2:" << num_cols << "] '" << file_prefix
                  << ".csv' using 1:col with linespoints title columnheader" << std::endl;

            gplot_file << "set term svg size 1200,500 name 'History'" << std::endl;
            gplot_file << "set output '" << file_prefix << ".svg'" << std::endl;
            gplot_file << "replot" << std::endl;
         }

         double norm_exact_param = state.norm_exact_param > 0 ? state.norm_exact_param : 1.0;
         csv_file << state.iteration_number << "," << state.current_discrepancy / state.norm_data << ",";
         csv_file << state.norm_current_estimate / norm_exact_param;

         if (state.norm_exact_param > 0)
            csv_file << "," << state.current_error / norm_exact_param;

         csv_file << std::endl;
         csv_file.close();

         std::string cmd = "cat " + file_prefix + ".gplot | gnuplot > /dev/null";
         std::system(cmd.c_str());

         return true;
      }

      const std::string& get_file_prefix() const {
         return file_prefix;
      }

      void set_file_prefix(const std::string& file_prefix) {
         this->file_prefix = file_prefix;
      }

   protected:
      std::string file_prefix = "stats";
};

template<typename Param, typename Sol, typename Exact = Param>
class InnerStatOutputProgressListener: public StatOutputProgressListener<Param, Sol, Exact> {
   public:

      virtual ~InnerStatOutputProgressListener() = default;

      InnerStatOutputProgressListener(std::string file_prefix)
            : StatOutputProgressListener<Param, Sol, Exact>(file_prefix) {
      }

      InnerStatOutputProgressListener(ParameterHandler &prm)
            : StatOutputProgressListener<Param, Sol, Exact>("") {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("inner output");
         {
            prm.declare_entry("interval", "10", Patterns::Integer(0),
                  "output stats of inner iteration every n outer iterations, or never if n == 0.");

            prm.declare_entry("destination", "./step-{{i}}/", Patterns::DirectoryName(),
                  "output path for step {{i}}; has to end with a slash");

            prm.declare_entry("statistics", "stats", Patterns::FileName(),
                             "output file for statistics (no extension). You can also use {{i}} here.");

         }
         prm.leave_subsection();
      }

      virtual void get_parameters(ParameterHandler &prm) {
         prm.enter_subsection("inner output");
         {
            interval = prm.get_integer("interval");
            destination_prefix = prm.get("destination");
            file_name = prm.get("statistics");
         }
         prm.leave_subsection();
      }

      int get_iteration() const {
         return outer_iteration;
      }

      void set_iteration(int outer_iteration) {
         this->outer_iteration = outer_iteration;

         if (interval > 0 && outer_iteration % interval == 0 && file_name.size() > 0) {
            std::map<std::string, std::string> subs;
            subs["i"] = Utilities::int_to_string(outer_iteration, 4);

            std::string dest = Helpers::replace(destination_prefix, subs);
            std::string f = Helpers::replace(file_name, subs);

            boost::filesystem::create_directories(dest);
            this->set_file_prefix(dest + f);
         } else {
            this->set_file_prefix("");
         }
      }

   private:
      int outer_iteration = 0;

      // params
      int interval = 10;
      std::string destination_prefix = "./step-{{i}}/";
      std::string file_name = "stats";
};

template<typename Param, typename Sol, typename Exact = Param>
class WatchdogProgressListener: public InversionProgressListener<Param, Sol, Exact> {
   public:

      virtual ~WatchdogProgressListener() = default;

      WatchdogProgressListener() = default;

      WatchdogProgressListener(ParameterHandler &prm) {
         get_parameters(prm);
      }

      static void declare_parameters(ParameterHandler &prm) {
         prm.enter_subsection("watchdog");
         {
            prm.declare_entry("discrepancy threshold", "10.0", Patterns::Double(),
                  "threshold for discrepancy: abort if it exceeds this factor times the initial discrepancy. Set to ≤ 1 to disable.");

            prm.declare_entry("discrepancy slope threshold", "0.0", Patterns::Double(),
                  "threshold for slope of discrepancy.");
            prm.declare_entry("discrepancy slope percentage", "0.25", Patterns::Double(),
                  "fraction of discrepancies that should be used to get slope. Set to ≤ 0 to disable slope checking.");
            prm.declare_entry("discrepancy slope min values", "25", Patterns::Integer(),
                  "enable slope checking only if computed from at least this many entries");
         }
         prm.leave_subsection();
      }

      virtual void get_parameters(ParameterHandler &prm) {
         prm.enter_subsection("watchdog");
         {
            initial_disc_factor = prm.get_double("discrepancy threshold");

            disc_max_slope = prm.get_double("discrepancy slope threshold");
            disc_slope_percentage = prm.get_double("discrepancy slope percentage");
            disc_slope_min_values = prm.get_integer("discrepancy slope min values");
         }
         prm.leave_subsection();
      }

      virtual bool progress(InversionProgress<Param, Sol, Exact> state) {
         if (state.iteration_number == 0)
            discrepancies.clear();

         LogStream::Prefix p = LogStream::Prefix("Watchdog");

         discrepancies.push_back(state.current_discrepancy);

         if (initial_disc_factor > 1 && state.current_discrepancy > initial_disc_factor * discrepancies[0]) {
            deallog << "current discrepancy > " << initial_disc_factor << " ⋅ initial discrepancy"
                  << std::endl;
            deallog << "Aborting Iteration!" << std::endl;
            return false;
         }

         if (disc_slope_percentage > 0
               && disc_slope_percentage * discrepancies.size() >= disc_slope_min_values) {
            int n_discs = disc_slope_percentage * discrepancies.size();

            double avg_i = 0;
            double avg_disc = 0;

            for (size_t i = discrepancies.size() - 1 - n_discs; i < discrepancies.size(); i++) {
               avg_i += i / n_discs; // one could calculate this one explicitly, but what gain is in that?
               avg_disc += discrepancies[i] / n_discs;
            }

            double var_i = 0;
            double slope = 0;

            for (size_t i = discrepancies.size() - 1 - n_discs; i < discrepancies.size(); i++) {
               var_i += (i - avg_i) * (i - avg_i);
               slope += (i - avg_i) * (discrepancies[i] - avg_disc);
            }

            slope /= var_i;
            double b = avg_disc - slope * avg_i;

            auto prev_prec = deallog.precision(1);
            deallog << "disc(i) ~ " << slope << " ⋅ i + " << b << " (computed over " << n_discs << " entries)"
                  << std::endl;
            deallog.precision(prev_prec);

            if (slope > disc_max_slope) {
               deallog << "slope is too large; Aborting Iteration!" << std::endl;
               return false;
            }
         }

         return true;
      }

   protected:
      std::vector<double> discrepancies;

      // threshold for discrepancy: abort if it exceeds this factor times the initial discrepancy.
      // set to <= 1 to disable.
      double initial_disc_factor = 10;

      // abort, if the slope of the discrepancy is greater than disc_max_slope.
      // slop is calculated on the last disc_slope_percentage * n entries, if these are more than disc_slope_min_values.
      // set disc_slope_percentage to <= 0 to disable.
      double disc_max_slope = 0.0;
      double disc_slope_percentage = 0.1;
      int disc_slope_min_values = 25;

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_INVERSIONPROGRESS_H_ */
