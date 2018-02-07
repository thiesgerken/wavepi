/*
 * ToleranceChoice.cpp
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#include <inversion/ToleranceChoice.h>

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/exceptions.h>
#include <fstream>

namespace wavepi {
namespace inversion {

using namespace dealii;

void ToleranceChoice::reset(double target_discrepancy, double initial_discrepancy) {
   previous_tolerances.clear();
   required_steps.clear();
   discrepancies.clear();

   this->target_discrepancy = target_discrepancy;
   this->initial_discrepancy = initial_discrepancy;
}

double ToleranceChoice::get_tolerance() {
   AssertThrow(discrepancies.size() == previous_tolerances.size(), ExcInternalError());

   double tol = calculate_tolerance();

   previous_tolerances.push_back(tol);
   return tol;
}

void ToleranceChoice::declare_parameters(ParameterHandler &prm) {
   prm.declare_entry("tolerance output", "history_reginn", Patterns::Anything(),
         "output of tolerances and linear steps (csv and gnuplot)");
}

void ToleranceChoice::get_parameters(ParameterHandler &prm) {
   tolerance_prefix = prm.get("tolerance output");
}

void ToleranceChoice::add_iteration(double new_discrepancy, int steps) {
   LogStream::Prefix p = LogStream::Prefix("ToleranceChoice");

   AssertThrow(discrepancies.size() == previous_tolerances.size() - 1, ExcInternalError());

   discrepancies.push_back(new_discrepancy);
   required_steps.push_back(steps);

   if (!tolerance_prefix.size())
      return;

   // truncate if i == 0
   auto opts = discrepancies.size() == 1 ? std::ios::trunc : std::ios::app;
   std::ofstream csv_file(tolerance_prefix + ".csv", std::ios::out | opts);

   if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) != 0)
      return;

   if (!csv_file) {
      deallog << "Could not open " + tolerance_prefix + ".csv" + " for output!" << std::endl;
      return;
   }

   if (csv_file.tellp() == 0) {
      csv_file << "Tolerance,Required Steps" << std::endl;

      std::ofstream gplot_file(tolerance_prefix + ".gplot", std::ios::out | std::ios::trunc);

      if (gplot_file) {
         gplot_file << "set xlabel 'Iteration'" << std::endl;
         gplot_file << "set grid" << std::endl;
         gplot_file << "set term png size 1200,500" << std::endl;
         gplot_file << "set output '" << tolerance_prefix << ".png'" << std::endl;
         gplot_file << "set datafile separator ','" << std::endl;
         gplot_file << "set key outside" << std::endl;

         gplot_file << "plot for [col=1:2] '" << tolerance_prefix
               << ".csv' using 0:col with linespoints title columnheader" << std::endl;

         gplot_file << "set term svg size 1200,500 name 'REGINN'" << std::endl;
         gplot_file << "set output '" << tolerance_prefix << ".svg'" << std::endl;
         gplot_file << "replot" << std::endl;
      } else
         deallog << "Could not open " + tolerance_prefix + ".gplot" + " for output!" << std::endl;
   }

   csv_file << previous_tolerances[previous_tolerances.size() - 1] << ","
         << required_steps[required_steps.size() - 1];
   csv_file << std::endl;
   csv_file.close();

   std::string cmd = "cat " + tolerance_prefix + ".gplot | gnuplot > /dev/null 2>&1";
   if (std::system(cmd.c_str()) != 0)
      deallog << "gnuplot exited with status code != 0 " << std::endl;
}

} /* namespace problems */
} /* namespace wavepi */
