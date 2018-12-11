/*
 * wavepi_inverse.cpp
 *
 *  Created on: 01.07.2017
 *      Author: thies
 */

#include <boost/program_options.hpp>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>

#include <Version.h>
#include <WavePI.h>
#include <base/DiscretizedFunction.h>
#include <measurements/SensorValues.h>

#include <regex>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace dealii;
using namespace wavepi;
using namespace wavepi::base;
using namespace wavepi::forward;
using namespace wavepi::measurements;

namespace po = boost::program_options;

int main(int argc, char *argv[]) {
   Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);
   size_t mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
   size_t mpi_size = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

   try {
      po::options_description desc(Version::get_identification() + "\nsupported options");

      desc.add_options()("help,h", "produce help message and exit");
      desc.add_options()("version", "print version information and exit");
      desc.add_options()("export", po::value<std::string>()->implicit_value("Text"),
            "generate config file with default values (unless [config] is specified) and exit. Options are "
                  "Text|LaTeX|Description|XML|JSON|ShortText|Diff");
      desc.add_options()("config,c", po::value<std::string>(), "read config from this file");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help") || argc == 1) {
         std::cout << desc << "\n";
         return 1;
      }

      if (vm.count("version")) {
         std::cout << Version::get_identification() << std::endl << std::endl;
         std::cout << Version::get_infos();
         return 1;
      }

      auto prm = std::make_shared<ParameterHandler>();
      SettingsManager::declare_parameters(prm);

      if (vm.count("config")) {
         deallog << "Loading parameter file " << vm["config"].as<std::string>() << std::endl;

         prm->parse_input(vm["config"].as<std::string>());
      } else
         deallog << "Using default parameters" << std::endl;

      auto cfg = std::make_shared<SettingsManager>();
      cfg->get_parameters(prm);

      if (vm.count("export")) {
         ParameterHandler::OutputStyle style = ParameterHandler::Text;
         std::string sstyle = vm["export"].as<std::string>();

         if (sstyle == "LaTeX")
            style = ParameterHandler::LaTeX;
         else if (sstyle == "Description")
            style = ParameterHandler::Description;
         else if (sstyle == "XML")
            style = ParameterHandler::XML;
         else if (sstyle == "Text")
            style = ParameterHandler::Text;
         else if (sstyle == "JSON")
            style = ParameterHandler::JSON;
         else if (sstyle == "ShortText")
            style = ParameterHandler::ShortText;
         else if (sstyle == "Text" || sstyle == "Diff")
            style = ParameterHandler::Text;
         else
            AssertThrow(false, ExcMessage("Invalid config-format: " + sstyle));

         if (style == ParameterHandler::Text || style == ParameterHandler::ShortText) {
            // also print version information
            // (very important for Diff, because default values may have changed and this does not break the config file)
            std::cout << "# WavePI settings file" << std::endl;
            std::cout << "# generated by " << Version::get_identification() << " (built " + Version::get_build_date()
                  << ")" << std::endl;
         }

         std::stringstream sstream;
         prm->print_parameters(sstream, style);
         std::string params = sstream.str();

         if (sstyle == "Diff") {
            // delete the first two lines of params
            params.erase(params.begin(), params.begin() + params.find("\n", params.find("\n") + 1));

            static const std::regex r1("\n[\t ]*#[^\n]+");
            static const std::regex r2("([\t ]*\n)+");

            params = std::regex_replace(params, r1, "\n");
            params = std::regex_replace(params, r2, "\n");

            // filter non-default values
            std::stringstream out;
            std::stringstream ss(params);
            std::string line;

            static const std::regex r_default("(.*# default:.*)|([\t ]*subsection .*)|([\t ]*end[\t ]*)");

            while (std::getline(ss, line, '\n'))
               if (std::regex_match(line, r_default)) out << line << std::endl;

            params = out.str();

            // filter empty sections
            static const std::regex r_sub("[ \t]*subsection [^\n]+\n[\n \t]*end[\t ]*\n");
            size_t length = -1;
            for (size_t i = 0; i < 10 && length != params.size(); i++) {
               length = params.size();
               params = std::regex_replace(params, r_sub, "");
            }
         } else if (style == ParameterHandler::Text) {
            // delete the first two lines of params
            params.erase(params.begin(), params.begin() + params.find("\n", params.find("\n") + 1));

            // delete double new lines
            for (size_t pos = params.rfind("\n\n\n"); pos != std::string::npos; pos = params.rfind("\n\n\n", pos - 1))
               params.erase(params.begin() + pos, params.begin() + pos + 1);

            // delete double new lines after "end"
            for (size_t pos = params.rfind("end\n"); pos != std::string::npos; pos = params.rfind("end\n", pos - 1))
               params.erase(params.begin() + pos + 3, params.begin() + pos + 4);
         }

         std::cout << params;

         return 1;
      }

      // has to be kept in scope
      std::shared_ptr<std::ofstream> logout;

      if (cfg->log_file.size()) {
         if (mpi_size > 1) cfg->log_file = cfg->log_file + std::to_string(mpi_rank);

         logout = std::make_shared<std::ofstream>(cfg->log_file);
         deallog.attach(*logout);
         deallog.depth_file(mpi_rank == 0 ? cfg->log_file_depth : cfg->log_file_depth_mpi);
      }

      deallog.depth_console(cfg->log_console_depth);
      deallog.precision(3);
      deallog.pop();

      if (mpi_rank > 0) {
         deallog << "node " << mpi_rank << " coming online" << std::endl;
         deallog.push("node" + std::to_string(mpi_rank));
         deallog.depth_console(0);
      } else if (mpi_size > 1) deallog << "parallel job on " << mpi_size << " nodes" << std::endl;

      deallog << Version::get_identification() << std::endl;
      deallog << Version::get_infos() << std::endl;

      if (cfg->dimension == 1) {
#ifdef WAVEPI_1D
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<1, SensorValues<1>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<1, DiscretizedFunction<1>> wavepi(cfg);
            wavepi.run();
         } else {
            AssertThrow(false, ExcInternalError())
         }
#else
         AssertThrow(false, ExcMessage("WavePI was compiled without 1D support!"));
#endif
      } else if (cfg->dimension == 2) {
#ifdef WAVEPI_2D
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<2, SensorValues<2>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<2, DiscretizedFunction<2>> wavepi(cfg);
            wavepi.run();
         } else {
            AssertThrow(false, ExcInternalError())
         }
#else
         AssertThrow(false, ExcMessage("WavePI was compiled without 2D support!"));
#endif
      } else if (cfg->dimension == 3) {
#ifdef WAVEPI_3D
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<3, SensorValues<3>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<3, DiscretizedFunction<3>> wavepi(cfg);
            wavepi.run();
         } else {
            AssertThrow(false, ExcInternalError())
         }
#else
         AssertThrow(false, ExcMessage("WavePI was compiled without 3D support!"));
#endif
      } else
         AssertThrow(false, ExcMessage("not built for dimension " + std::to_string(cfg->dimension)));
   } catch (std::exception &exc) {
      if (mpi_rank != 0) std::cerr << "rank " << mpi_rank << ": ";
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
   } catch (...) {
      if (mpi_rank != 0) std::cerr << "rank " << mpi_rank << ": ";
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
   }

   return 0;
}
