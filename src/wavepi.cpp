/*
 * wavepi_inverse.cpp
 *
 *  Created on: 01.07.2017
 *      Author: thies
 */

#include <boost/program_options.hpp>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>

#include <forward/DiscretizedFunction.h>

#include <inversion/InversionProgress.h>

#include <util/Version.h>
#include <measurements/MeasuredValues.h>
#include <WavePI.h>

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>

using namespace dealii;
using namespace wavepi;
using namespace wavepi::util;

namespace po = boost::program_options;

int main(int argc, char * argv[]) {
   try {
      po::options_description desc(Version::get_identification() + "\nsupported options");

      desc.add_options()("help,h", "produce help message and exit");
      desc.add_options()("version", "print version information and exit");
      desc.add_options()("export-config",
            "generate config file with default values (unless [config] is specified) and exit");
      desc.add_options()("config-format", po::value<std::string>(),
            "format for --export-config. Options are Text|LaTeX|Description|XML|JSON|ShortText; default is Text.");
      desc.add_options()("config,c", po::value<std::string>(), "read config from this file");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help") || argc == 1) {
         std::cout << desc << "\n";
         return 1;
      }

      if (vm.count("version")) {
         std::cout << Version::get_identification() << std::endl;
         std::cout << Version::get_infos() << std::endl;
         return 1;
      }

      auto prm = std::make_shared<ParameterHandler>();
      SettingsManager::declare_parameters(prm);

      if (vm.count("config")) {
         deallog << "Loading parameter file " << vm["config"].as<std::string>() << std::endl;

         prm->parse_input(vm["config"].as<std::string>());
      } else
         deallog << "Using default parameters" << std::endl;

      if (vm.count("export-config")) {
         ParameterHandler::OutputStyle style = ParameterHandler::Text;

         if (vm.count("config-format")) {
            std::string sstyle = vm["config-format"].as<std::string>();

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
            else if (sstyle == "Text")
               style = ParameterHandler::Text;
            else
               AssertThrow(false, ExcMessage("Invalid --config-format: " + sstyle));
         }

         prm->print_parameters(std::cout, style);
         return 1;
      }

      AssertThrow(vm.count("config-format") == 0,
            ExcMessage("--config-format is useless without --export-config"));

      auto cfg = std::make_shared<SettingsManager>();
      cfg->get_parameters(prm);

      // has to be kept in scope
      std::ofstream logout;

      if (cfg->log_file.size()) {
         logout = std::ofstream(cfg->log_file);
         deallog.attach(logout);
         deallog.depth_file(cfg->log_file_depth);
      }

      deallog.depth_console(cfg->log_console_depth);

      deallog.precision(3);
      deallog.pop();
      deallog << Version::get_identification() << std::endl;
      // deallog << Version::get_infos() << std::endl;
      // deallog.log_execution_time(true);

      if (cfg->dimension == 1) {
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<1, MeasuredValues<1>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<1, DiscretizedFunction<1>> wavepi(cfg);
            wavepi.run();
         } else
            AssertThrow(false, ExcInternalError());
      } else if (cfg->dimension == 2) {
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<2, MeasuredValues<2>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<2, DiscretizedFunction<2>> wavepi(cfg);
            wavepi.run();
         } else
            AssertThrow(false, ExcInternalError());
      } else if (cfg->dimension == 3) {
         if (cfg->measure_type == SettingsManager::MeasureType::vector) {
            WavePI<3, MeasuredValues<3>> wavepi(cfg);
            wavepi.run();
         } else if (cfg->measure_type == SettingsManager::MeasureType::discretized_function) {
            WavePI<3, DiscretizedFunction<3>> wavepi(cfg);
            wavepi.run();
         } else
            AssertThrow(false, ExcInternalError());
      } else
         AssertThrow(false, ExcInternalError());

      // deallog.timestamp();
   } catch (std::exception &exc) {
      std::cerr << "Exception on processing: " << exc.what() << std::endl;
      return 1;
   } catch (...) {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
   }

   return 0;
}
