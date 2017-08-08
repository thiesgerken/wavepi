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

#include <util/Version.h>
#include <util/WavePI.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace dealii;
using namespace wavepi::util;
namespace po = boost::program_options;

int main(int argc, char * argv[]) {
   try {
      int log_file_depth;
      int log_console_depth;

      po::options_description desc(Version::get_identification() + "\nsupported options");

      desc.add_options()("help,h", "produce help message and exit");
      desc.add_options()("version", "print version information and exit");
      desc.add_options()("export-config",
            "generate config file with default values (unless [config] is specified) and exit");
      desc.add_options()("config-format", po::value<std::string>(),
            "format for --export-config. Options are Text|LaTeX|Description|XML|JSON|ShortText; default is Text.");

      desc.add_options()("config,c", po::value<std::string>(), "read config from this file");
      desc.add_options()("log,l", po::value<std::string>(), "external log file");
      desc.add_options()("log-file-depth", po::value<int>(&log_file_depth)->default_value(100),
            "log depth that goes to [log]");
      desc.add_options()("log-console-depth", po::value<int>(&log_console_depth)->default_value(2),
            "log depth that goes to stdout");

      po::variables_map vm;
      po::store(po::parse_command_line(argc, argv, desc), vm);
      po::notify(vm);

      if (vm.count("help")) {
         std::cout << desc << "\n";
         return 1;
      }

      if (vm.count("version")) {
         std::cout << Version::get_identification() << std::endl;
         std::cout << Version::get_infos() << std::endl;
         return 1;
      }

      std::ofstream logout;
      if (vm.count("log-file")) {
         logout = std::ofstream(vm["log-file"].as<std::string>());
         deallog.attach(logout);
         deallog.depth_file(log_file_depth);
      }

      auto prm = std::make_shared<ParameterHandler>();
      prm->declare_entry("dimension", "2", Patterns::Integer(1, 3), "problem dimension");
      WavePI<2>::declare_parameters(*prm);

      if (vm.count("config")) {
         deallog << "Loading parameter file " << vm["config"].as<std::string>() << std::endl;

         prm->parse_input(vm["config"].as<std::string>());
      } else {
         deallog << "Using default parameters" << std::endl;

         //   AssertThrow(vm.count("make-config"),
         //       ExcMessage("No config file specified. Use `wavepi --make-config` to create one."));
      }

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

      deallog.depth_console(log_console_depth);
      deallog.precision(3);
      deallog.pop();
      deallog << Version::get_identification() << std::endl;
      // deallog << Version::get_infos() << std::endl;
      // deallog.log_execution_time(true);

      prm->log_parameters(deallog);

      int dim = prm->get_integer("dimension");

      if (dim == 1) {
         WavePI<1> wavepi(prm);
         wavepi.run();
      } else if (dim == 2) {
         WavePI<2> wavepi(prm);
         wavepi.run();
      } else {
         WavePI<3> wavepi(prm);
         wavepi.run();
      }

      // deallog.timestamp();
   } catch (std::exception &exc) {
      std::cerr << "Exception on processing: " << exc.what();
      return 1;
   } catch (...) {
      std::cerr << "Unknown exception!" << std::endl;
      return 1;
   }

   return 0;
}
