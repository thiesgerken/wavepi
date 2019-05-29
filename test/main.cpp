/*
 * main.cpp
 *
 *  Created on: 11.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <gtest/gtest.h>
#include <fstream>
#include <Version.h>

using namespace dealii;
using namespace wavepi;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::ofstream logout("wavepi-test.log", std::ios_base::trunc);
  deallog.attach(logout);
  deallog.depth_console(0);
  deallog.depth_file(100);
  deallog.precision(3);
  deallog.pop();

  deallog << Version::get_identification() << std::endl;
  deallog << Version::get_infos() << std::endl;

  Timer timer; timer.start();
  int rc = RUN_ALL_TESTS();
  timer.stop();

  deallog << "Total wall time: " << timer.wall_time() << "s, cpu time: " << timer.cpu_time() << "s" << std::endl;

  return rc;
}
