/*
 * main.cpp
 *
 *  Created on: 11.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <gtest/gtest.h>
#include <fstream>

using namespace dealii;

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  std::ofstream logout("wavepi_test.log", std::ios_base::app);
  deallog.attach(logout);
  deallog.depth_console(0);
  deallog.depth_file(100);
  deallog.precision(3);
  deallog.pop();

  return RUN_ALL_TESTS();
}
