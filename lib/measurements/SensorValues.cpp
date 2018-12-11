/*
 * SensorValues.cpp
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>

#include <measurements/SensorValues.h>

#include <stddef.h>
#include <tgmath.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {

template<int dim>
SensorValues<dim> SensorValues<dim>::noise(std::shared_ptr<SensorDistribution<dim>> grid) {
   SensorValues<dim> res(grid);

   // std::default_random_engine generator(time.time_since_epoch().count() % 1000000);
   std::default_random_engine generator(2307);
   std::uniform_real_distribution<double> distribution(-1, 1);

   for (size_t i = 0; i < grid->size(); i++)
      res[i] = distribution(generator);

   return res;
}

template<int dim>
SensorValues<dim> SensorValues<dim>::noise(const SensorValues<dim>& like) {
   return noise(like.grid);
}

template<int dim>
SensorValues<dim> SensorValues<dim>::noise(const SensorValues<dim>& like, double norm) {
   auto res = noise(like.grid);
   res *= norm / res.norm();

   return res;
}

template<int dim>
double SensorValues<dim>::relative_error(const SensorValues<dim>& other) const {
   SensorValues<dim> tmp(*this);
   tmp -= other;

   double denom = this->norm();
   return tmp.norm() / (denom == 0.0 ? 1.0 : denom);
}

template<int dim>
void SensorValues<dim>::write_pvd(std::string path, std::string filename, std::string name) const {
   AssertThrow(grid, ExcNotInitialized());

   grid->write_pvd(elements, path, filename, name);
}

#ifdef WAVEPI_MPI
template<int dim>
void SensorValues<dim>::mpi_irecv(size_t source, std::vector<MPI_Request>& reqs) {
   AssertThrow(reqs.size() == 0, ExcInternalError());

   reqs.emplace_back();
   MPI_Irecv(&elements[0], elements.size(), MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &reqs[0]);
}

template<int dim>
void SensorValues<dim>::mpi_send(size_t destination) {
   MPI_Send(&elements[0], elements.size(), MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);
}

template<int dim>
void SensorValues<dim>::mpi_isend(size_t destination, std::vector<MPI_Request>& reqs) {
   AssertThrow(reqs.size() == 0, ExcInternalError());

   reqs.emplace_back();
   MPI_Isend(&elements[0], elements.size(), MPI_DOUBLE, destination, 1, MPI_COMM_WORLD, &reqs[0]);
}

template<int dim>
void SensorValues<dim>::mpi_all_reduce(SensorValues<dim> source, MPI_Op op) {
   MPI_Allreduce(&source.elements[0], &elements[0], elements.size(), MPI_DOUBLE, op, MPI_COMM_WORLD);
}

template<int dim>
void SensorValues<dim>::mpi_bcast(size_t root) {
   MPI_Bcast(&elements[0], elements.size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
}

#endif

template class SensorValues<1> ;
template class SensorValues<2> ;
template class SensorValues<3> ;

} /* namespace measurements */
} /* namespace wavepi */
