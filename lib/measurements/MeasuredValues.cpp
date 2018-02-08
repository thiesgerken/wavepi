/*
 * MeasuredValues.cpp
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <measurements/MeasuredValues.h>
#include <stddef.h>
#include <tgmath.h>
#include <util/SpaceTimeGrid.h>
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

template <int dim>
MeasuredValues<dim> MeasuredValues<dim>::noise(const MeasuredValues<dim>& like) {
  MeasuredValues<dim> res(like.grid);

  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(-1, 1);

  for (size_t i = 0; i < like.size(); i++)
    res[i] = distribution(generator);

  return res;
}

template <int dim>
MeasuredValues<dim> MeasuredValues<dim>::noise(const MeasuredValues<dim>& like, double norm) {
  auto res = noise(like);
  res *= norm / res.norm();

  return res;
}

template <int dim>
void MeasuredValues<dim>::write_vts(std::string path, std::string filename, std::string name) const {
  // currently only implemented for structured grids
  // ( ... and only in 1D )
  AssertThrow(grid->get_grid_extents().size(), ExcNotImplemented());
  AssertThrow(dim == 1, ExcInternalError());
  Assert(grid->size() == elements.size(), ExcInternalError());

  std::ofstream fvts(path + filename + ".vts", std::ios::out | std::ios::trunc);
  auto extents = grid->get_grid_extents();
  auto times   = grid->get_times();

  int extentX = extents[0].size() - 1;
  int extentY = times.size() - 1;
  int extentZ = 0;

  fvts << std::fixed
       << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
       << std::endl
       << "<StructuredGrid WholeExtent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">" << std::endl
       << "<Piece Extent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">\n<PointData Scalars=\""
       << name << "\"><DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">";

  fvts.precision(16);
  fvts << std::scientific;

  for (auto x : elements)
    fvts << x << " ";

  fvts << "</DataArray></PointData><CellData></CellData>"
       << "<Points><DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">";

  fvts.precision(6);
  fvts << std::fixed;

  for (auto p : grid->get_space_time_points())
    fvts << p(0) << " " << p(1) << " 0 ";

  fvts << "</DataArray></Points></Piece></StructuredGrid></VTKFile>" << std::endl;
  fvts.close();
}

template <int dim>
void MeasuredValues<dim>::write_pvd(std::string path, std::string filename, std::string name) const {
  // currently only implemented for structured grids!
  AssertThrow(grid->get_grid_extents().size(), ExcNotImplemented());
  Assert(grid->size() == elements.size(), ExcInternalError());
  Assert(grid->get_times().size() < 10000, ExcNotImplemented());
  Assert(0 < dim && dim < 4, ExcNotImplemented());

  if (dim == 1) write_vts(path, filename, name);

  auto extents = grid->get_grid_extents();
  auto times   = grid->get_times();

  size_t extentX           = extents[0].size() - 1;
  size_t extentY           = dim > 1 ? extents[1].size() - 1 : 0;
  size_t extentZ           = dim > 2 ? extents[2].size() - 1 : 0;
  size_t nb_spatial_points = grid->get_points()[0].size();

  std::ofstream fpvd(path + filename + ".pvd", std::ios::out | std::ios::trunc);
  fpvd.precision(8);
  fpvd << std::fixed << "<?xml version=\"1.0\"?>" << std::endl
       << "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl
       << "<Collection>";

  for (size_t ti = 0; ti < times.size(); ti++) {
    fpvd << "<DataSet timestep=\"" << times[ti] << "\" group=\"\" part=\"0\" file=\"" << filename << "-"
         << Utilities::to_string(ti, 4) << ".vts\"/>" << std::endl;

    std::ofstream fvts(path + filename + "-" + Utilities::to_string(ti, 4) + ".vts", std::ios::out | std::ios::trunc);

    fvts << std::fixed
         << "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">"
         << std::endl
         << "<StructuredGrid WholeExtent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">" << std::endl
         << "<Piece Extent=\"0 " << extentX << " 0 " << extentY << " 0 " << extentZ << "\">" << std::endl
         << "<PointData Scalars=\"" << name << "\">" << std::endl
         << "<DataArray type=\"Float64\" Name=\"" << name << "\" format=\"ascii\">" << std::endl;

    fvts.precision(16);
    fvts << std::scientific;

    for (size_t i = 0; i < nb_spatial_points; i++)
      fvts << elements[nb_spatial_points * ti + i] << " ";

    fvts << std::endl
         << "</DataArray>" << std::endl
         << "</PointData>" << std::endl
         << "<CellData></CellData>" << std::endl
         << "<Points>" << std::endl
         << "<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">" << std::endl;

    fvts.precision(6);
    fvts << std::fixed;

    for (auto p : grid->get_points()[ti])
      fvts << p(0) << " " << (dim > 1 ? p(1) : 0.0) << " " << (dim > 2 ? p(2) : 0.0) << " ";

    fvts << std::endl
         << "</DataArray>" << std::endl
         << "</Points>" << std::endl
         << "</Piece>" << std::endl
         << "</StructuredGrid>" << std::endl
         << "</VTKFile>" << std::endl;
    fvts.close();
  }

  fpvd << "</Collection>" << std::endl << "</VTKFile>" << std::endl;
  fpvd.close();
}

#ifdef WAVEPI_MPI
template <int dim>
void MeasuredValues<dim>::mpi_irecv(size_t source, std::vector<MPI_Request>& reqs) {
  AssertThrow(reqs.size() == 0, ExcInternalError());

  reqs.emplace_back();
  MPI_Irecv(&elements[0], elements.size(), MPI_DOUBLE, source, 1, MPI_COMM_WORLD, &reqs[0]);
}

template <int dim>
void MeasuredValues<dim>::mpi_send(size_t destination) {
  MPI_Send(&elements[0], elements.size(), MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);
}

template <int dim>
void MeasuredValues<dim>::mpi_all_reduce(MeasuredValues<dim> source, MPI_Op op) {
  MPI_Allreduce(&source.elements[0], &elements[0], elements.size(), MPI_DOUBLE, op, MPI_COMM_WORLD);
}

#endif

template class MeasuredValues<1>;
template class MeasuredValues<2>;
template class MeasuredValues<3>;

} /* namespace measurements */
} /* namespace wavepi */
