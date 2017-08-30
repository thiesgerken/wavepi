/*
 * MeasuredValues.cpp
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#include <measurements/MeasuredValues.h>
#include <deal.II/base/exceptions.h>
#include <stddef.h>
#include <memory>
#include <random>
#include <string>
#include <vector>
#include <util/SpaceTimeGrid.h>
#include <stddef.h>
#include <tgmath.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {

template<int dim>
MeasuredValues<dim> MeasuredValues<dim>::noise(const MeasuredValues<dim>& like) {
   MeasuredValues<dim> res(like.grid);

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(-1, 1);

   for (size_t i = 0; i < like.size(); i++)
      res[i] = distribution(generator);

   return res;
}

template<int dim>
MeasuredValues<dim> MeasuredValues<dim>::noise(const MeasuredValues<dim>& like, double norm) {
   auto res = noise(like);
   res *= norm / res.norm();

   return res;
}

template<>
void MeasuredValues<1>::write_pvd(std::string path, std::string filename, std::string name) const {
   // TODO
   AssertThrow(false, ExcNotImplemented());

   // TODO: in 1D also output as pvd? this as extra?
   // std::ofstream fvts (path + filename + ".vtu", std::ios::out | std::ios::trunc);

   //    int pointsSpace = pow_d(m.opt.opt.nbSpace);
   //    int extentX, extentY;
   //    extentX = m.opt.opt.nbSpace-1; extentY = m.opt.opt.nbTime-1;
   //
   //    fprintf(fvts, "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n<StructuredGrid WholeExtent=\"0 %i 0 %i 0 %i\">\n<Piece Extent=\"0 %i 0 %i 0 %i\">\n<PointData Scalars=\"u_h\">\n<DataArray type=\"Float64\" Name=\"u_h\" format=\"ascii\">\n", extentX, extentY, 0, extentX, extentY, 0);
   //
   //    for (int j=0; j < m.opt.opt.nbTime; j++) {
   //      for (int i=0; i<pointsSpace; i++) {
   //        int idx = j*pointsSpace+i;
   //        fprintf(fvts, "%4.8e ", m.data[midx].sensors[idx]);
   //      }
   //    }
   //
   //    fprintf(fvts, "\n</DataArray>\n</PointData>\n<CellData>\n</CellData>\n<Points>\n<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
   //
   //    for (int j=0; j < m.opt.opt.nbTime; j++) {
   //      REAL t = calculateTime(j);
   //
   //      for (int i=0; i<pointsSpace; i++) {
   //        REAL_D coord; calculateCoordinates(i, m.opt.opt.nbSpace, coord);
   //        fprintf(fvts, "%4.3f %4.3f 0 ", coord[0], t);
   //      }
   //    }
   //
   //    fprintf(fvts, "\n</DataArray>\n</Points>\n</Piece>\n</StructuredGrid>\n</VTKFile>\n");
   //    fclose(fvts);
   //
}

template<int dim>
void MeasuredValues<dim>::write_pvd(std::string path, std::string filename, std::string name) const {
   // TODO
   AssertThrow(false, ExcNotImplemented());

//    sprintf(path, "%s/%s/data/%s_grid.pvd", outputPath, subdir, title);
//    FILE *fpvd = fopen(path, "w");
//
//    fprintf(fpvd, "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n<Collection>\n");
//
//    for (int j=0; j < m.opt.opt.nbTime; j++) {
//      REAL t = calculateTime(j);
//      fprintf(fpvd, "<DataSet timestep=\"%4.6f\" group=\"\" part=\"0\" file=\"%s%06i_grid.vts\"/>\n", t, title, j);
//
//      int pointsSpace = pow_d(m.opt.opt.nbSpace);
//      sprintf(path, "%s/%s/data/%s%06i_grid.vts", outputPath, subdir, title, j);
//      FILE *fvts = fopen(path, "w");
//
//      int extentX, extentY, extentZ;
//
//      #if DIM_OF_WORLD == 1
//      extentX = m.opt.opt.nbSpace-1; extentY = extentZ = 0;
//      #elif DIM_OF_WORLD == 2
//      extentX = extentY = m.opt.opt.nbSpace-1; extentZ = 0;
//      #elif DIM_OF_WORLD == 3
//      extentX = extentY = extentZ = m.opt.opt.nbSpace-1;
//      #endif
//
//      fprintf(fvts, "<VTKFile type=\"StructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n<StructuredGrid WholeExtent=\"0 %i 0 %i 0 %i\">\n<Piece Extent=\"0 %i 0 %i 0 %i\">\n<PointData Scalars=\"u_h\">\n<DataArray type=\"Float64\" Name=\"u_h\" format=\"ascii\">\n", extentX, extentY, extentZ, extentX, extentY, extentZ);
//
//      for (int i=0; i<pointsSpace; i++) {
//        int idx = j*pointsSpace+i;
//        fprintf(fvts, "%4.8e ", m.data[midx].sensors[idx]);
//      }
//
//      fprintf(fvts, "\n</DataArray>\n</PointData>\n<CellData>\n</CellData>\n<Points>\n<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\">\n");
//
//      for (int i=0; i<pointsSpace; i++) {
//        REAL_D coord; calculateCoordinates(i, m.opt.opt.nbSpace, coord);
//
//        #if DIM_OF_WORLD == 1
//        fprintf(fvts, "%4.3f 0 0 ", coord[0]);
//        #elif DIM_OF_WORLD == 2
//        fprintf(fvts, "%4.3f %4.3f 0 ", coord[0], coord[1]);
//        #elif DIM_OF_WORLD == 3
//        fprintf(fvts, "%4.3f %4.3f %4.3f ", coord[0], coord[1], coord[2]);
//        #endif
//      }
//
//      fprintf(fvts, "\n</DataArray>\n</Points>\n</Piece>\n</StructuredGrid>\n</VTKFile>\n");
//      fclose(fvts);
//    }
//
//    fprintf(fpvd, "</Collection>\n</VTKFile>\n");
//    fclose(fpvd);
}


template class MeasuredValues<1> ;
template class MeasuredValues<2> ;
template class MeasuredValues<3> ;

} /* namespace measurements */
} /* namespace wavepi */
