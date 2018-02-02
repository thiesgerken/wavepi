/*
 * MeasuredValues.h
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_
#define INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_

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
#include <deal.II/base/mpi.h>

namespace wavepi {
namespace measurements {
using namespace dealii;
using namespace wavepi::util;

/**
 * `std::vector<dim>` with vector space functionality and l2-norms and scalar products.
 */
template<int dim>
class MeasuredValues {
   public:
      explicit MeasuredValues(std::shared_ptr<SpaceTimeGrid<dim>> grid)
            : grid(grid), elements(grid->size()) {
      }

      MeasuredValues(MeasuredValues&& o)
            : grid(std::move(o.grid)), elements(std::move(o.elements)) {
      }

      MeasuredValues(const MeasuredValues& o)
            : grid(o.grid), elements(o.elements) {
      }

      MeasuredValues<dim>& operator+=(const MeasuredValues<dim>& o) {
         add(1.0, o);

         return *this;
      }

      MeasuredValues<dim>& operator-=(const MeasuredValues<dim>& o) {
         add(-1.0, o);

         return *this;
      }

      MeasuredValues<dim>& operator*=(const double factor) {
         for (size_t i = 0; i < this->size(); i++)
            elements[i] *= factor;

         return *this;
      }

      MeasuredValues<dim>& operator/=(const double factor) {
         return *this *= (1.0 / factor);
      }

      double norm() const {
         return sqrt(this->dot(*this));
      }

      double dot(const MeasuredValues<dim>& o) const {
         return (*this) * o;
      }

      double operator*(const MeasuredValues<dim>& o) const {
         AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

         double tmp = 0.0;

         for (size_t i = 0; i < this->size(); i++)
            tmp += o[i] * elements[i];

         return tmp;
      }

      void add(const double a, const MeasuredValues<dim>& o) {
         AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i] += a * o[i];
      }

      /**
       * scale by s and add a*V
       */
      void sadd(const double s, const double a, const MeasuredValues<dim> &o) {
         AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

         for (size_t i = 0; i < this->size(); i++)
            elements[i] = s * elements[i] + a * o[i];
      }

      MeasuredValues<dim>& operator=(MeasuredValues<dim> && o) {
         elements = std::move(o.elements);

         return *this;
      }

      MeasuredValues<dim>& operator=(const MeasuredValues<dim> & o) {
         elements = o.elements;

         return *this;
      }

      static MeasuredValues<dim> noise(const MeasuredValues<dim>& like);

      static MeasuredValues<dim> noise(const MeasuredValues<dim>& like, double norm);

      size_t size() const {
         return elements.size();
      }

      double operator[](size_t i) const {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      double& operator[](size_t i) {
         Assert(i < size(), ExcIndexRange(i, 0, size()));

         return elements[i];
      }

      void write_pvd(std::string path, std::string filename, std::string name) const;

      /**
       * Only in 1D: write space + time data as 2D vts file
       */
      void write_vts(std::string path, std::string filename, std::string name) const;

      /**
         * @name MPI support
         */

      /**
       * make windows to the storage of this object
       */
       std::vector<MPI_Win> make_windows();

       /**
        * copy the data of this object to another process
        */
       void copy_to(std::vector<MPI_Win> destination, size_t rank);

      /**
       * @}
       */

   private:
      std::shared_ptr<SpaceTimeGrid<dim>> grid;
      std::vector<double> elements;
};

} /* namespace measurements */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_MEASUREDVALUES_H_ */
