/*
 * SensorValues.h
 *
 *  Created on: 30.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_SENSORVALUES_H_
#define INCLUDE_MEASUREMENTS_SENSORVALUES_H_

#include <deal.II/base/exceptions.h>
#include <measurements/SensorDistribution.h>
#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

namespace wavepi {
namespace measurements {
using namespace dealii;

/**
 * `std::vector<dim>` with vector space functionality and l2-norms and scalar products.
 */
template <int dim>
class SensorValues {
 public:
  explicit SensorValues(std::shared_ptr<SensorDistribution<dim>> grid) : grid(grid), elements(grid->size()) {}

  SensorValues(SensorValues&& o) : grid(std::move(o.grid)), elements(std::move(o.elements)) {}

  SensorValues(const SensorValues& o) : grid(o.grid), elements(o.elements) {}

  SensorValues<dim>& operator+=(const SensorValues<dim>& o) {
    add(1.0, o);

    return *this;
  }

  SensorValues<dim>& operator-=(const SensorValues<dim>& o) {
    add(-1.0, o);

    return *this;
  }

  SensorValues<dim>& operator*=(const double factor) {
    for (size_t i = 0; i < this->size(); i++)
      elements[i] *= factor;

    return *this;
  }

  SensorValues<dim>& operator/=(const double factor) { return *this *= (1.0 / factor); }

  double norm() const { return sqrt(this->dot(*this)); }

  double dot(const SensorValues<dim>& o) const { return (*this) * o; }

  double operator*(const SensorValues<dim>& o) const {
    AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

    double tmp = 0.0;

    for (size_t i = 0; i < this->size(); i++)
      tmp += o[i] * elements[i];

    return tmp;
  }

  void add(const double a, const SensorValues<dim>& o) {
    AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

    for (size_t i = 0; i < this->size(); i++)
      elements[i] += a * o[i];
  }

  /**
   * scale by s and add a*V
   */
  void sadd(const double s, const double a, const SensorValues<dim>& o) {
    AssertThrow(o.size() == this->size() && o.grid == this->grid, ExcInternalError());

    for (size_t i = 0; i < this->size(); i++)
      elements[i] = s * elements[i] + a * o[i];
  }

  SensorValues<dim>& operator=(SensorValues<dim>&& o) {
    elements = std::move(o.elements);

    return *this;
  }

  SensorValues<dim>& operator=(const SensorValues<dim>& o) {
    elements = o.elements;

    return *this;
  }

  static SensorValues<dim> noise(std::shared_ptr<SensorDistribution<dim>> grid);

  static SensorValues<dim> noise(const SensorValues<dim>& like);

  static SensorValues<dim> noise(const SensorValues<dim>& like, double norm);

  size_t size() const { return elements.size(); }

  double operator[](size_t i) const {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return elements[i];
  }

  double& operator[](size_t i) {
    Assert(i < size(), ExcIndexRange(i, 0, size()));

    return elements[i];
  }

  void write_pvd(std::string path, std::string filename, std::string name) const;

  double relative_error(const SensorValues<dim>& other) const;

#ifdef WAVEPI_MPI
  /**
   * @name MPI support
   */

  /**
   * set up irecvs on the data of this object
   */
  void mpi_irecv(size_t source, std::vector<MPI_Request>& reqs);

  /**
   * send the data of this object to another process
   */
  void mpi_send(size_t destination);

  /**
   * reduce the stuff in source using the given operation, everyone gets the result in this object
   * (should be empty before!)
   */
  void mpi_all_reduce(SensorValues<dim> source, MPI_Op op);

/**
 * @}
 */
#endif

 private:
  std::shared_ptr<SensorDistribution<dim>> grid;
  std::vector<double> elements;
};  // namespace measurements

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_SENSORVALUES_H_ */
