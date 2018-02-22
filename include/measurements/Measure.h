/*
 * Measurement.h
 *
 *  Created on: 17.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_MEASURE_H_
#define INCLUDE_MEASUREMENTS_MEASURE_H_

namespace wavepi {

/**
 * Measurement operators (and adjoints) as well as the needed structure to have measurements.
 */
namespace measurements {

/**
 * Interface for linear Measurements. `Measurement` will most likely be `std::vector<double>`.
 * The type `Sol` is the type of the continuous things this class measures.
 */
template <typename Sol, typename Measurement>
class Measure {
 public:
  virtual ~Measure() = default;

  /**
   * Compute a measurement.
   */
  virtual Measurement evaluate(const Sol& field) = 0;

  /**
   * Compute the adjoint of what `evaluate` does.
   */
  virtual Sol adjoint(const Measurement& measurements) = 0;

  /**
   * allocate memory for a measurement (e.g. to be received by MPI)
   */
  virtual Measurement zero() = 0;
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_MEASURE_H_ */
