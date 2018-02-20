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
};

/**
 * No real measurement, just returns the argument.
 * This way one can still reconstruct from the whole field even when the rest of the code expects measurement operators.
 */
template <typename Sol>
class IdenticalMeasure : public Measure<Sol, Sol> {
 public:
  virtual ~IdenticalMeasure() = default;

  virtual Sol evaluate(const Sol& field) { return field; }

  virtual Sol adjoint(const Sol& measurements) { return measurements; }
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_MEASURE_H_ */
