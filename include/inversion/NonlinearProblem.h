/*
 * NonlinearProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NONLINEARPROBLEM_H_
#define INVERSION_NONLINEARPROBLEM_H_

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>

#include <memory>

namespace wavepi {
namespace inversion {

struct NonlinearProblemStats {
 public:
  int calls_forward;
  int calls_linearization_forward;
  int calls_linearization_adjoint;

  double time_forward;
  double time_linearization_forward;
  double time_linearization_adjoint;

  int calls_measure_forward;
  int calls_measure_adjoint;

  double time_measure_forward;
  double time_measure_adjoint;

  double time_communication;
};

template <typename Param, typename Sol>
class NonlinearProblem : public InverseProblem<Param, Sol> {
 public:
  virtual ~NonlinearProblem() = default;

  /**
   * returns the derivative (as linear operator) at p.
   */
  virtual std::unique_ptr<LinearProblem<Param, Sol>> derivative(const Param& p) = 0;

  /**
   * If supported, return statistics for the calls made to this class.
   * Otherwise, return `nullptr`, like the default implementation does.
   */
  virtual std::shared_ptr<NonlinearProblemStats> get_statistics() { return nullptr; }
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NONLINEARPROBLEM_H_ */
