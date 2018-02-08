/*
 * MeasurementProblem.h
 *
 *  Created on: 21.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_MEASUREMENTPROBLEM_H_
#define INCLUDE_PROBLEMS_MEASUREMENTPROBLEM_H_

#include <forward/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>

#include <inversion/InverseProblem.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <memory>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

/**
 * Extend a given Nonlinear Problem with one measurement.
 */
template <typename Param, typename Sol, typename Measurement>
class MeasurementProblem : public NonlinearProblem<Param, Measurement> {
 public:
  virtual ~MeasurementProblem() = default;

  MeasurementProblem(std::shared_ptr<NonlinearProblem<Param, Sol>> base,
                     std::shared_ptr<Measure<Sol, Measurement>> measure)
      : base(base), measure(measure) {
    Assert(base, ExcInternalError());
  }

  virtual std::unique_ptr<LinearProblem<Param, Measurement>> derivative(const Param& p) {
    return std::make_unique<MeasurementProblem<Param, Sol, Measurement>::Linearization>(base->derivative(p), measure);
  }

  virtual Measurement forward(const Param& x) { return measure->evaluate(base->forward(x)); }

 private:
  std::shared_ptr<NonlinearProblem<Param, Sol>> base;
  std::shared_ptr<Measure<Sol, Measurement>> measure;

  class Linearization : public LinearProblem<Param, Measurement> {
   public:
    virtual ~Linearization() = default;

    Linearization(std::shared_ptr<LinearProblem<Param, Sol>> base, std::shared_ptr<Measure<Sol, Measurement>> measure)
        : base(base), measure(measure) {
      Assert(base, ExcInternalError());
    }

    virtual Measurement forward(const Param& h) { return measure->evaluate(base->forward(h)); }

    virtual Param adjoint(const Measurement& g) { return base->adjoint(measure->adjoint(g)); }

    virtual Param zero() { return base->zero(); }

   private:
    std::unique_ptr<LinearProblem<Param, Sol>> base;
    std::shared_ptr<Measure<Sol, Measurement>> measure;
  };
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_MEASUREMENTPROBLEM_H_ */
