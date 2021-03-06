/*
 * NewtonRegularization.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_NEWTONREGULARIZATION_H_
#define INVERSION_NEWTONREGULARIZATION_H_

#include <inversion/NonlinearProblem.h>
#include <inversion/Regularization.h>

#include <memory>

namespace wavepi {
namespace inversion {

// Param and Sol need at least banach space structure
template <typename Param, typename Sol, typename Exact>
class NewtonRegularization : public Regularization<Param, Sol, Exact> {
 public:
  virtual ~NewtonRegularization() = default;

  NewtonRegularization(std::shared_ptr<NonlinearProblem<Param, Sol>> problem) : problem(problem) {}

  NewtonRegularization() : problem() {}

  const std::shared_ptr<NonlinearProblem<Param, Sol>>& get_problem() const { return problem; }

  void set_problem(const std::shared_ptr<NonlinearProblem<Param, Sol>>& problem) { this->problem = problem; }

 protected:
  std::shared_ptr<NonlinearProblem<Param, Sol>> problem;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_NEWTONREGULARIZATION_H_ */
