/*
 * InverseProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_INVERSEPROBLEM_H_
#define INVERSION_INVERSEPROBLEM_H_

#include <memory>

namespace wavepi {

/**
 * Algorithms to solve the inverse problem and possibly linear subproblems
 */
namespace inversion {

template <typename Param, typename Sol>
class InverseProblem {
 public:
  virtual ~InverseProblem() = default;

  virtual Sol forward(const Param& f) = 0;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSEPROBLEM_H_ */
