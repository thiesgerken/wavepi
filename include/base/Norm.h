/*
 * Norm.h
 *
 *  Created on: 19.01.2018
 *      Author: thies
 */

#ifndef INCLUDE_BASE_NORM_H_
#define INCLUDE_BASE_NORM_H_

#include <deal.II/base/exceptions.h>
#include <string>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * possible norm settings
 */
enum class Norm {
  /**
   * Invalid norm setting. This is the default setting for newly constructed objects.
   */
  Invalid = 0,

  /**
   * 2-norm on the underlying vectors.
   * Fast, but only a crude approximation (even in case of uniform space-time grids and P1-elements)
   */
  Coefficients,

  /**
   * L^2([0,T], L^2(\Omega)) norm, using the trapezoidal rule in time (approximation)
   * and the mass matrix in space (exact)
   */
  L2L2,

  /**
   * H^1([0,T], L^2(\Omega)) norm, using the trapezoidal rule in time (approximation),
   * the mass matrix in space (exact) and finite differences of order h^2 (inner) and h (boundary)
   */
  H1L2,

  /**
   * H^2([0,T], L^2(\Omega)) norm, using the trapezoidal rule in time (approximation),
   * the mass matrix in space (exact) and finite differences of order h^2 (inner) and h (boundary)
   */
  H2L2,

  /**
   * H^1([0,T], H^1(\Omega)) norm, using the trapezoidal rule in time (approximation),
   * the mass+laplace matrix in space (exact) and finite differences of order h^2 (inner) and h (boundary)
   */
  H1H1
};

std::string to_string(const Norm &norm);

}  // namespace base
}  // namespace wavepi

#endif /* INCLUDE_BASE_NORM_H_ */
