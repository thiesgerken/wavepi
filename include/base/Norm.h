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

template <typename T>
class Norm {
 public:
  virtual ~Norm() = default;
  Norm()          = default;

  /**
   * returns the norm of `u`.
   */
  virtual double norm(const T& u) const = 0;

  /**
   * returns the scalar product between `u` and `v`.
   * Throws an error if this norm  does not define a scalar product.
   */
  virtual double dot(const T& u, const T& v) const = 0;

  /**
   * applies the matrix M (spd), which describes the used scalar product, i.e.
   * `this->dot(y) = y^t * M * this` (regarding this and y as long vectors)
   * to this function, that is `this <- B * this`.
   *
   * This function is useful for computing the adjoint A^* of a linear operator A from its transpose A^t:
   * `x^t A^t M y = (A x, y) = (x, A^* y) = x^t M A^* y` for all `x` and `y`,
   * hence `A^t M = M A^*`, that is `A^* = M^{-1} A^t M`.
   *
   * For the standard vector norm (`L2L2_Vector`) of the coefficients, `M` is equal to the identity.
   * To get an approximation to the L^2([0,T], L^2(\Omega)) norm (`L2L2_Trapezoidal_Mass`),
   * `M` is a diagonal block matrix consisting of the mass matrix for every time step and a factor to account for the
   * trapezoidal rule.
   *
   * Throws and error if this norm does not define a scalar product.
   */
  virtual void dot_transform(T& u) = 0;

  /**
   * applies the inverse to `dot_transform`, i.e. it applies `M^{-1}`.
   * Throws and error if this norm does not define a scalar product.
   */
  virtual void dot_transform_inverse(T& u) = 0;

  /**
   * same as `dot_transform`, but applies the inverse mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `L2L2_Trapezoidal_Mass`.
   * In that case only the factors for the trapezoidal rule have to be taken into account.
   *
   * Throws and error if this norm does not define a scalar product.
   */
  virtual void dot_solve_mass_and_transform(T& u) = 0;

  /**
   * same as `dot_transform_inverse`, but applies the mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `L2L2_Trapezoidal_Mass`.
   * In that case only the inverted factors for the trapezoidal rule have to be taken into account.
   *
   * Throws and error if this norm does not define a scalar product.
   */
  virtual void dot_mult_mass_and_transform_inverse(T& u) = 0;

  /**
   * does this norm define a scalar product?
   */
  virtual bool hilbert() const = 0;

  /**
   * human readable name of the corresponding space, e.g. `L²([0,T], L²(Ω))`.
   */
  virtual std::string name() const = 0;

  /**
   * id of the norm to conclude whether two norms match, e.g. `H¹([0,T], L²(Ω)) with α=0.3`.
   */
  virtual std::string unique_id() const = 0;

  /**
   * are the two norms the same?
   */
  bool operator==(const Norm<T>& other) const { return other.unique_id() == this->unique_id(); }
};

template <typename T>
class InvalidNorm : public Norm<T> {
 public:
  virtual ~InvalidNorm() = default;
  InvalidNorm()          = default;

  virtual double norm(const T& u __attribute((unused))) const override {
    AssertThrow(false, ExcMessage("call to norm on InvalidNorm"));
    return 0.0;
  }

  virtual double dot(const T& u __attribute((unused)), const T& v __attribute((unused))) const override {
    AssertThrow(false, ExcMessage("call to dot on InvalidNorm"));
    return 0.0;
  }

  virtual void dot_transform(T& u __attribute((unused))) override {
    AssertThrow(false, ExcMessage("call to dot_transform on InvalidNorm"));
  }

  virtual void dot_transform_inverse(T& u __attribute((unused))) override {
    AssertThrow(false, ExcMessage("call to dot_transform_inverse on InvalidNorm"));
  }

  virtual void dot_solve_mass_and_transform(T& u __attribute((unused))) override {
    AssertThrow(false, ExcMessage("call to dot_solve_mass_and_transform on InvalidNorm"));
  }

  virtual void dot_mult_mass_and_transform_inverse(T& u __attribute((unused))) override {
    AssertThrow(false, ExcMessage("call to dot_mult_mass_and_transform_inverse on InvalidNorm"));
  }

  virtual bool hilbert() const override { return false; }

  virtual std::string name() const override { return "Invalid"; }

  virtual std::string unique_id() const override { return "Invalid"; }
};

}  // namespace base
}  // namespace wavepi

#endif /* INCLUDE_BASE_NORM_H_ */
