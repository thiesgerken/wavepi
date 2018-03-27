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
 * Class representing a particular norm or scalar product.
 */
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
   * returns the norm of `u` as an element of the dual space.
   * Default implementation is suitable for hilbert spaces (X = X').
   */
  virtual double norm_dual(const T& u) const { return norm(u); }

  /**
   * returns the scalar product between `u` and `v`.
   * Throws an error if this norm  does not define a scalar product.
   */
  virtual double dot(const T& u, const T& v) const = 0;

  /**
   * applies the matrix \f$M\f$ (spd), which describes the used scalar product, i.e.
   * \f$ (u, v)  = u^t M  v\f$ (regarding \f$u\f$ and \f$v\f$ as long vectors)
   * to \f$u\f$.
   *
   * This function is useful for computing the adjoint \f$A^*\f$ of a linear operator \f$A\f$ from its transpose
   * \f$A^t\f$: \f$ x^t A^t M y = (A x, y) = (x, A^* y) = x^t M A^* y \f$ for all \f$x\f$ and \f$y\f$, hence \f$A^t M =
   * M A^*\f$, that is \f$A^* = M^{-1} A^t M\f$.
   *
   * For the standard vector norm (`norms::L2Coefficients`) of the coefficients, `M` is equal to the identity.
   * To get an approximation to the `L²([0,T], L²(Ω))` norm (`norms::L2L2`),
   * `M` is a diagonal block matrix consisting of the mass matrix for every time step and a factor to account for the
   * trapezoidal rule.
   *
   * This operation can be seen as the canonical mapping from X to X' (not identified with X) via Riesz.
   *
   * Throws an error if this norm does not define a scalar product.
   */
  virtual void dot_transform(T& u) = 0;

  /**
   * applies the inverse to `dot_transform`, i.e. it applies \f$M^{-1}\f$.
   * Throws and error if this norm does not define a scalar product.
   *
   * This operation can be seen as the canonical mapping from X' (not identified with X) to X via Riesz.
   *
   * Throws an error if this norm does not define a scalar product.
   */
  virtual void dot_transform_inverse(T& u) = 0;

  /**
   * same as `dot_transform`, but applies the inverse mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `norms::L2L2`.
   * In that case only the factors for the trapezoidal rule have to be taken into account.
   *
   * Throws an error if this norm does not define a scalar product.
   */
  virtual void dot_solve_mass_and_transform(T& u) = 0;

  /**
   * same as `dot_transform_inverse`, but applies the mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `norms::L2L2`.
   * In that case only the inverted factors for the trapezoidal rule have to be taken into account.
   *
   * Throws an error if this norm does not define a scalar product.
   */
  virtual void dot_mult_mass_and_transform_inverse(T& u) = 0;

  /**
   * does this norm define a scalar product? Default implementation returns `true`.
   */
  virtual bool hilbert() const { return true; }

  /**
   * apply the p-duality mapping to a given element. Default implementation works for hilbert spaces (i.e. \f$J_p(x) =
   * \|x\|^{p-2} x\f$).
   */
  virtual void duality_mapping(T& x, double p) {
    if (p != 2) x *= std::pow(x.norm(), p - 2);
  }

  /**
   * apply the q-duality mapping to a given element of the dual space, i.e. the inverse to `duality_mapping` with
   * \f$p = \frac q {q-1}\f$. Default implementation works for hilbert spaces (i.e. \f$J_p(x) = \|x\|^{q-2} x\f$).
   */
  virtual void duality_mapping_dual(T& x, double q) { duality_mapping(x, q); }

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

/**
 * Dummy norm which should be the default for `T`, so that no wrong assumptions are made.
 */
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
