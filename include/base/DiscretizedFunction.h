/*
 * DiscretizedFunction.h
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DISCRETIZEDFUNCTION_H_
#define FORWARD_DISCRETIZEDFUNCTION_H_

#include <base/Norm.h>
#include <base/SpaceTimeMesh.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * Essentially, a `std::vector<dealii::Vector<double>>`,
 * representing a function that is pointwise discretized in time and FE-discretized in space.
 * It can be equipped with different norms and scalar products.
 */
template <int dim>
class DiscretizedFunction : public Function<dim> {
 public:
  virtual ~DiscretizedFunction() = default;

  /**
   * @name Constructors and Assignment operators
   * @{
   *
   */

  /**
   * copy constructor
   */
  DiscretizedFunction(const DiscretizedFunction<dim>& that);

  /**
   * move constructor
   */
  DiscretizedFunction(DiscretizedFunction<dim>&& that);

  /**
   * Creates a new discretized function and initializes it with zeroes.
   *
   * @param mesh the mesh you want this function to be attached to
   * @param norm the norm this function should be measured with
   * @param store_derivative true iff this function should also track the derivative of the function.
   */
  DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm,
                      bool store_derivative);

  /**
   * Creates a new discretized function and initializes it with zeroes.
   * The resulting object will _not_ keep track of the coefficients of its derivative.
   *
   * @param mesh the mesh you want this function to be attached to
   * @param norm the norm this function should be measured with
   */
  DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm);

  /**
   * Creates a new discretized function and initializes it with zeroes.
   * The resulting object will _not_ keep track of the coefficients of its derivative.
   *
   * @param mesh the mesh you want this function to be attached to
   */
  explicit DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

  /**
   * Creates a new discretized function from a given (possibly continuous) function.
   * The resulting object will _not_ keep track of the coefficients of its derivative.
   *
   * @param mesh the mesh you want this function to be attached to
   * @param function the function that should be interpolated by this object
   * @param norm the norm this function should be measured with
   */
  DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Function<dim>& function,
                      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm);

  /**
   * Creates a new discretized function from a given (possibly continuous) function.
   * The resulting object will _not_ keep track of the coefficients of its derivative.
   *
   * @param mesh the mesh you want this function to be attached to
   * @param function the function that should be interpolated by this object
   */
  DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Function<dim>& function);

  /**
   * move assignment
   */
  DiscretizedFunction<dim>& operator=(DiscretizedFunction<dim>&& V);

  /**
   * copy assignment
   */
  DiscretizedFunction<dim>& operator=(const DiscretizedFunction<dim>& V);

  /**
   * set all function (and if needed derivative) coefficients to 0
   *
   * @param x has to be zero.
   */
  DiscretizedFunction<dim>& operator=(double x);

  /**
   * @}
   *
   * @name Vector space operations
   */

  /**
   * perform this <- this + constant function
   */
  DiscretizedFunction<dim>& operator+=(double offset);

  /**
   * perform this <- this + V
   */
  DiscretizedFunction<dim>& operator+=(const DiscretizedFunction<dim>& V);

  /**
   * perform this <- this - V
   */
  DiscretizedFunction<dim>& operator-=(const DiscretizedFunction<dim>& V);

  /**
   * perform this <- factor * this
   */
  DiscretizedFunction<dim>& operator*=(const double factor);

  /**
   * perform this <- factor^{-1} * this
   */
  DiscretizedFunction<dim>& operator/=(const double factor);

  /**
   * perform this <- this + a*V
   */
  void add(const double a, const DiscretizedFunction<dim>& V);

  /**
   * perform this <- s*this + a*V
   */
  void sadd(const double s, const double a, const DiscretizedFunction<dim>& V);

  /**
   * @}
   *
   * @name Derivative access
   */

  /**
   * @returns `store_derivative`
   */
  inline bool has_derivative() const { return store_derivative; }

  /**
   * some functions complain if one operand has a derivative and the other doesn't
   */
  void throw_away_derivative();

  /**
   * Creates a DiscretizedFunction with the first time derivative of this one.
   * Works only if `has_derivative` is true.
   */
  DiscretizedFunction<dim> derivative() const;

  /**
   * Approximate the first time derivative using finite differences
   * (one-sided at begin/end and central everywhere else)
   * Throws an error if this function keeps track of its derivative.
   */
  DiscretizedFunction<dim> calculate_derivative() const;

  /**
   * Approximate the first time derivative using finite differences
   * (one-sided at begin/end and central everywhere else)
   * Throws an error if this function keeps track of its derivative.
   */
  DiscretizedFunction<dim> calculate_second_derivative() const;

  /**
   * calculate the transpose (i.e. adjoint using vector norms / dot products) of what `calculate_derivative` does.
   * For constant time step size this is equivalent to g -> -g' in inner nodes (-> partial integration!).
   */
  DiscretizedFunction<dim> calculate_derivative_transpose() const;

  /**
   * calculate the transpose (i.e. adjoint using vector norms / dot products) of what `calculate_second_derivative`
   * does. For constant time step size this is equivalent to g -> g'' in inner nodes (-> partial integration!).
   */
  DiscretizedFunction<dim> calculate_second_derivative_transpose() const;

  /**
   * @}
   *
   * @name Banach- and Hilbert space operations
   */

  /**
   * returns the current norm setting.
   */
  const std::shared_ptr<Norm<DiscretizedFunction<dim>>> get_norm() const;

  /**
   * equip this object with a given norm setting
   */
  void set_norm(std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm);

  /**
   * returns this object's norm, or throws an error if the norm is set to `Invalid`.
   */
  double norm() const;

  /**
   * returns the norm of this object as an element of the dual space.
   */
  double norm_dual() const;

  /**
   * returns the scalar product between this object and another one.
   * Throws an error if the norm is set to `Invalid`, does not define a scalar product
   * or the norms of both functions do not match.
   */
  double dot(const DiscretizedFunction<dim>& V) const;

  /**
   * alias for `dot`.
   */
  double operator*(const DiscretizedFunction<dim>& V) const;

  /**
   * returns whether the used norm also defines a scalar product.
   */
  bool hilbert() const;

  /**
   * relative error (using this object's norm settings).
   * if `this->norm() == 0` this function returns the absolute error.
   */
  double relative_error(const DiscretizedFunction<dim>& other) const;

  /**
   * Calculate the absolute error to a given continuous function.
   * Uses the norm specified by `set_norm`.
   *
   * @param other The other function.
   */
  double absolute_error(Function<dim>& other) const;

  /**
   * Calculate the absolute error to a given continuous function.
   * Uses the norm specified by `set_norm`.
   *
   * @param other The other function.
   * @param norm_out `double` that is filled with the norm of `other`, or `nullptr`.
   */
  double absolute_error(Function<dim>& other, double* norm_out) const;

  /**
   * Calculate the absolute error to a given discretized function.
   * Uses the norm specified by `set_norm`.
   *
   * @param other The other function.
   * @param norm_out `double` that is filled with the norm of `other`, or `nullptr`.
   */
  double absolute_error(DiscretizedFunction<dim>& other, double* norm_out) const;

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
   */
  void dot_transform();

  /**
   * applies the inverse to `dot_transform`, i.e. it applies `M^{-1}`.
   */
  void dot_transform_inverse();

  /**
   * same as `dot_transform`, but applies the inverse mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `L2L2_Trapezoidal_Mass`.
   * In that case only the factors for the trapezoidal rule have to be taken into account.
   */
  void dot_solve_mass_and_transform();

  /**
   * same as `dot_transform_inverse`, but applies the mass matrix to every time step beforehand.
   * This allows for some optimization where M is also built using the mass matrices, e.g. for `L2L2_Trapezoidal_Mass`.
   * In that case only the inverted factors for the trapezoidal rule have to be taken into account.
   */
  void dot_mult_mass_and_transform_inverse();

  /**
   * apply the p-duality mapping to this vector, i.e. it becomes |f|^{p-1} sign(f).
   *
   * @param p space index, has to be larger than 1.
   */
  void duality_mapping_lp(double p);

  /**
   * returns the p-norm the underlying vector.
   *
   * @param p space index, the result is not a norm if p < 1.
   */
  double norm_p(double p);

  /**
   * apply the p-duality mapping to a given element. Default implementation works for hilbert spaces (i.e. J_p(x) =
   * ||x||^{p-2} x).
   */
  void duality_mapping(double p);

  /**
   * apply the q-duality mapping to a given element of the dual space, i.e. the inverse to `duality_mapping` with
   * p = q/(q-1).
   */
  void duality_mapping_dual(double q);

  /**
   * @}
   *
   * @name Access to coefficients
   */

  /**
   * Read access to function coefficients
   *
   * @param idx the time index
   */
  inline const Vector<double>& get_function_coefficients(size_t idx) const {
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return function_coefficients[idx];
  }

  /**
   * Read access to derivative coefficients
   *
   * @param idx the time index
   */
  inline const Vector<double>& get_derivative_coefficients(size_t idx) const {
    Assert(store_derivative, ExcInvalidState());
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return derivative_coefficients[idx];
  }

  /**
   * Read-Write access to function coefficients
   *
   * @param idx the time index
   */
  inline Vector<double>& get_function_coefficients(size_t idx) {
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return function_coefficients[idx];
  }

  /**
   * Read-Write access to derivative coefficients
   *
   * @param idx the time index
   */
  inline Vector<double>& get_derivative_coefficients(size_t idx) {
    Assert(store_derivative, ExcInvalidState());
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return derivative_coefficients[idx];
  }

  /**
   * same as `get_function_coefficients`.
   */
  inline const Vector<double>& operator[](size_t idx) const {
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return function_coefficients[idx];
  }

  /**
   * same as `get_function_coefficients`.
   */
  inline Vector<double>& operator[](size_t idx) {
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));

    return function_coefficients[idx];
  }

  /**
   * Write access to both function- and derivative coefficients. Must only be called if `store_derivative`.
   *
   * @param idx the time index
   * @param u new function coefficients
   * @param v new time derivative coefficients
   */
  inline void set(size_t idx, const Vector<double>& u, const Vector<double>& v) {
    Assert(mesh, ExcNotInitialized());
    Assert(store_derivative, ExcInternalError());
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));
    Assert(function_coefficients[idx].size() == u.size(),
           ExcDimensionMismatch(function_coefficients[idx].size(), u.size()));
    Assert(derivative_coefficients[idx].size() == v.size(),
           ExcDimensionMismatch(derivative_coefficients[idx].size(), v.size()));

    function_coefficients[idx]   = u;
    derivative_coefficients[idx] = v;
  }

  /**
   * Write access to function coefficients. Must only be called if `!store_derivative`.
   *
   * @param idx the time index
   * @param u new function coefficients
   */
  inline void set_function_coefficients(size_t idx, const Vector<double>& u) {
    Assert(mesh, ExcNotInitialized());
    Assert(!store_derivative, ExcInternalError());
    Assert(idx >= 0 && idx < mesh->length(), ExcIndexRange(idx, 0, mesh->length()));
    Assert(function_coefficients[idx].size() == u.size(),
           ExcDimensionMismatch(function_coefficients[idx].size(), u.size()));

    function_coefficients[idx] = u;
  }

  /**
   * returns the number of time steps in this function (short hand for this->get_mesh()->length())
   */
  inline size_t length() const { return function_coefficients.size(); }

  /**
   * returns a pointer to the underlying `SpaceTimeMesh`.
   */
  std::shared_ptr<SpaceTimeMesh<dim>> get_mesh() const;

  /**
   * @}
   *
   * @name Evaluation functions
   */

  /** @copydoc dealii::Function<double>::value*/
  virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  /** @copydoc dealii::Function<double>::gradient(const double new_time) */
  virtual Tensor<1, dim, double> gradient(const Point<dim>& p, const unsigned int component) const;

  /** @copydoc dealii::FunctionTime<double>::set_time(const double new_time) */
  virtual void set_time(const double new_time);

  /**
   * returns the time index for the current return value of `get_time`.
   */
  double get_time_index() const;

  /**
   * @}
   *
   * @name Utilities
   */

  /**
   * returns the value of the smallest coefficent (!) over all time steps
   *
   * note that for P1-elements this is also the smallest function value, whereas for higher order elements, it might not
   * be.
   */
  double min_value() const;

  /**
   * returns the value of the biggest coefficent (!) over all time steps
   *
   * note that for P1-elements this is also the biggest function value, whereas for higher order elements, it might not
   * be.
   */
  double max_value() const;

  /**
   * calculates `min_value` and `max_value` simultaneously
   */
  void min_max_value(double* min_out, double* max_out) const;

  /**
   * this <- fe-interpolation of this * V, i.e. a pointwise multiplication of all nodal values of `this` and `V`.
   */
  void pointwise_multiplication(const DiscretizedFunction<dim>& V);

  /**
   * apply the inverse to the mass matrix for every time step
   *
   * (This has nothing to do with the used scalar product)
   */
  void solve_mass();

  /**
   * apply the mass matrix for every time step
   *
   * (This has nothing to do with the used scalar product)
   */
  void mult_mass();

  /**
   * @}
   *
   * @name Output
   */

  /**
   * pvd output  with derivative
   */
  void write_pvd(std::string path, std::string filename, std::string name, std::string name_deriv) const;

  /**
   * pvd output
   */
  void write_pvd(std::string path, std::string filename, std::string name) const;

  /**
   * vtk output of a specified time step
   */
  void write_vtk(const std::string name, const std::string name_deriv, const std::string filename, size_t i) const;

  /**
   * @}
   */

#ifdef WAVEPI_MPI
  /**
   *
   * @name MPI support
   */

  /**
   * set up irecvs on the data of this object, nonblocking
   */
  void mpi_irecv(size_t source, std::vector<MPI_Request>& reqs);

  /**
   * send the data of this object to another process
   */
  void mpi_send(size_t destination);

  /**
   * send the data of this object to another process, nonblocking
   */
  void mpi_isend(size_t destination, std::vector<MPI_Request>& reqs);

  /**
   * reduce the stuff in source using the given operation, everyone gets the result in this object
   * (should be empty before!)
   */
  void mpi_all_reduce(DiscretizedFunction<dim> source, MPI_Op op);

  /**
   * everyone's version of this object gets overwritten by the stuff `root` has in there.
   */
  void mpi_bcast(size_t root);

  /**
   * @}
   */
#endif

  /**
   * Create a new `DiscretizedFunction` filled with random nodal values between -1 and 1.
   *
   * @param like Template to take mesh size from
   */
  static DiscretizedFunction<dim> noise(const DiscretizedFunction<dim>& like);

  /**
   * Create a new `DiscretizedFunction` filled with random nodal values between -1 and 1.
   *
   * @param mesh the associated mesh
   */
  static DiscretizedFunction<dim> noise(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

  /**
   * Create a new `DiscretizedFunction` filled with random nodal values between -1 and 1, and scale it s.t. the norm of
   * the return value is `norm`.
   *
   * @param like Template to take mesh size and norm setting from
   */
  static DiscretizedFunction<dim> noise(const DiscretizedFunction<dim>& like, double norm);

 private:
  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_;

  bool store_derivative = false;
  size_t cur_time_idx   = 0;

  std::vector<Vector<double>> function_coefficients;
  std::vector<Vector<double>> derivative_coefficients;
};
}  // namespace base
} /* namespace wavepi */

#endif /* DISCRETIZEDFUNCTION_H_ */
