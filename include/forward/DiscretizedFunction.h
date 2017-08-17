/*
 * DiscretizedFunction.h
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DISCRETIZEDFUNCTION_H_
#define FORWARD_DISCRETIZEDFUNCTION_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <stddef.h>
#include <memory>
#include <string>
#include <vector>

#include <forward/SpaceTimeMesh.h>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class DiscretizedFunction: public Function<dim> {
   public:
      enum Norm {
         L2L2_Vector, L2L2_Trapezoidal_Mass
      };

      
      virtual ~DiscretizedFunction() = default;

      DiscretizedFunction(const DiscretizedFunction& that);
      DiscretizedFunction(DiscretizedFunction&& that);

      DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, bool store_derivative);
      DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Function<dim>& function);
      DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

      void set(size_t i, const Vector<double>& u, const Vector<double>& v);
      void set(size_t i, const Vector<double>& u);

      DiscretizedFunction<dim>& operator=(DiscretizedFunction<dim> && V);
      DiscretizedFunction<dim>& operator=(const DiscretizedFunction<dim> & V);

      // works only for x = 0
      DiscretizedFunction<dim>& operator=(double x);

      DiscretizedFunction<dim>& operator+=(const DiscretizedFunction<dim>& V);
      DiscretizedFunction<dim>& operator-=(const DiscretizedFunction<dim>& V);

      DiscretizedFunction<dim>& operator*=(const double factor);
      DiscretizedFunction<dim>& operator/=(const double factor);

      inline bool has_derivative() const {
         return store_derivative;
      }

      // some functions complain if one operand has a derivative and the other doesn't
      // also you might want to conserve memory
      void throw_away_derivative();

      // returns a DiscretizedFunction with the first time derivative of this one
      DiscretizedFunction<dim> derivative() const;

      // calculate the first time derivative using finite differences (one-sided at begin/end and central everywhere else)
      DiscretizedFunction<dim> calculate_derivative() const;

      // calculate the adjoint (w.r.t. vector norms / dot products!) of what calculate_derivative does
      // for constant time step size this is equivalent to g -> -g' in inner nodes (-> partial integration)
      DiscretizedFunction<dim> calculate_derivative_transpose() const;

      void add(const double a, const DiscretizedFunction<dim>& V);

      // scale by s and add a*V
      void sadd(const double s, const double a, const DiscretizedFunction<dim>& V);

      void pointwise_multiplication(const DiscretizedFunction<dim>& V);

      // depending on the norm setting
      double norm() const;

      // depending on the norm setting
      double operator*(const DiscretizedFunction<dim> & V) const;
      double dot(const DiscretizedFunction<dim> & V) const;

      // depending on the norm setting
      void mult_space_time_mass();
      void solve_space_time_mass();
      void mult_time_mass();
      void solve_time_mass();
      bool norm_uses_mass_matrix() const;

      // fill this function with random values
      void rand();

      static DiscretizedFunction<dim> noise(const DiscretizedFunction<dim>& like, double norm);

      void write_pvd(std::string path, std::string filename, std::string name, std::string name_deriv) const;
      void write_pvd(std::string path, std::string filename, std::string name) const;

      double value(const Point<dim> &p, const unsigned int component = 0) const;
      Tensor<1, dim, double> gradient(const Point<dim> &p, const unsigned int component) const;

      double get_time_index() const;
      void set_time(const double new_time);

      inline const Vector<double>& get_derivative_coefficient(size_t idx) const {
         Assert(store_derivative, ExcInvalidState());

         return derivative_coefficients[idx];
      }

      inline const Vector<double>& get_function_coefficient(size_t idx) const {
         return function_coefficients[idx];
      }

      // get / set what `norm()` and `*` do.
      Norm get_norm() const;
      void set_norm(Norm norm);

      // relative error (using this object's norm settings)
      // if this->norm() == 0 it returns the absolute error.
      double relative_error(const DiscretizedFunction<dim>& other) const;

      std::shared_ptr<SpaceTimeMesh<dim> > get_mesh() const;

   private:
      Norm norm_type = L2L2_Trapezoidal_Mass; // L2L2_Vector
      bool store_derivative = false;
      size_t cur_time_idx = 0;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;

      std::vector<Vector<double>> function_coefficients;
      std::vector<Vector<double>> derivative_coefficients;

      void write_vtk(const std::string name, const std::string name_deriv, const std::string filename,
            size_t i) const;

      // vector l2 norm and dot product in time and space
      // fast, but only a crude approximation (even in case of uniform space-time grids and P1-elements)
      double l2l2_vec_norm() const;
      double l2l2_vec_dot(const DiscretizedFunction<dim> & V) const;

      // l2 norm and dot product in time and space
      // through trapezoidal rule in time, mass matrix in space
      double l2l2_mass_norm() const;
      double l2l2_mass_dot(const DiscretizedFunction<dim> & V) const;

      void l2l2_mass_mult_space_time_mass();
      void l2l2_mass_solve_space_time_mass();

      void l2l2_mass_mult_time_mass();
      void l2l2_mass_solve_time_mass();

};
} /* namespace forward */
} /* namespace wavepi */

#endif /* DISCRETIZEDFUNCTION_H_ */
