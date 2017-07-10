/*
 * DiscretizedFunction.h
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DISCRETIZEDFUNCTION_H_
#define FORWARD_DISCRETIZEDFUNCTION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <utility>
#include <fstream>
#include <iostream>

namespace wavepi {
namespace forward {
using namespace dealii;

template<int dim>
class DiscretizedFunction: public Function<dim> {
   public:
      DiscretizedFunction(bool store_derivative, int capacity);
      DiscretizedFunction(bool store_derivative);
      DiscretizedFunction();
      DiscretizedFunction(const DiscretizedFunction& that);
      DiscretizedFunction(const DiscretizedFunction&& that);
      DiscretizedFunction(Function<dim>& function, const std::vector<double>& times,
            const std::vector<DoFHandler<dim>*>& handlers);
      DiscretizedFunction(Function<dim>& function, const std::vector<double>& times, DoFHandler<dim>* handler);
      DiscretizedFunction(const std::vector<double>& times, DoFHandler<dim>* handler);

      DiscretizedFunction<dim>& operator=(DiscretizedFunction<dim> && V);

      // works only for x = 0
      DiscretizedFunction<dim>& operator=(double x);

      DiscretizedFunction<dim>& operator+=(const DiscretizedFunction<dim>& V);
      DiscretizedFunction<dim>& operator-=(const DiscretizedFunction<dim>& V);

      DiscretizedFunction<dim>& operator*=(const double factor);
      DiscretizedFunction<dim>& operator/=(const double factor);

      void add(const double a, const DiscretizedFunction<dim>& V);

      // scale by s and add a*V
      void sadd(const double s, const double a, const DiscretizedFunction<dim>& V);

      // vector l2 scalar product
      double operator*(const DiscretizedFunction<dim> & V) const;

      void pointwise_multiplication(const DiscretizedFunction<dim>& V);

      // vector l2 norm in time and space
      double l2_norm() const;

      // same as l2_norm
      double norm() const;

      // fill this function with random values
      void rand();

      static DiscretizedFunction<dim> noise(const DiscretizedFunction<dim>& like, double norm);

      // some functions complain if one operand has a derivative and the other doesn't
      // also you might want to conserve memory
      void throw_away_derivative();

      void push_back(DoFHandler<dim>* dof_handler, double time, const Vector<double>& function_coeff);
      void push_back(DoFHandler<dim>* dof_handler, double time, const Vector<double>& function_coeff,
            const Vector<double>& deriv_coeff);
      void push_back(DoFHandler<dim>* dof_handler, double time, Function<dim>& function);

      void write_pvd(std::string path, std::string name, std::string name_deriv) const;
      void write_pvd(std::string path, std::string name) const;

      virtual ~DiscretizedFunction();

      void reverse();

      size_t find_time(double time) const;
      size_t find_nearest_time(double time) const;

      void at(double time, const Vector<double>* &coeffs, const Vector<double>* &deriv_coeffs,
            DoFHandler<dim>* &handler) const;
      void at(double time, const Vector<double>* &coeffs, DoFHandler<dim>* &handler) const;
      void at(double time, const Vector<double>* &coeffs) const;

      double value(const Point<dim> &p, const unsigned int component = 0) const;
      Tensor<1, dim, double> gradient(const Point<dim> &p, const unsigned int component) const;

      double get_time_index() const;
      void set_time(const double new_time);

      const std::vector<Vector<double> >& get_derivative_coefficients() const;
      const std::vector<DoFHandler<dim> *>& get_dof_handlers() const;
      const std::vector<Vector<double> >& get_function_coefficients() const;
      const std::vector<double>& get_times() const;

      bool has_derivative() const;
   private:
      bool store_derivative = false;
      size_t cur_time_idx = 0;

      void write_vtk(const std::string name, const std::string name_deriv, const std::string filename, size_t i) const;

      size_t find_time(double time, size_t low, size_t up, bool increasing) const;
      bool near_enough(double time, size_t idx) const;

      std::vector<double> times;
      std::vector<DoFHandler<dim>*> dof_handlers;
      std::vector<Vector<double>> function_coefficients;
      std::vector<Vector<double>> derivative_coefficients;
};
} /* namespace forward */
} /* namespace wavepi */

#endif /* DISCRETIZEDFUNCTION_H_ */
