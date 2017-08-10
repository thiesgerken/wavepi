/*
 * DiscretizedFunction.cpp
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#include <deal.II/base/timer.h>
#include <deal.II/base/data_out_base.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

#include <forward/DiscretizedFunction.h>

using namespace dealii;

namespace wavepi {
namespace forward {

template class DiscretizedFunction<1> ;
template class DiscretizedFunction<2> ;
template class DiscretizedFunction<3> ;

template<int dim>
DiscretizedFunction<dim>::~DiscretizedFunction() {
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, bool store_derivative)
      : store_derivative(store_derivative), cur_time_idx(0), mesh(mesh) {
   Assert(mesh, ExcNotInitialized());

   function_coefficients.reserve(mesh->get_times().size());

   if (store_derivative)
      derivative_coefficients.reserve(mesh->get_times().size());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      function_coefficients.emplace_back(mesh->get_n_dofs(i));

      if (store_derivative)
         derivative_coefficients.emplace_back(mesh->get_n_dofs(i));
   }

}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
      : DiscretizedFunction(mesh, false) {
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      Function<dim>& function)
      : store_derivative(false), cur_time_idx(0), mesh(mesh) {
   Assert(mesh, ExcNotInitialized());

   function_coefficients.reserve(mesh->get_times().size());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      auto dof_handler = mesh->get_dof_handler(i);

      Vector<double> tmp(dof_handler->n_dofs());
      function.set_time(mesh->get_time(i));

      VectorTools::interpolate(*dof_handler, function, tmp);
      function_coefficients.push_back(std::move(tmp));
   }
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(DiscretizedFunction&& o)
      : Function<dim>(), norm_type(o.norm_type), store_derivative(o.store_derivative), cur_time_idx(
            o.cur_time_idx), mesh(std::move(o.mesh)), function_coefficients(
            std::move(o.function_coefficients)), derivative_coefficients(std::move(o.derivative_coefficients)) {
   Assert(mesh, ExcNotInitialized());

   o.mesh = std::shared_ptr<SpaceTimeMesh<dim>>();
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(const DiscretizedFunction& o)
      : Function<dim>(), norm_type(o.norm_type), store_derivative(o.store_derivative), cur_time_idx(
            o.cur_time_idx), mesh(o.mesh), function_coefficients(o.function_coefficients), derivative_coefficients(
            o.derivative_coefficients) {
   Assert(mesh, ExcNotInitialized());
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(DiscretizedFunction<dim> && o) {
   norm_type = o.norm_type;
   store_derivative = o.store_derivative;
   cur_time_idx = o.cur_time_idx;
   mesh = std::move(o.mesh);
   function_coefficients = std::move(o.function_coefficients);
   derivative_coefficients = std::move(o.derivative_coefficients);

   o.mesh = std::shared_ptr<SpaceTimeMesh<dim>>();

   Assert(mesh, ExcNotInitialized());
   return *this;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(const DiscretizedFunction<dim> & o) {
   norm_type = o.norm_type;
   store_derivative = o.store_derivative;
   cur_time_idx = o.cur_time_idx;
   mesh = o.mesh;
   function_coefficients = o.function_coefficients;
   derivative_coefficients = o.derivative_coefficients;

   Assert(mesh, ExcNotInitialized());
   return *this;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::derivative() const {
   Assert(store_derivative, ExcInternalError());
   Assert(mesh, ExcNotInitialized());

   DiscretizedFunction<dim> result(mesh, false);
   result.function_coefficients = this->derivative_coefficients;

   return result;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative() const {
   Assert(mesh, ExcNotInitialized());
   Assert(mesh->get_times().size() > 1, ExcInternalError());
   Assert(!store_derivative, ExcInternalError()); // why would you want to calculate it in this case?

   DiscretizedFunction<dim> result(mesh, false);

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      if (i == 0) {
         // TODO: get DoFHandlers and so on to do this, maybe even go back to the theory and check what is appropriate here
         AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(),
               ExcNotImplemented());

         result.function_coefficients[i] = function_coefficients[i + 1];
         result.function_coefficients[i] -= function_coefficients[i];
         result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i);
      } else if (i == mesh->get_times().size() - 1) {
         result.function_coefficients[i] = function_coefficients[i];
         result.function_coefficients[i] -= function_coefficients[i - 1];
         result.function_coefficients[i] /= mesh->get_time(i) - mesh->get_time(i - 1);
      } else {
         result.function_coefficients[i] = function_coefficients[i + 1];
         result.function_coefficients[i] -= function_coefficients[i - 1];
         result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i - 1);
      }
   }

   return result;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative_transpose() const {
   Assert(mesh, ExcNotInitialized());

   // why would you want to calculate it in this case?
   Assert(!store_derivative, ExcInternalError());

   // because of the special cases
   Assert(mesh->get_times().size() > 3, ExcInternalError());

   DiscretizedFunction<dim> result(mesh, false);

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      auto dest = &result.function_coefficients[i];

      if (i == 0) {
         // TODO: get DoFHandlers and so on to do this, maybe even go back to the theory and check what is appropriate here
         AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(),
               ExcNotImplemented());

         *dest = function_coefficients[i + 1];
         dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
               -1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
      } else if (i == 1) {
         *dest = function_coefficients[i + 1];
         dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
               1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i - 1]);
      } else if (i == mesh->get_times().size() - 1) {
         *dest = function_coefficients[i];
         dest->sadd(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)),
               1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      } else if (i == mesh->get_times().size() - 2) {
         *dest = function_coefficients[i + 1];
         dest->sadd(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)),
               1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      } else {
         *dest = function_coefficients[i + 1];
         dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
               1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      }
   }

   return result;
}

template<int dim>
void DiscretizedFunction<dim>::set(size_t i, const Vector<double>& u, const Vector<double>& v) {
   Assert(mesh, ExcNotInitialized());
   Assert(store_derivative, ExcInternalError());
   Assert(i >= 0 && i < mesh->get_times().size(), ExcIndexRange(i, 0, mesh->get_times().size()));
   Assert(function_coefficients[i].size() == u.size(),
         ExcDimensionMismatch(function_coefficients[i].size(), u.size()));
   Assert(derivative_coefficients[i].size() == v.size(),
         ExcDimensionMismatch(derivative_coefficients[i].size(), v.size()));

   function_coefficients[i] = u;
   derivative_coefficients[i] = v;
}

template<int dim>
void DiscretizedFunction<dim>::set(size_t i, const Vector<double>& u) {
   Assert(mesh, ExcNotInitialized());
   Assert(!store_derivative, ExcInternalError());
   Assert(i >= 0 && i < mesh->get_times().size(), ExcIndexRange(i, 0, mesh->get_times().size()));
   Assert(function_coefficients[i].size() == u.size(),
         ExcDimensionMismatch(function_coefficients[i].size(), u.size()));

   function_coefficients[i] = u;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(double x) {
   Assert(mesh, ExcNotInitialized());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      function_coefficients[i] = x;

      if (store_derivative)
         derivative_coefficients[i] = 0.0;
   }

   return *this;
}
template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator+=(const DiscretizedFunction<dim> & V) {
   this->add(1.0, V);

   return *this;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator-=(const DiscretizedFunction<dim> & V) {
   this->add(-1.0, V);

   return *this;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator*=(const double factor) {
   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      function_coefficients[i] *= factor;

      if (store_derivative)
         derivative_coefficients[i] *= factor;
   }

   return *this;
}

template<int dim>
void DiscretizedFunction<dim>::rand() {
   Assert(!store_derivative, ExcInternalError ());
   Assert(mesh, ExcNotInitialized());

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0, 1);

   for (size_t i = 0; i < mesh->get_times().size(); i++)
      for (size_t j = 0; j < function_coefficients[i].size(); j++)
         function_coefficients[i][j] = distribution(generator);
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(const DiscretizedFunction<dim>& like, double norm) {
   DiscretizedFunction<dim> result(like);

   result.throw_away_derivative(); // to be on the safe side
   result.rand();
   result *= norm / result.norm();

   return result;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator/=(const double factor) {
   return this->operator*=(1.0 / factor);
}

template<int dim>
void DiscretizedFunction<dim>::pointwise_multiplication(const DiscretizedFunction<dim> & V) {
   Assert(mesh, ExcNotInitialized());
   Assert(mesh == V.mesh, ExcInternalError ());
   Assert(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      if (store_derivative) {
         Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
               ExcDimensionMismatch (derivative_coefficients[i].size() , V.derivative_coefficients[i].size()));

         derivative_coefficients[i].scale(V.function_coefficients[i]);

         Vector<double> tmp = function_coefficients[i];
         tmp.scale(V.derivative_coefficients[i]);

         derivative_coefficients[i] += tmp;
      }

      function_coefficients[i].scale(V.function_coefficients[i]);
   }
}

template<int dim>
void DiscretizedFunction<dim>::add(const double a, const DiscretizedFunction<dim> & V) {
   Assert(mesh, ExcNotInitialized());
   Assert(mesh == V.mesh, ExcInternalError ());
   Assert(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      function_coefficients[i].add(a, V.function_coefficients[i]);

      if (store_derivative) {
         Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
               ExcDimensionMismatch (derivative_coefficients[i].size() , V.derivative_coefficients[i].size()));

         derivative_coefficients[i].add(a, V.derivative_coefficients[i]);
      }
   }
}

template<int dim>
void DiscretizedFunction<dim>::sadd(const double s, const double a, const DiscretizedFunction<dim> & V) {
   Assert(mesh, ExcNotInitialized());
   Assert(mesh == V.mesh, ExcInternalError ());
   Assert(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      function_coefficients[i].sadd(s, a, V.function_coefficients[i]);

      if (store_derivative) {
         Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
               ExcDimensionMismatch (derivative_coefficients[i].size() , V.derivative_coefficients[i].size()));

         derivative_coefficients[i].sadd(s, a, V.derivative_coefficients[i]);
      }
   }
}

template<int dim>
void DiscretizedFunction<dim>::throw_away_derivative() {
   store_derivative = false;
   derivative_coefficients = std::vector<Vector<double>>();
}

template<int dim>
double DiscretizedFunction<dim>::norm() const {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return l2l2_vec_norm();
      case L2L2_Trapezoidal_Mass:
         return l2l2_mass_norm();
   }

   Assert(false, ExcInternalError ());
   return 0.0;
}

template<int dim>
double DiscretizedFunction<dim>::operator*(const DiscretizedFunction<dim> & V) const {
   Assert(mesh, ExcNotInitialized());
   Assert(mesh == V.mesh, ExcInternalError ());
   Assert(norm_type == V.norm_type, ExcInternalError());

   switch (norm_type) {
      case L2L2_Vector:
         return l2l2_vec_dot(V);
      case L2L2_Trapezoidal_Mass:
         return l2l2_mass_dot(V);
   }

   Assert(false, ExcInternalError ());
   return 0.0;
}

template<int dim>
double DiscretizedFunction<dim>::dot(const DiscretizedFunction<dim> & V) const {
   return (*this) * V;
}

template<int dim>
void DiscretizedFunction<dim>::mult_space_time_mass() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return;
      case L2L2_Trapezoidal_Mass:
         l2l2_mass_mult_space_time_mass();
         return;
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::solve_space_time_mass() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return;
      case L2L2_Trapezoidal_Mass:
         l2l2_mass_solve_space_time_mass();
         return;
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::mult_time_mass() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return;
      case L2L2_Trapezoidal_Mass:
         l2l2_mass_mult_time_mass();
         return;
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::solve_time_mass() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return;
      case L2L2_Trapezoidal_Mass:
         l2l2_mass_solve_time_mass();
         return;
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
bool DiscretizedFunction<dim>::norm_uses_mass_matrix() const {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case L2L2_Vector:
         return false;
      case L2L2_Trapezoidal_Mass:
         return true;
   }

   Assert(false, ExcInternalError ());
   return false;
}

template<int dim>
void DiscretizedFunction<dim>::l2l2_mass_mult_space_time_mass() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Vector<double> tmp(function_coefficients[i].size());
      mesh->get_mass_matrix(i)->vmult(tmp, function_coefficients[i]);
      function_coefficients[i] = tmp;
   }

   l2l2_mass_mult_time_mass();
}

template<int dim>
void DiscretizedFunction<dim>::l2l2_mass_mult_time_mass() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      double factor = 0.0;

      if (i > 0)
         factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;

      if (i < mesh->get_times().size() - 1)
         factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

      function_coefficients[i] *= factor;
   }
}

template<int dim>
void DiscretizedFunction<dim>::l2l2_mass_solve_space_time_mass() {
   Assert(!store_derivative, ExcInternalError ());

   LogStream::Prefix p("solve_space_time_mass");
   Timer timer;
   timer.start();

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      LogStream::Prefix p("step-" + Utilities::int_to_string(i, 4));

      Vector<double> tmp(function_coefficients[i].size());

      SolverControl solver_control(2000, 1e-10 * function_coefficients[i].l2_norm());
      SolverCG<> cg(solver_control);
      PreconditionIdentity precondition = PreconditionIdentity();

      cg.solve(*mesh->get_mass_matrix(i), tmp, function_coefficients[i], precondition);
      function_coefficients[i] = tmp;
   }

   l2l2_mass_solve_time_mass();

   deallog << "solved space-time-mass matrix in " << timer.wall_time() << "s" << std::endl;
}

template<int dim>
void DiscretizedFunction<dim>::l2l2_mass_solve_time_mass() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      double factor = 0.0;

      if (i > 0)
         factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;

      if (i < mesh->get_times().size() - 1)
         factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

      Assert(factor != 0.0, ExcInternalError ());
      function_coefficients[i] /= factor;
   }
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_vec_dot(const DiscretizedFunction<dim> & V) const {
  // remember to sync this implementation with l2_norm and all l2 adjoints!
   // uses vector l2 norm in time and vector l2 norm in space
   // (only approx to L2(0,T, L2) inner product if spatial and temporal grid is uniform!
   double result = 0;

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      double doti = (function_coefficients[i] * V.function_coefficients[i]) / function_coefficients[i].size();
      result += doti;
   }

   return result;
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_vec_norm() const {
   // remember to sync this implementation with l2_norm and all l2 adjoints!
   // uses vector l2 norm in time and vector l2 norm in space
   // (only approx to L2(0,T, L2) inner product if spatial and temporal grid is uniform!
   double result = 0;

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      double nrm2 = function_coefficients[i].norm_sqr() / function_coefficients[i].size();
      result += nrm2;
   }

   return std::sqrt(result);
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_mass_dot(const DiscretizedFunction<dim> & V) const {
   // remember to sync this implementation with l2_norm and all l2 adjoints!
   // uses trapezoidal rule in time and vector l2 norm in space
   double result = 0.0;

   // trapezoidal rule in time:
   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
            V.function_coefficients[i]);

      if (i > 0)
         result += doti / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

      if (i < mesh->get_times().size() - 1)
         result += doti / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
   }

   // assume that both functions are linear in time (consistent with crank-nicolson!)
   // and integrate that exactly (Simpson rule)
   // problem when mesh changes in time!
   //   for (size_t i = 0; i < mesh->get_times().size(); i++) {
   //      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
   //            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));
   //
   //      double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            V.function_coefficients[i]);
   //
   //      if (i > 0)
   //         result += doti / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
   //
   //      if (i < mesh->get_times().size() - 1)
   //         result += doti / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }
   //
   //   for (size_t i = 0; i < mesh->get_times().size() - 1; i++) {
   //      Assert(function_coefficients[i].size() == V.function_coefficients[i+1].size(),
   //            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i+1].size()));
   //      Assert(function_coefficients[i+1].size() == V.function_coefficients[i].size(),
   //             ExcDimensionMismatch (function_coefficients[i+1].size() , V.function_coefficients[i].size()));
   //
   //      double dot1 = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            V.function_coefficients[i + 1]);
   //      double dot2 = mesh->get_mass_matrix(i + 1)->matrix_scalar_product(function_coefficients[i + 1],
   //            V.function_coefficients[i]);
   //
   //      result += (dot1 + dot2) / 6 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }

   return result;
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_mass_norm() const {
   // remember to sync this implementation with l2_dot and all l2 adjoints!
   double result = 0;

   // trapezoidal rule in time:
   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);

      if (i > 0)
         result += nrm2 / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

      if (i < mesh->get_times().size() - 1)
         result += nrm2 / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
   }

   // assume that function is linear in time (consistent with crank-nicolson!)
   // and integrate that exactly (Simpson rule)
   // problem when mesh changes in time!
   //   for (size_t i = 0; i < mesh->get_times().size(); i++) {
   //      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);
   //
   //      if (i > 0)
   //         result += nrm2 / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
   //
   //      if (i < mesh->get_times().size() - 1)
   //         result += nrm2 / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }
   //
   //   for (size_t i = 0; i < mesh->get_times().size() - 1; i++) {
   //      double tmp = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            function_coefficients[i + 1]);
   //
   //      result += tmp / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }

   return std::sqrt(result);
}

template<int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string filename, std::string name) const {
   write_pvd(path, filename, name, name + "_prime");
}

template<int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string filename, std::string name,
      std::string name_deriv) const {
   Assert(mesh, ExcNotInitialized());

   LogStream::Prefix p("write_pvd");
   deallog << "Writing " << path << filename << ".pvd" << std::endl;

   Assert(mesh->get_times().size() < 10000, ExcNotImplemented()); // 4 digits are ok
   std::vector<std::pair<double, std::string>> times_and_names(mesh->get_times().size(),
         std::pair<double, std::string>(0.0, ""));

   if (mesh->allows_parallel_access()) {
      Threads::TaskGroup<void> task_group;
      {
         LogStream::Prefix pp("write_vtu");

         for (size_t i = 0; i < mesh->get_times().size(); i++) {
            const std::string vtuname = filename + "-" + Utilities::int_to_string(i, 4) + ".vtu";
            times_and_names[i] = std::pair<double, std::string>(mesh->get_time(i), vtuname);

            task_group += Threads::new_task(&DiscretizedFunction<dim>::write_vtk, *this, name, name_deriv,
                  path + vtuname, i);
         }

         task_group.join_all();
      }
   } else {
      for (size_t i = 0; i < mesh->get_times().size(); i++) {
         const std::string vtuname = filename + "-" + Utilities::int_to_string(i, 4) + ".vtu";
         times_and_names[i] = std::pair<double, std::string>(mesh->get_time(i), vtuname);

         write_vtk(name, name_deriv, path + vtuname, i);
      }
   }

   std::ofstream pvd_output(path + filename + ".pvd");
   AssertThrow(pvd_output, ExcInternalError());

   DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template<int dim>
void DiscretizedFunction<dim>::write_vtk(const std::string name, const std::string name_deriv,
      const std::string filename, size_t i) const {
   DataOut<dim> data_out;

   data_out.attach_dof_handler(*mesh->get_dof_handler(i));
   data_out.add_data_vector(function_coefficients[i], name);

   if (store_derivative)
      data_out.add_data_vector(derivative_coefficients[i], name_deriv);

   data_out.build_patches();

   deallog << "Writing " << filename << std::endl;

   std::ofstream output(filename.c_str());
   AssertThrow(output, ExcInternalError());

   data_out.write_vtu(output);
}

template<int dim> typename DiscretizedFunction<dim>::Norm DiscretizedFunction<dim>::get_norm() const {
   return norm_type;
}

template<int dim> void DiscretizedFunction<dim>::set_norm(Norm norm) {
   this->norm_type = norm;
}

template<int dim>
double DiscretizedFunction<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   Assert(mesh->near_enough(Function<dim>::get_time(), cur_time_idx), ExcNotImplemented());
   Assert(cur_time_idx >= 0 && cur_time_idx < mesh->get_times().size(),
         ExcIndexRange(cur_time_idx, 0, mesh->get_times().size()));

   return VectorTools::point_value(*mesh->get_dof_handler(cur_time_idx), function_coefficients[cur_time_idx], p);
}

template<int dim>
Tensor<1, dim, double> DiscretizedFunction<dim>::gradient(const Point<dim> &p,
      const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   Assert(mesh->near_enough(Function<dim>::get_time(), cur_time_idx), ExcNotImplemented());
   Assert(cur_time_idx >= 0 && cur_time_idx < mesh->get_times().size(),
         ExcIndexRange(cur_time_idx, 0, mesh->get_times().size()));

   return VectorTools::point_gradient(*mesh->get_dof_handler(cur_time_idx), function_coefficients[cur_time_idx], p);
}

template<int dim>
double DiscretizedFunction<dim>::get_time_index() const {
   return cur_time_idx;
}

template<int dim>
void DiscretizedFunction<dim>::set_time(const double new_time) {
   Function<dim>::set_time(new_time);
   cur_time_idx = mesh->find_nearest_time(new_time);
}

template<int dim>
std::shared_ptr<SpaceTimeMesh<dim> > DiscretizedFunction<dim>::get_mesh() const {
   return mesh;
}

template<int dim>
double DiscretizedFunction<dim>::relative_error(const DiscretizedFunction<dim>& other) const {
   DiscretizedFunction<dim> tmp(*this);
   tmp -= other;

   double denom = this->norm();
   return tmp.norm() / (denom == 0.0 ? 1.0 : denom);
}

} /* namespace forward */
} /* namespace wavepi */

