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
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, bool store_derivative)
      : store_derivative(store_derivative), cur_time_idx(0), mesh(mesh) {
   Assert(mesh, ExcNotInitialized());

   function_coefficients.reserve(mesh->length());

   if (store_derivative)
      derivative_coefficients.reserve(mesh->length());

   for (size_t i = 0; i < mesh->length(); i++) {
      function_coefficients.emplace_back(mesh->n_dofs(i));

      if (store_derivative)
         derivative_coefficients.emplace_back(mesh->n_dofs(i));
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

   function_coefficients.reserve(mesh->length());

   for (size_t i = 0; i < mesh->length(); i++) {
      auto dof_handler = mesh->get_dof_handler(i);

      Vector<double> tmp(dof_handler->n_dofs());
      function.set_time(mesh->get_time(i));

      VectorTools::interpolate(*dof_handler, function, tmp);
      mesh->get_constraint_matrix(i)->distribute(tmp);

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
   AssertThrow(store_derivative, ExcInternalError());
   AssertThrow(mesh, ExcNotInitialized());

   DiscretizedFunction<dim> result(mesh, false);
   result.function_coefficients = this->derivative_coefficients;

   return result;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative() const {
   AssertThrow(mesh, ExcNotInitialized());
   AssertThrow(mesh->length() > 1, ExcInternalError());
   AssertThrow(!store_derivative, ExcInternalError()); // why would you want to calculate it in this case?

   DiscretizedFunction<dim> result(mesh, false);

   /* implementation for constant mesh */
   /*
    for (size_t i = 0; i < mesh->length(); i++) {
    if (i < mesh->length() - 1)
    AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(),
    ExcNotImplemented());

    if (i == 0) {
    result.function_coefficients[i] = function_coefficients[i + 1];
    result.function_coefficients[i] -= function_coefficients[i];
    result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i);
    } else if (i == mesh->length() - 1) {
    result.function_coefficients[i] = function_coefficients[i];
    result.function_coefficients[i] -= function_coefficients[i - 1];
    result.function_coefficients[i] /= mesh->get_time(i) - mesh->get_time(i - 1);
    } else {
    result.function_coefficients[i] = function_coefficients[i + 1];
    result.function_coefficients[i] -= function_coefficients[i - 1];
    result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i - 1);
    }
    }
    */

   /* naive, but working implementation for non-constant mesh */
   /*
    for (size_t i = 0; i < mesh->length(); i++) {
    if (i == 0) {
    Vector<double> next_coefficients = function_coefficients[i + 1];
    mesh->transfer(i + 1, i, { &next_coefficients });

    result.function_coefficients[i] = next_coefficients;
    result.function_coefficients[i] -= function_coefficients[i];
    result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i);
    } else if (i == mesh->length() - 1) {
    Vector<double> last_coefficients = function_coefficients[i - 1];
    mesh->transfer(i - 1, i, { &last_coefficients });

    result.function_coefficients[i] = function_coefficients[i];
    result.function_coefficients[i] -= last_coefficients;
    result.function_coefficients[i] /= mesh->get_time(i) - mesh->get_time(i - 1);
    } else {
    Vector<double> last_coefficients = function_coefficients[i - 1];
    Vector<double> next_coefficients = function_coefficients[i + 1];

    mesh->transfer(i - 1, i, { &last_coefficients });
    mesh->transfer(i + 1, i, { &next_coefficients });

    result.function_coefficients[i] = next_coefficients;
    result.function_coefficients[i] -= last_coefficients;
    result.function_coefficients[i] /= mesh->get_time(i + 1) - mesh->get_time(i - 1);
    }
    }
    */

   /* better: forward- and backward sweep */
   // forward sweep
   for (size_t i = 0; i < mesh->length(); i++) {
      if (i == 0) {
         result.function_coefficients[i].equ(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)),
               function_coefficients[i]);
      } else if (i == mesh->length() - 1) {
         Vector<double> last_coefficients = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &last_coefficients });

         result.function_coefficients[i] = function_coefficients[i];
         result.function_coefficients[i] -= last_coefficients;
         result.function_coefficients[i] /= mesh->get_time(i) - mesh->get_time(i - 1);
      } else {
         Vector<double> last_coefficients = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &last_coefficients });

         result.function_coefficients[i].equ(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i - 1)),
               last_coefficients);
      }
   }

   // backward sweep
   for (size_t j = 0; j < mesh->length(); j++) {
      size_t i = mesh->length() - 1 - j;

      if (i == 0) {
         Vector<double> next_coefficients = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &next_coefficients });

         result.function_coefficients[i].add(1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)),
               next_coefficients);
      } else if (i == mesh->length() - 1) {
         // nothing to be done
      } else {
         Vector<double> next_coefficients = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &next_coefficients });

         result.function_coefficients[i].add(1.0 / (mesh->get_time(i + 1) - mesh->get_time(i - 1)),
               next_coefficients);
      }
   }

   return result;
}

template<int dim>
double DiscretizedFunction<dim>::absolute_error(Function<dim>& other) const {
   return absolute_error(other, nullptr);
}

template<int dim>
double DiscretizedFunction<dim>::absolute_error(Function<dim>& other, double* norm_out) const {
   LogStream::Prefix p("calculate_error");

   DiscretizedFunction<dim> tmp(mesh, other);
   tmp.set_norm(norm_type);

   if (norm_out)
      *norm_out = tmp.norm();

   tmp -= *this;

   return tmp.norm();
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative_transpose() const {
   AssertThrow(mesh, ExcNotInitialized());

   // because of the special cases
   AssertThrow(mesh->length() > 3, ExcInternalError());

   DiscretizedFunction<dim> result(mesh, false);

   /* implementation for constant mesh */
   /*
    for (size_t i = 0; i < mesh->length(); i++) {
    auto dest = &result.function_coefficients[i];

    if (i < mesh->length() - 1)
    AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(),
    ExcNotImplemented());

    if (i == 0) {
    *dest = function_coefficients[i + 1];
    dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
    -1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
    } else if (i == 1) {
    *dest = function_coefficients[i + 1];
    dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
    1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i - 1]);
    } else if (i == mesh->length() - 1) {
    *dest = function_coefficients[i];
    dest->sadd(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)),
    1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
    } else if (i == mesh->length() - 2) {
    *dest = function_coefficients[i + 1];
    dest->sadd(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)),
    1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
    } else {
    *dest = function_coefficients[i + 1];
    dest->sadd(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)),
    1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
    }
    }
    */

   /* naive, but working implementation for non-constant mesh */
   /*for (size_t i = 0; i < mesh->length(); i++) {
    auto& dest = result.function_coefficients[i];

    if (i == 0) {
    Vector<double> tmp = function_coefficients[i + 1];
    mesh->transfer(i + 1, i, {&tmp});
    dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);

    dest.add(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
    } else if (i == 1) {
    Vector<double> tmp = function_coefficients[i + 1];
    mesh->transfer(i + 1, i, {&tmp});
    dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);

    tmp = function_coefficients[i - 1];
    mesh->transfer(i - 1, i, {&tmp});
    dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), tmp);
    } else if (i == mesh->length() - 1) {
    dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i]);

    Vector<double> tmp = function_coefficients[i - 1];
    mesh->transfer(i - 1, i, &tmp);
    dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
    } else if (i == mesh->length() - 2) {
    Vector<double> tmp = function_coefficients[i + 1];
    mesh->transfer(i + 1, i, {&tmp});
    dest.add(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), tmp);

    tmp = function_coefficients[i - 1];
    mesh->transfer(i - 1, i, {&tmp});
    dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
    } else {
    Vector<double> tmp = function_coefficients[i + 1];
    mesh->transfer(i + 1, i, {&tmp});
    dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);

    tmp = function_coefficients[i - 1];
    mesh->transfer(i - 1, i, {&tmp});
    dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
    }
    }
    */

   /* better: forward- and backward sweep */
   // forward sweep
   for (size_t i = 0; i < mesh->length(); i++) {
      auto& dest = result.function_coefficients[i];

      if (i == 0) {
         dest.add(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
      } else if (i == 1) {
         Vector<double> tmp = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &tmp });
         dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), tmp);
      } else if (i == mesh->length() - 1) {
         Vector<double> tmp = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &tmp });
         dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);

         dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i]);
      } else if (i == mesh->length() - 2) {
         Vector<double> tmp = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &tmp });
         dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
      } else {
         Vector<double> tmp = function_coefficients[i - 1];
         mesh->transfer(i - 1, i, { &tmp });
         dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
      }
   }

   // backward sweep
   for (size_t j = 0; j < mesh->length(); j++) {
      size_t i = mesh->length() - 1 - j;
      auto& dest = result.function_coefficients[i];

      if (i == 0) {
         Vector<double> tmp = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &tmp });
         dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
      } else if (i == 1) {
         Vector<double> tmp = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &tmp });
         dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
      } else if (i == mesh->length() - 1) {
         // nothing to be done
      } else if (i == mesh->length() - 2) {
         Vector<double> tmp = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &tmp });
         dest.add(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), tmp);
      } else {
         Vector<double> tmp = function_coefficients[i + 1];
         mesh->transfer(i + 1, i, { &tmp });
         dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
      }
   }

   return result;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(double x) {
   Assert(mesh, ExcNotInitialized());

   for (size_t i = 0; i < mesh->length(); i++) {
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
   for (size_t i = 0; i < mesh->length(); i++) {
      function_coefficients[i] *= factor;

      if (store_derivative)
         derivative_coefficients[i] *= factor;
   }

   return *this;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(const DiscretizedFunction<dim>& like) {
   Assert(!like.store_derivative, ExcInternalError ());
   Assert(like.mesh, ExcNotInitialized());

   DiscretizedFunction<dim> res(like.mesh);

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(-1, 1);

   for (size_t i = 0; i < res.mesh->length(); i++)
      for (size_t j = 0; j < res.function_coefficients[i].size(); j++)
         res.function_coefficients[i][j] = distribution(generator);

   return res;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(const DiscretizedFunction<dim>& like, double norm) {
   DiscretizedFunction<dim> result = noise(like);
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

   for (size_t i = 0; i < mesh->length(); i++) {
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

   for (size_t i = 0; i < mesh->length(); i++) {
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

   for (size_t i = 0; i < mesh->length(); i++) {
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
      case Norm::Vector:
         return norm_vector();
      case Norm::L2L2:
         return norm_l2l2();
      case Norm::H1L2:
         return norm_h1l2();
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
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
      case Norm::Vector:
         return dot_vector(V);
      case Norm::L2L2:
         return dot_l2l2(V);
      case Norm::H1L2:
         return dot_h1l2(V);
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
   }

   Assert(false, ExcInternalError ());
   return 0.0;
}

template<int dim>
double DiscretizedFunction<dim>::dot(const DiscretizedFunction<dim> & V) const {
   return (*this) * V;
}

template<int dim>
bool DiscretizedFunction<dim>::is_hilbert() const {
   return norm_type != Norm::Invalid;
}

template<int dim> void DiscretizedFunction<dim>::dot_transform_vector() {
}
template<int dim> void DiscretizedFunction<dim>::dot_transform_inverse_vector() {
}
template<int dim> void DiscretizedFunction<dim>::dot_solve_mass_and_transform_vector() {
   solve_mass();
}
template<int dim> void DiscretizedFunction<dim>::dot_mult_mass_and_transform_inverse_vector() {
   mult_mass();
}

template<int dim>
void DiscretizedFunction<dim>::dot_transform() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case Norm::Vector:
         dot_transform_vector();
         return;
      case Norm::L2L2:
         dot_transform_l2l2();
         return;
      case Norm::H1L2:
         dot_transform_h1l2();
         return;
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::dot_transform_inverse() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case Norm::Vector:
         dot_transform_inverse_vector();
         return;
      case Norm::L2L2:
         dot_transform_inverse_l2l2();
         return;
      case Norm::H1L2:
         dot_transform_inverse_h1l2();
         return;
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::dot_solve_mass_and_transform() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case Norm::Vector:
         dot_solve_mass_and_transform_vector();
         return;
      case Norm::L2L2:
         dot_solve_mass_and_transform_l2l2();
         return;
      case Norm::H1L2:
         dot_solve_mass_and_transform_h1l2();
         return;
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::dot_mult_mass_and_transform_inverse() {
   Assert(mesh, ExcNotInitialized());

   switch (norm_type) {
      case Norm::Vector:
         dot_mult_mass_and_transform_inverse_vector();
         return;
      case Norm::L2L2:
         dot_mult_mass_and_transform_inverse_l2l2();
         return;
      case Norm::H1L2:
         dot_mult_mass_and_transform_inverse_h1l2();
         return;
      default:
         AssertThrow(false, ExcMessage("norm_type == Invalid"))
   }

   Assert(false, ExcInternalError ());
}

template<int dim>
void DiscretizedFunction<dim>::mult_mass() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->length(); i++) {
      Vector<double> tmp(function_coefficients[i].size());
      mesh->get_mass_matrix(i)->vmult(tmp, function_coefficients[i]);
      function_coefficients[i] = tmp;
   }
}

template<int dim>
void DiscretizedFunction<dim>::dot_transform_l2l2() {
   Assert(!store_derivative, ExcInternalError ());

   mult_mass();
   dot_solve_mass_and_transform_l2l2();
}

template<int dim>
void DiscretizedFunction<dim>::dot_solve_mass_and_transform_l2l2() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->length(); i++) {
      double factor = 0.0;

      if (i > 0)
         factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;

      if (i < mesh->length() - 1)
         factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

      function_coefficients[i] *= factor;
   }
}

template<int dim>
void DiscretizedFunction<dim>::solve_mass() {
   Assert(!store_derivative, ExcInternalError ());

   LogStream::Prefix p("solve_mass");
   Timer timer;
   timer.start();

   for (size_t i = 0; i < mesh->length(); i++) {
      LogStream::Prefix p("step-" + Utilities::int_to_string(i, 4));

      Vector<double> tmp(function_coefficients[i].size());

      SolverControl solver_control(2000, 1e-10 * function_coefficients[i].l2_norm());
      SolverCG<> cg(solver_control);
      PreconditionIdentity precondition = PreconditionIdentity();

      cg.solve(*mesh->get_mass_matrix(i), tmp, function_coefficients[i], precondition);
      function_coefficients[i] = tmp;
   }

   deallog << "solved space-time-mass matrices in " << timer.wall_time() << "s" << std::endl;
}

template<int dim>
void DiscretizedFunction<dim>::dot_transform_inverse_l2l2() {
   Assert(!store_derivative, ExcInternalError ());

   solve_mass();
   dot_mult_mass_and_transform_inverse_l2l2();
}

template<int dim>
void DiscretizedFunction<dim>::dot_mult_mass_and_transform_inverse_l2l2() {
   Assert(!store_derivative, ExcInternalError ());

   for (size_t i = 0; i < mesh->length(); i++) {
      double factor = 0.0;

      if (i > 0)
         factor += std::abs(mesh->get_time(i) - mesh->get_time(i - 1)) / 2.0;

      if (i < mesh->length() - 1)
         factor += std::abs(mesh->get_time(i + 1) - mesh->get_time(i)) / 2.0;

      Assert(factor != 0.0, ExcInternalError ());
      function_coefficients[i] /= factor;
   }
}

template<int dim>
double DiscretizedFunction<dim>::dot_vector(const DiscretizedFunction<dim> & V) const {
   double result = 0;

   for (size_t i = 0; i < mesh->length(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      double doti = (function_coefficients[i] * V.function_coefficients[i]) / function_coefficients[i].size();
      result += doti;
   }

   return result;
}

template<int dim>
double DiscretizedFunction<dim>::norm_vector() const {
   double result = 0;

   for (size_t i = 0; i < mesh->length(); i++) {
      double nrm2 = function_coefficients[i].norm_sqr() / function_coefficients[i].size();
      result += nrm2;
   }

   return std::sqrt(result);
}

template<int dim>
double DiscretizedFunction<dim>::dot_l2l2(const DiscretizedFunction<dim> & V) const {
   double result = 0.0;

   // trapezoidal rule in time:
   for (size_t i = 0; i < mesh->length(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
            V.function_coefficients[i]);

      if (i > 0)
         result += doti / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

      if (i < mesh->length() - 1)
         result += doti / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
   }

   // assume that both functions are linear in time (consistent with crank-nicolson!)
   // and integrate that exactly (Simpson rule)
   // problem when mesh changes in time!
   //   for (size_t i = 0; i < mesh->length(); i++) {
   //      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
   //            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));
   //
   //      double doti = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            V.function_coefficients[i]);
   //
   //      if (i > 0)
   //         result += doti / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
   //
   //      if (i < mesh->length() - 1)
   //         result += doti / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }
   //
   //   for (size_t i = 0; i < mesh->length() - 1; i++) {
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
double DiscretizedFunction<dim>::norm_l2l2() const {
   double result = 0;

   // trapezoidal rule in time:
   for (size_t i = 0; i < mesh->length(); i++) {
      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);

      if (i > 0)
         result += nrm2 / 2 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));

      if (i < mesh->length() - 1)
         result += nrm2 / 2 * (std::abs(mesh->get_time(i + 1) - mesh->get_time(i)));
   }

   // assume that function is linear in time (consistent with crank-nicolson!)
   // and integrate that exactly (Simpson rule)
   // problem when mesh changes in time!
   //   for (size_t i = 0; i < mesh->length(); i++) {
   //      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);
   //
   //      if (i > 0)
   //         result += nrm2 / 3 * (std::abs(mesh->get_time(i) - mesh->get_time(i - 1)));
   //
   //      if (i < mesh->length() - 1)
   //         result += nrm2 / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }
   //
   //   for (size_t i = 0; i < mesh->length() - 1; i++) {
   //      double tmp = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            function_coefficients[i + 1]);
   //
   //      result += tmp / 3 * (std::abs(mesh->get_time(i+1) - mesh->get_time(i)));
   //   }

   return std::sqrt(result);
}

template<int dim> void DiscretizedFunction<dim>::dot_transform_h1l2() {
   AssertThrow(false, ExcNotImplemented())
   // TODO
}

template<int dim> void DiscretizedFunction<dim>::dot_transform_inverse_h1l2() {
   AssertThrow(false, ExcNotImplemented())
   // TODO
}

template<int dim> void DiscretizedFunction<dim>::dot_solve_mass_and_transform_h1l2() {
   AssertThrow(false, ExcNotImplemented())
   // TODO
}

template<int dim> void DiscretizedFunction<dim>::dot_mult_mass_and_transform_inverse_h1l2() {
   AssertThrow(false, ExcNotImplemented())
   // TODO
}

template<int dim> double DiscretizedFunction<dim>::dot_h1l2(const DiscretizedFunction<dim> & V) const {
   AssertThrow(false, ExcNotImplemented())
   // TODO
}

template<int dim>
double DiscretizedFunction<dim>::norm_h1l2() const {
   AssertThrow(false, ExcNotImplemented())
   // TODO
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

   Assert(mesh->length() < 10000, ExcNotImplemented()); // 4 digits are ok
   std::vector<std::pair<double, std::string>> times_and_names(mesh->length(),
         std::pair<double, std::string>(0.0, ""));

   if (mesh->allows_parallel_access()) {
      Threads::TaskGroup<void> task_group;
      {
         LogStream::Prefix pp("write_vtu");

         for (size_t i = 0; i < mesh->length(); i++) {
            const std::string vtuname = filename + "-" + Utilities::int_to_string(i, 4) + ".vtu";
            times_and_names[i] = std::pair<double, std::string>(mesh->get_time(i), vtuname);

            task_group += Threads::new_task(&DiscretizedFunction<dim>::write_vtk, *this, name, name_deriv,
                  path + vtuname, i);
         }

         task_group.join_all();
      }
   } else {
      for (size_t i = 0; i < mesh->length(); i++) {
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
   Assert(cur_time_idx >= 0 && cur_time_idx < mesh->length(), ExcIndexRange(cur_time_idx, 0, mesh->length()));

   return VectorTools::point_value(*mesh->get_dof_handler(cur_time_idx), function_coefficients[cur_time_idx],
         p);
}

template<int dim>
Tensor<1, dim, double> DiscretizedFunction<dim>::gradient(const Point<dim> &p,
      const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   Assert(cur_time_idx >= 0 && cur_time_idx < mesh->length(), ExcIndexRange(cur_time_idx, 0, mesh->length()));

   return VectorTools::point_gradient(*mesh->get_dof_handler(cur_time_idx),
         function_coefficients[cur_time_idx], p);
}

template<int dim>
double DiscretizedFunction<dim>::get_time_index() const {
   return cur_time_idx;
}

template<int dim>
void DiscretizedFunction<dim>::set_time(const double new_time) {
   Function<dim>::set_time(new_time);
   cur_time_idx = mesh->find_time(new_time);
}

template<int dim>
std::shared_ptr<SpaceTimeMesh<dim> > DiscretizedFunction<dim>::get_mesh() const {
   return mesh;
}

template<int dim>
void DiscretizedFunction<dim>::min_max_value(double* min_out, double* max_out) const {
   *min_out = std::numeric_limits<double>::infinity();
   *max_out = -std::numeric_limits<double>::infinity();

   for (size_t i = 0; i < this->length(); i++)
      for (size_t j = 0; j < function_coefficients[i].size(); j++) {
         if (*min_out > function_coefficients[i][j])
            *min_out = function_coefficients[i][j];

         if (*max_out < function_coefficients[i][j])
            *max_out = function_coefficients[i][j];
      }
}

template<int dim>
double DiscretizedFunction<dim>::min_value() const {
   double tmp = std::numeric_limits<double>::infinity();

   for (size_t i = 0; i < this->length(); i++)
      for (size_t j = 0; j < function_coefficients[i].size(); j++)
         if (tmp > function_coefficients[i][j])
            tmp = function_coefficients[i][j];

   return tmp;
}

template<int dim>
double DiscretizedFunction<dim>::max_value() const {
   double tmp = -std::numeric_limits<double>::infinity();

   for (size_t i = 0; i < this->length(); i++)
      for (size_t j = 0; j < function_coefficients[i].size(); j++)
         if (tmp < function_coefficients[i][j])
            tmp = function_coefficients[i][j];

   return tmp;
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

