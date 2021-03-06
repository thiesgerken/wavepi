/*
 * DiscretizedFunction.cpp
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <utility>

#include <base/DiscretizedFunction.h>

using namespace dealii;

namespace wavepi {
namespace base {

inline double square(const double x) { return x * x; }

inline double pow4(const double x) { return x * x * x * x; }

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                              std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm,
                                              bool store_derivative)
    : mesh(mesh), norm_(norm), store_derivative(store_derivative), cur_time_idx(0) {
  Assert(mesh && norm, ExcNotInitialized());

  function_coefficients.reserve(mesh->length());

  if (store_derivative) derivative_coefficients.reserve(mesh->length());

  for (size_t i = 0; i < mesh->length(); i++) {
    function_coefficients.emplace_back(mesh->n_dofs(i));

    if (store_derivative) derivative_coefficients.emplace_back(mesh->n_dofs(i));
  }
}

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
                                              std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm)
    : DiscretizedFunction(mesh, norm, false) {}

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
    : DiscretizedFunction(mesh, std::make_shared<InvalidNorm<DiscretizedFunction<dim>>>()) {}

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Function<dim>& function,
                                              std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm)
    : mesh(mesh), norm_(norm), store_derivative(false), cur_time_idx(0) {
  Assert(mesh && norm, ExcNotInitialized());

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

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Function<dim>& function)
    : DiscretizedFunction(mesh, function, std::make_shared<InvalidNorm<DiscretizedFunction<dim>>>()) {}

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(DiscretizedFunction<dim>&& o)
    : LightFunction<dim>(),
      mesh(std::move(o.mesh)),
      norm_(o.norm_),
      store_derivative(o.store_derivative),
      cur_time_idx(o.cur_time_idx),
      function_coefficients(std::move(o.function_coefficients)),
      derivative_coefficients(std::move(o.derivative_coefficients)) {
  Assert(mesh, ExcNotInitialized());

  o.mesh = std::shared_ptr<SpaceTimeMesh<dim>>();
}

template <int dim>
DiscretizedFunction<dim>::DiscretizedFunction(const DiscretizedFunction<dim>& o)
    : LightFunction<dim>(),
      mesh(o.mesh),
      norm_(o.norm_),
      store_derivative(o.store_derivative),
      cur_time_idx(o.cur_time_idx),
      function_coefficients(o.function_coefficients),
      derivative_coefficients(o.derivative_coefficients) {
  Assert(mesh, ExcNotInitialized());
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(DiscretizedFunction<dim>&& o) {
  mesh                    = std::move(o.mesh);
  norm_                   = o.norm_;
  store_derivative        = o.store_derivative;
  cur_time_idx            = o.cur_time_idx;
  function_coefficients   = std::move(o.function_coefficients);
  derivative_coefficients = std::move(o.derivative_coefficients);

  o.mesh = std::shared_ptr<SpaceTimeMesh<dim>>();

  Assert(mesh, ExcNotInitialized());
  return *this;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(const DiscretizedFunction<dim>& o) {
  mesh                    = o.mesh;
  norm_                   = o.norm_;
  store_derivative        = o.store_derivative;
  cur_time_idx            = o.cur_time_idx;
  function_coefficients   = o.function_coefficients;
  derivative_coefficients = o.derivative_coefficients;

  Assert(mesh, ExcNotInitialized());
  return *this;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::derivative() const {
  AssertThrow(store_derivative, ExcInternalError());
  AssertThrow(mesh, ExcNotInitialized());

  DiscretizedFunction<dim> result(mesh);
  result.function_coefficients = this->derivative_coefficients;

  return result;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative() const {
  AssertThrow(mesh, ExcNotInitialized());
  AssertThrow(mesh->length() > 1, ExcInternalError());
  // AssertThrow(!store_derivative, ExcInternalError()); // why would you want to calculate it in this case?

  DiscretizedFunction<dim> result(mesh, norm_);

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
      result.function_coefficients[i].equ(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
    } else if (i == mesh->length() - 1) {
      Vector<double> last_coefficients = function_coefficients[i - 1];
      mesh->transfer(i - 1, i, {&last_coefficients});

      result.function_coefficients[i] = function_coefficients[i];
      result.function_coefficients[i] -= last_coefficients;
      result.function_coefficients[i] /= mesh->get_time(i) - mesh->get_time(i - 1);
    } else {
      Vector<double> last_coefficients = function_coefficients[i - 1];
      mesh->transfer(i - 1, i, {&last_coefficients});

      result.function_coefficients[i].equ(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i - 1)), last_coefficients);
    }
  }

  // backward sweep
  for (size_t j = 0; j < mesh->length(); j++) {
    size_t i = mesh->length() - 1 - j;

    if (i == 0) {
      Vector<double> next_coefficients = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&next_coefficients});

      result.function_coefficients[i].add(1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), next_coefficients);
    } else if (i == mesh->length() - 1) {
      // nothing to be done
    } else {
      Vector<double> next_coefficients = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&next_coefficients});

      result.function_coefficients[i].add(1.0 / (mesh->get_time(i + 1) - mesh->get_time(i - 1)), next_coefficients);
    }
  }

  return result;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_second_derivative() const {
  AssertThrow(mesh, ExcNotInitialized());
  AssertThrow(mesh->length() > 1, ExcInternalError());

  DiscretizedFunction<dim> result(mesh, norm_);

  /* implementation for constant mesh */

  for (size_t i = 0; i < mesh->length(); i++) {
    if (i < mesh->length() - 1)
      AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(), ExcNotImplemented());

    if (i == 0) {
      result.function_coefficients[i] = function_coefficients[i];
      result.function_coefficients[i].add(-2, function_coefficients[i + 1]);
      result.function_coefficients[i].add(1, function_coefficients[i + 2]);
      result.function_coefficients[i] /= square(mesh->get_time(i + 1) - mesh->get_time(i));
    } else if (i == mesh->length() - 1) {
      result.function_coefficients[i] = function_coefficients[i];
      result.function_coefficients[i].add(-2, function_coefficients[i - 1]);
      result.function_coefficients[i].add(1, function_coefficients[i - 2]);
      result.function_coefficients[i] /= square(mesh->get_time(i) - mesh->get_time(i - 1));
    } else {
      result.function_coefficients[i].equ(-2, function_coefficients[i]);
      result.function_coefficients[i].add(1, function_coefficients[i + 1]);
      result.function_coefficients[i].add(1, function_coefficients[i - 1]);
      result.function_coefficients[i] /= square(mesh->get_time(i + 1) - mesh->get_time(i - 1)) / 4;
    }
  }

  return result;
}

template <int dim>
double DiscretizedFunction<dim>::absolute_error(DiscretizedFunction<dim>& other, double* norm_out) const {
  LogStream::Prefix p("calculate_error");
  AssertThrow(other.mesh == mesh, ExcInternalError());

  DiscretizedFunction<dim> tmp = other;
  tmp.set_norm(norm_);

  if (norm_out) *norm_out = tmp.norm();

  tmp -= *this;

  return tmp.norm();
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_derivative_transpose() const {
  AssertThrow(mesh, ExcNotInitialized());

  // because of the special cases
  AssertThrow(mesh->length() > 3, ExcInternalError());

  DiscretizedFunction<dim> result(mesh, norm_);

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
      mesh->transfer(i - 1, i, {&tmp});
      dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), tmp);
    } else if (i == mesh->length() - 1) {
      Vector<double> tmp = function_coefficients[i - 1];
      mesh->transfer(i - 1, i, {&tmp});
      dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);

      dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i]);
    } else if (i == mesh->length() - 2) {
      Vector<double> tmp = function_coefficients[i - 1];
      mesh->transfer(i - 1, i, {&tmp});
      dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
    } else {
      Vector<double> tmp = function_coefficients[i - 1];
      mesh->transfer(i - 1, i, {&tmp});
      dest.add(1.0 / (mesh->get_time(i) - mesh->get_time(i - 2)), tmp);
    }
  }

  // backward sweep
  for (size_t j = 0; j < mesh->length(); j++) {
    size_t i   = mesh->length() - 1 - j;
    auto& dest = result.function_coefficients[i];

    if (i == 0) {
      Vector<double> tmp = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&tmp});
      dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
    } else if (i == 1) {
      Vector<double> tmp = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&tmp});
      dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
    } else if (i == mesh->length() - 1) {
      // nothing to be done
    } else if (i == mesh->length() - 2) {
      Vector<double> tmp = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&tmp});
      dest.add(-1.0 / (mesh->get_time(i + 1) - mesh->get_time(i)), tmp);
    } else {
      Vector<double> tmp = function_coefficients[i + 1];
      mesh->transfer(i + 1, i, {&tmp});
      dest.add(-1.0 / (mesh->get_time(i + 2) - mesh->get_time(i)), tmp);
    }
  }

  return result;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::calculate_second_derivative_transpose() const {
  AssertThrow(mesh, ExcNotInitialized());

  // because of the special cases
  AssertThrow(mesh->length() > 3, ExcInternalError());

  DiscretizedFunction<dim> result(mesh, norm_);

  /* implementation for constant mesh */

  for (size_t i = 0; i < mesh->length(); i++) {
    auto dest = &result.function_coefficients[i];

    if (i < mesh->length() - 1)
      AssertThrow(function_coefficients[i + 1].size() == function_coefficients[i].size(), ExcNotImplemented());

    if (i == 0) {
      dest->equ(1.0 / square(mesh->get_time(i + 1) - mesh->get_time(i)), function_coefficients[i]);
      dest->add(1.0 * 4 / square(mesh->get_time(i + 2) - mesh->get_time(i)), function_coefficients[i + 1]);
    } else if (i == 1) {
      dest->equ(-2.0 * 4 / square(mesh->get_time(i + 1) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(-2.0 / square(mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i - 1]);
      dest->add(1.0 * 4 / square(mesh->get_time(i + 2) - mesh->get_time(i)), function_coefficients[i + 1]);
    } else if (i == 2) {
      dest->equ(-2.0 * 4 / square(mesh->get_time(i + 1) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(1.0 * 4 / square(mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      dest->add(1.0 * 4 / square(mesh->get_time(i + 2) - mesh->get_time(i)), function_coefficients[i + 1]);
      dest->add(1.0 / square(mesh->get_time(i - 2) - mesh->get_time(i - 1)), function_coefficients[i - 2]);
    } else if (i == mesh->length() - 1) {
      dest->equ(1.0 / square(mesh->get_time(i) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(1.0 * 4 / square(mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
    } else if (i == mesh->length() - 2) {
      dest->equ(-2.0 * 4 / square(mesh->get_time(i + 1) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(-2.0 / square(mesh->get_time(i) - mesh->get_time(i + 1)), function_coefficients[i + 1]);
      dest->add(1.0 * 4 / square(mesh->get_time(i - 2) - mesh->get_time(i)), function_coefficients[i - 1]);
    } else if (i == mesh->length() - 3) {
      dest->equ(-2.0 * 4 / square(mesh->get_time(i + 1) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(1.0 * 4 / square(mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      dest->add(1.0 * 4 / square(mesh->get_time(i + 2) - mesh->get_time(i)), function_coefficients[i + 1]);
      dest->add(1.0 / square(mesh->get_time(i + 2) - mesh->get_time(i + 1)), function_coefficients[i + 2]);
    } else {
      dest->equ(-2.0 * 4 / square(mesh->get_time(i + 1) - mesh->get_time(i - 1)), function_coefficients[i]);
      dest->add(1.0 * 4 / square(mesh->get_time(i) - mesh->get_time(i - 2)), function_coefficients[i - 1]);
      dest->add(1.0 * 4 / square(mesh->get_time(i + 2) - mesh->get_time(i)), function_coefficients[i + 1]);
    }
  }

  return result;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(double x) {
  AssertThrow(mesh, ExcNotInitialized());

  for (size_t i = 0; i < mesh->length(); i++) {
    function_coefficients[i] = x;

    if (store_derivative) derivative_coefficients[i] = 0.0;
  }

  return *this;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator+=(double offset) {
  // note that this is correct independent of store_derivative.

  for (size_t i = 0; i < mesh->length(); i++) {
    function_coefficients[i].add(offset);
  }

  return *this;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator+=(const DiscretizedFunction<dim>& V) {
  AssertThrow(norm_ && V.norm_, ExcNotInitialized());
  AssertThrow(*norm_ == *V.norm_, ExcMessage("DiscretizedFunction<dim>::operator+= : Norms not compatible"));
  this->add(1.0, V);

  return *this;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator-=(const DiscretizedFunction<dim>& V) {
  AssertThrow(norm_ && V.norm_, ExcNotInitialized());
  AssertThrow(*norm_ == *V.norm_, ExcMessage("DiscretizedFunction<dim>::operator-= : Norms not compatible"));
  this->add(-1.0, V);

  return *this;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator*=(const double factor) {
  AssertThrow(mesh, ExcNotInitialized());

  for (size_t i = 0; i < mesh->length(); i++) {
    function_coefficients[i] *= factor;

    if (store_derivative) derivative_coefficients[i] *= factor;
  }

  return *this;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(const DiscretizedFunction<dim>& like) {
  Assert(!like.store_derivative, ExcInternalError());

  DiscretizedFunction<dim> res = noise(like.mesh);
  res.set_norm(like.get_norm());

  return res;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(std::shared_ptr<SpaceTimeMesh<dim>> mesh) {
  Assert(mesh, ExcNotInitialized());

  DiscretizedFunction<dim> res(mesh);

  // auto time = std::chrono::high_resolution_clock::now();
  // std::default_random_engine generator(time.time_since_epoch().count() % 1000000);
  std::default_random_engine generator(2307);
  std::uniform_real_distribution<double> distribution(-1, 1);

  for (size_t i = 0; i < res.mesh->length(); i++)
    for (size_t j = 0; j < res.function_coefficients[i].size(); j++)
      res.function_coefficients[i][j] = distribution(generator);

  return res;
}

template <int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::noise(const DiscretizedFunction<dim>& like, double norm) {
  DiscretizedFunction<dim> result = noise(like);

  result *= norm / result.norm();

  return result;
}

template <int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator/=(const double factor) {
  return this->operator*=(1.0 / factor);
}

template <int dim>
void DiscretizedFunction<dim>::pointwise_multiplication(const DiscretizedFunction<dim>& V) {
  Assert(mesh, ExcNotInitialized());
  Assert(mesh == V.mesh, ExcInternalError());
  Assert(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError());

  for (size_t i = 0; i < mesh->length(); i++) {
    Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
           ExcDimensionMismatch(function_coefficients[i].size(), V.function_coefficients[i].size()));

    if (store_derivative) {
      Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
             ExcDimensionMismatch(derivative_coefficients[i].size(), V.derivative_coefficients[i].size()));

      derivative_coefficients[i].scale(V.function_coefficients[i]);

      Vector<double> tmp = function_coefficients[i];
      tmp.scale(V.derivative_coefficients[i]);

      derivative_coefficients[i] += tmp;
    }

    function_coefficients[i].scale(V.function_coefficients[i]);
  }
}

template <int dim>
void DiscretizedFunction<dim>::add(const double a, const DiscretizedFunction<dim>& V) {
  AssertThrow(mesh && norm_ && V.norm_, ExcNotInitialized());
  AssertThrow(mesh == V.mesh, ExcInternalError());
  AssertThrow(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError());
  AssertThrow(*norm_ == *V.norm_, ExcMessage("DiscretizedFunction<dim>::add : Norms not compatible"));

  for (size_t i = 0; i < mesh->length(); i++) {
    Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
           ExcDimensionMismatch(function_coefficients[i].size(), V.function_coefficients[i].size()));

    function_coefficients[i].add(a, V.function_coefficients[i]);

    if (store_derivative) {
      Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
             ExcDimensionMismatch(derivative_coefficients[i].size(), V.derivative_coefficients[i].size()));

      derivative_coefficients[i].add(a, V.derivative_coefficients[i]);
    }
  }
}

template <int dim>
void DiscretizedFunction<dim>::sadd(const double s, const double a, const DiscretizedFunction<dim>& V) {
  AssertThrow(mesh && norm_ && V.norm_, ExcNotInitialized());
  AssertThrow(mesh == V.mesh, ExcInternalError());
  AssertThrow(!store_derivative || (store_derivative == V.store_derivative), ExcInternalError());
  AssertThrow(*norm_ == *V.norm_, ExcMessage("DiscretizedFunction<dim>::sadd : Norms not compatible"));

  for (size_t i = 0; i < mesh->length(); i++) {
    Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
           ExcDimensionMismatch(function_coefficients[i].size(), V.function_coefficients[i].size()));

    function_coefficients[i].sadd(s, a, V.function_coefficients[i]);

    if (store_derivative) {
      Assert(derivative_coefficients[i].size() == V.derivative_coefficients[i].size(),
             ExcDimensionMismatch(derivative_coefficients[i].size(), V.derivative_coefficients[i].size()));

      derivative_coefficients[i].sadd(s, a, V.derivative_coefficients[i]);
    }
  }
}

template <int dim>
void DiscretizedFunction<dim>::throw_away_derivative() {
  store_derivative        = false;
  derivative_coefficients = std::vector<Vector<double>>();
}

template <int dim>
double DiscretizedFunction<dim>::norm() const {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  return norm_->norm(*this);
}

template <int dim>
double DiscretizedFunction<dim>::operator*(const DiscretizedFunction<dim>& V) const {
  Assert(mesh && norm_ && V.norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());
  AssertThrow(*V.norm_ == *norm_, ExcMessage("DiscretizedFunction<dim>::operator* : Norms not compatible"));

  return norm_->dot(*this, V);
}

template <int dim>
double DiscretizedFunction<dim>::dot(const DiscretizedFunction<dim>& V) const {
  return (*this) * V;
}

template <int dim>
void DiscretizedFunction<dim>::duality_mapping_lp(double p) {
  AssertThrow(p > 1, ExcMessage("duality_mapping_lp: p has to be larger than 1!"));

  for (size_t i = 0; i < mesh->length(); i++)
    for (size_t j = 0; j < function_coefficients[i].size(); j++)
      if (function_coefficients[i][j] != 0.0)
        function_coefficients[i][j] =
            std::pow(std::abs(function_coefficients[i][j]), p - 2) * function_coefficients[i][j];
}

template <int dim>
void DiscretizedFunction<dim>::duality_mapping(double p) {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->duality_mapping(*this, p);
}

template <int dim>
void DiscretizedFunction<dim>::duality_mapping_dual(double q) {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->duality_mapping_dual(*this, q);
}

template <int dim>
double DiscretizedFunction<dim>::norm_dual() const {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  return norm_->norm_dual(*this);
}

template <int dim>
double DiscretizedFunction<dim>::norm_p(double p) {
  AssertThrow(p >= 1, ExcMessage("norm_p: p has to be >= 1!"));
  double result = 0.0;

  for (size_t i = 0; i < mesh->length(); i++)
    for (size_t j = 0; j < function_coefficients[i].size(); j++)
      result += std::pow(std::abs(function_coefficients[i][j]), p);

  return std::pow(result, 1 / p);
}

template <int dim>
bool DiscretizedFunction<dim>::hilbert() const {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  return norm_->hilbert();
}

template <int dim>
void DiscretizedFunction<dim>::dot_transform() {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->dot_transform(*this);
}

template <int dim>
void DiscretizedFunction<dim>::dot_transform_inverse() {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->dot_transform_inverse(*this);
}

template <int dim>
void DiscretizedFunction<dim>::dot_solve_mass_and_transform() {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->dot_solve_mass_and_transform(*this);
}

template <int dim>
void DiscretizedFunction<dim>::dot_mult_mass_and_transform_inverse() {
  Assert(mesh && norm_, ExcNotInitialized());
  Assert(!store_derivative, ExcInternalError());

  norm_->dot_mult_mass_and_transform_inverse(*this);
}

template <int dim>
void DiscretizedFunction<dim>::mult_mass() {
  Assert(!store_derivative, ExcInternalError());

  for (size_t i = 0; i < mesh->length(); i++) {
    Vector<double> tmp(function_coefficients[i].size());
    mesh->get_mass_matrix(i)->vmult(tmp, function_coefficients[i]);
    function_coefficients[i] = tmp;
  }
}

template <int dim>
void DiscretizedFunction<dim>::solve_mass() {
  Assert(!store_derivative, ExcInternalError());

  LogStream::Prefix p("solve_mass");
  Timer timer;
  timer.start();

  // PreconditionIdentity precondition;
  PreconditionSSOR<SparseMatrix<double>> precondition;
  precondition.initialize(*mesh->get_mass_matrix(0), PreconditionSSOR<SparseMatrix<double>>::AdditionalData(1.0));

  for (size_t i = 0; i < mesh->length(); i++) {
    LogStream::Prefix p("step-" + Utilities::int_to_string(i, 4));

    Vector<double> tmp(function_coefficients[i].size());

    SolverControl solver_control(2000, 1e-10 * function_coefficients[i].l2_norm());
    SolverCG<> cg(solver_control);

    cg.solve(*mesh->get_mass_matrix(i), tmp, function_coefficients[i], precondition);
    function_coefficients[i] = tmp;
  }

  deallog << "solved space-time-mass matrices in " << timer.wall_time() << "s" << std::endl;
}

template <int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string filename, std::string name) const {
  write_pvd(path, filename, name, name + "_prime");
}

template <int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string filename, std::string name,
                                         std::string name_deriv) const {
  Assert(mesh, ExcNotInitialized());

  LogStream::Prefix p("write_pvd");
  deallog << "Writing " << path << filename << ".pvd" << std::endl;

  Assert(mesh->length() < 10000, ExcNotImplemented());  // 4 digits are ok
  std::vector<std::pair<double, std::string>> times_and_names(mesh->length(), std::pair<double, std::string>(0.0, ""));

  for (size_t i = 0; i < mesh->length(); i++) {
    const std::string vtuname = filename + "-" + Utilities::int_to_string(i, 4) + ".vtu";
    times_and_names[i]        = std::pair<double, std::string>(mesh->get_time(i), vtuname);

    write_vtu(name, name_deriv, path + vtuname, i);
  }

  std::ofstream pvd_output(path + filename + ".pvd");
  AssertThrow(pvd_output, ExcMessage("write_pvd :: output handle invalid"));

  DataOutBase::write_pvd_record(pvd_output, times_and_names);
  // deallog << "Wrote " << filename << std::endl;
}

template <int dim>
void DiscretizedFunction<dim>::write_vtu(const std::string name, const std::string name_deriv,
                                         const std::string filename, size_t i) const {
  DataOut<dim> data_out;

  data_out.attach_dof_handler(*mesh->get_dof_handler(i));
  data_out.add_data_vector(function_coefficients[i], name);

  if (store_derivative) data_out.add_data_vector(derivative_coefficients[i], name_deriv);

  data_out.build_patches();

  deallog << "Writing " << filename << std::endl;

  std::ofstream output(filename.c_str());
  AssertThrow(output, ExcMessage("write_vtk :: output handle invalid"));

  data_out.write_vtu(output);
  // deallog << "Wrote " << filename << std::endl;
}

template <int dim>
const std::shared_ptr<Norm<DiscretizedFunction<dim>>> DiscretizedFunction<dim>::get_norm() const {
  return norm_;
}

template <int dim>
void DiscretizedFunction<dim>::set_norm(std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm) {
  this->norm_ = norm;
}

template <int dim>
double DiscretizedFunction<dim>::evaluate(const Point<dim>& p, const double time) const {
  const size_t time_idx = mesh->find_time(time);

  return VectorTools::point_value(*mesh->get_dof_handler(time_idx), function_coefficients[time_idx], p);
}

template <int dim>
double DiscretizedFunction<dim>::value(const Point<dim>& p, const unsigned int component) const {
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  Assert(cur_time_idx >= 0 && cur_time_idx < mesh->length(), ExcIndexRange(cur_time_idx, 0, mesh->length()));

  return VectorTools::point_value(*mesh->get_dof_handler(cur_time_idx), function_coefficients[cur_time_idx], p);
}

template <int dim>
Tensor<1, dim, double> DiscretizedFunction<dim>::gradient(const Point<dim>& p, const unsigned int component) const {
  Assert(component == 0, ExcIndexRange(component, 0, 1));
  Assert(cur_time_idx >= 0 && cur_time_idx < mesh->length(), ExcIndexRange(cur_time_idx, 0, mesh->length()));

  return VectorTools::point_gradient(*mesh->get_dof_handler(cur_time_idx), function_coefficients[cur_time_idx], p);
}

template <int dim>
double DiscretizedFunction<dim>::get_time_index() const {
  return cur_time_idx;
}

template <int dim>
void DiscretizedFunction<dim>::set_time(const double new_time) {
  Function<dim>::set_time(new_time);
  cur_time_idx = mesh->find_time(new_time);
}

template <int dim>
std::shared_ptr<SpaceTimeMesh<dim>> DiscretizedFunction<dim>::get_mesh() const {
  return mesh;
}

template <int dim>
void DiscretizedFunction<dim>::min_max_value(double* min_out, double* max_out) const {
  *min_out = std::numeric_limits<double>::infinity();
  *max_out = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < this->length(); i++)
    for (size_t j = 0; j < function_coefficients[i].size(); j++) {
      if (*min_out > function_coefficients[i][j]) *min_out = function_coefficients[i][j];

      if (*max_out < function_coefficients[i][j]) *max_out = function_coefficients[i][j];
    }
}

template <int dim>
double DiscretizedFunction<dim>::min_value() const {
  double tmp = std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < this->length(); i++)
    for (size_t j = 0; j < function_coefficients[i].size(); j++)
      if (tmp > function_coefficients[i][j]) tmp = function_coefficients[i][j];

  return tmp;
}

template <int dim>
double DiscretizedFunction<dim>::max_value() const {
  double tmp = -std::numeric_limits<double>::infinity();

  for (size_t i = 0; i < this->length(); i++)
    for (size_t j = 0; j < function_coefficients[i].size(); j++)
      if (tmp < function_coefficients[i][j]) tmp = function_coefficients[i][j];

  return tmp;
}

template <int dim>
double DiscretizedFunction<dim>::relative_error(const DiscretizedFunction<dim>& other) const {
  DiscretizedFunction<dim> tmp(*this);
  tmp -= other;

  double denom = this->norm();
  return tmp.norm() / (denom == 0.0 ? 1.0 : denom);
}

#ifdef WAVEPI_MPI
template <int dim>
void DiscretizedFunction<dim>::mpi_irecv(size_t source, std::vector<MPI_Request>& reqs) {
  AssertThrow(reqs.size() == 0, ExcInternalError());

  reqs.reserve(function_coefficients.size());

  for (size_t i = 0; i < mesh->length(); i++) {
    reqs.emplace_back();
    MPI_Irecv(&function_coefficients[i][0], function_coefficients[i].size(), MPI_DOUBLE, source, 1, MPI_COMM_WORLD,
              &reqs[i]);
  }

  if (store_derivative) {
    reqs.reserve(function_coefficients.size() + derivative_coefficients.size());

    for (size_t i = 0; i < mesh->length(); i++) {
      reqs.emplace_back();
      MPI_Irecv(&derivative_coefficients[i][0], derivative_coefficients[i].size(), MPI_DOUBLE, source, 1,
                MPI_COMM_WORLD, &reqs[function_coefficients.size() + i]);
    }
  }
}

template <int dim>
void DiscretizedFunction<dim>::mpi_send(size_t destination) {
  for (size_t i = 0; i < mesh->length(); i++)
    MPI_Send(&function_coefficients[i][0], function_coefficients[i].size(), MPI_DOUBLE, destination, 1, MPI_COMM_WORLD);

  if (store_derivative) {
    for (size_t i = 0; i < mesh->length(); i++)
      MPI_Send(&derivative_coefficients[i][0], derivative_coefficients[i].size(), MPI_DOUBLE, destination, 1,
               MPI_COMM_WORLD);
  }
}

template <int dim>
void DiscretizedFunction<dim>::mpi_bcast(size_t root) {
  for (size_t i = 0; i < mesh->length(); i++)
    MPI_Bcast(&function_coefficients[i][0], function_coefficients[i].size(), MPI_DOUBLE, root, MPI_COMM_WORLD);

  if (store_derivative) {
    for (size_t i = 0; i < mesh->length(); i++)
      MPI_Bcast(&derivative_coefficients[i][0], derivative_coefficients[i].size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
  }
}

template <int dim>
void DiscretizedFunction<dim>::mpi_isend(size_t destination, std::vector<MPI_Request>& reqs) {
  AssertThrow(reqs.size() == 0, ExcInternalError());

  reqs.reserve(function_coefficients.size());

  for (size_t i = 0; i < mesh->length(); i++) {
    reqs.emplace_back();
    MPI_Isend(&function_coefficients[i][0], function_coefficients[i].size(), MPI_DOUBLE, destination, 1, MPI_COMM_WORLD,
              &reqs[i]);
  }

  if (store_derivative) {
    reqs.reserve(function_coefficients.size() + derivative_coefficients.size());

    for (size_t i = 0; i < mesh->length(); i++) {
      reqs.emplace_back();
      MPI_Isend(&derivative_coefficients[i][0], derivative_coefficients[i].size(), MPI_DOUBLE, destination, 1,
                MPI_COMM_WORLD, &reqs[function_coefficients.size() + i]);
    }
  }
}

template <int dim>
void DiscretizedFunction<dim>::mpi_all_reduce(DiscretizedFunction<dim> source, MPI_Op op) {
  for (size_t i = 0; i < mesh->length(); i++)
    MPI_Allreduce(&source.function_coefficients[i][0], &function_coefficients[i][0], function_coefficients[i].size(),
                  MPI_DOUBLE, op, MPI_COMM_WORLD);

  if (store_derivative) {
    for (size_t i = 0; i < mesh->length(); i++)
      MPI_Allreduce(&source.derivative_coefficients[i][0], &derivative_coefficients[i][0],
                    derivative_coefficients[i].size(), MPI_DOUBLE, op, MPI_COMM_WORLD);
  }
}
#endif

template class DiscretizedFunction<1>;
template class DiscretizedFunction<2>;
template class DiscretizedFunction<3>;

}  // namespace base
}  // namespace wavepi
