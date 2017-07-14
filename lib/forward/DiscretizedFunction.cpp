/*
 * DiscretizedFunction.cpp
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#include <deal.II/base/data_out_base.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/utilities.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

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
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler, bool store_derivative)
      : store_derivative(store_derivative), cur_time_idx(0), mesh(mesh), dof_handler(dof_handler) {
   function_coefficients.reserve(mesh->get_times().size());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Vector<double> tmp(dof_handler->n_dofs());

      function_coefficients.push_back(std::move(tmp));
   }

   if (store_derivative) {
      derivative_coefficients.reserve(mesh->get_times().size());

      for (size_t i = 0; i < mesh->get_times().size(); i++) {
         Vector<double> tmp(dof_handler->n_dofs());
         derivative_coefficients.push_back(std::move(tmp));
      }
   }
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler)
      : DiscretizedFunction(mesh, dof_handler, false) {
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
      std::shared_ptr<DoFHandler<dim>> dof_handler, Function<dim>& function)
      : store_derivative(false), cur_time_idx(0), mesh(mesh), dof_handler(dof_handler) {
   function_coefficients.reserve(mesh->get_times().size());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Vector<double> tmp(dof_handler->n_dofs());
      function.set_time(mesh->get_times()[i]);

      VectorTools::interpolate(*dof_handler, function, tmp);
      function_coefficients.push_back(std::move(tmp));
   }
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(DiscretizedFunction&& o)
      : Function<dim>(), norm_type(o.norm_type), store_derivative(o.store_derivative), cur_time_idx(
            o.cur_time_idx), mesh(std::move(o.mesh)), dof_handler(std::move(o.dof_handler)), function_coefficients(
            std::move(o.function_coefficients)), derivative_coefficients(std::move(o.derivative_coefficients)) {
}

template<int dim>
DiscretizedFunction<dim>::DiscretizedFunction(const DiscretizedFunction& o)
      : Function<dim>(), norm_type(o.norm_type), store_derivative(o.store_derivative), cur_time_idx(
            o.cur_time_idx), mesh(o.mesh), dof_handler(std::move(o.dof_handler)), function_coefficients(
            o.function_coefficients), derivative_coefficients(o.derivative_coefficients) {
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(DiscretizedFunction<dim> && o) {
   norm_type = o.norm_type;
   store_derivative = o.store_derivative;
   cur_time_idx = o.cur_time_idx;
   mesh = std::move(o.mesh);
   dof_handler = std::move(o.dof_handler);
   function_coefficients = std::move(o.function_coefficients);
   derivative_coefficients = std::move(o.derivative_coefficients);

   return *this;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(const DiscretizedFunction<dim> & o) {
   norm_type = o.norm_type;
   store_derivative = o.store_derivative;
   cur_time_idx = o.cur_time_idx;
   mesh = o.mesh;
   dof_handler = o.dof_handler;
   function_coefficients = o.function_coefficients;
   derivative_coefficients = o.derivative_coefficients;

   return *this;
}

template<int dim>
DiscretizedFunction<dim> DiscretizedFunction<dim>::derivative() {
   Assert(store_derivative, ExcInternalError());

   DiscretizedFunction<dim> result(mesh, dof_handler, false);
   result.function_coefficients = this->derivative_coefficients;

   return result;
}

template<int dim>
void DiscretizedFunction<dim>::set(size_t i, const Vector<double>& u, const Vector<double>& v) {
   Assert(store_derivative, ExcInternalError());
   Assert(i >= 0 && i < mesh->get_times().size(), ExcIndexRange(i, 0, mesh->get_times().size()));

   function_coefficients[i] = u;
   derivative_coefficients[i] = v;

}

template<int dim>
void DiscretizedFunction<dim>::set(size_t i, const Vector<double>& u) {
   Assert(!store_derivative, ExcInternalError());
   Assert(i >= 0 && i < mesh->get_times().size(), ExcIndexRange(i, 0, mesh->get_times().size()));

   function_coefficients[i] = u;
}

template<int dim>
DiscretizedFunction<dim>& DiscretizedFunction<dim>::operator=(double x) {
   Assert(x == 0, ExcNotImplemented());

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      function_coefficients[i] = 0.0;

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

   std::default_random_engine generator;
   std::uniform_real_distribution<double> distribution(0, 1);

   for (auto coeff : function_coefficients)
      for (size_t i = 0; i < coeff.size(); i++)
         coeff[i] = distribution(generator);
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
   return this->operator *=(1.0 / factor);
}

template<int dim>
void DiscretizedFunction<dim>::pointwise_multiplication(const DiscretizedFunction<dim> & V) {
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
   switch (norm_type) {
      case L2L2_Vector:
         return l2l2_vec_norm();
      case L2L2_Mass:
         return l2l2_mass_norm();
   }

   Assert(false, ExcInternalError ());
   return 0.0;
}

template<int dim>
double DiscretizedFunction<dim>::operator*(const DiscretizedFunction<dim> & V) const {
   switch (norm_type) {
      case L2L2_Vector:
         return l2l2_vec_dot(V);
      case L2L2_Mass:
         return l2l2_mass_dot(V);
   }

   Assert(false, ExcInternalError ());
   return 0.0;
}

template<int dim>
double DiscretizedFunction<dim>::dot(const DiscretizedFunction<dim> & V) const {
   return *this * V;
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_vec_dot(const DiscretizedFunction<dim> & V) const {
   Assert(mesh == V.mesh, ExcInternalError ());
   // remember to sync this implementation with l2_norm and all l2 adjoints!
   // uses trapezoidal rule in time and vector l2 norm in space
   // (only approx to L2(0,T, L2) inner product if spatial grid is uniform!

   double result = 0;

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      Assert(function_coefficients[i].size() == V.function_coefficients[i].size(),
            ExcDimensionMismatch (function_coefficients[i].size() , V.function_coefficients[i].size()));

      double doti = (function_coefficients[i] * V.function_coefficients[i]) / function_coefficients[i].size();

      if (i > 0)
         result += doti / 2 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));

      if (i < mesh->get_times().size() - 1)
         result += doti / 2 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
   }

   return result;
}

template<int dim>
double DiscretizedFunction<dim>::l2l2_vec_norm() const {
   // remember to sync this implementation with l2_dot and all l2 adjoints!
   // uses trapezoidal rule in time and vector l2 norm in space
   // (only approx to L2(0,T, L2) norm if spatial grid is uniform!
   double result = 0;

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      double nrm2 = function_coefficients[i].norm_sqr() / function_coefficients[i].size();

      if (i > 0)
         result += nrm2 / 2 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));

      if (i < mesh->get_times().size() - 1)
         result += nrm2 / 2 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
   }

   return std::sqrt(result);
}
template<int dim>
double DiscretizedFunction<dim>::l2l2_mass_dot(const DiscretizedFunction<dim> & V) const {
   Assert(mesh == V.mesh, ExcInternalError ());
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
         result += doti / 2 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));

      if (i < mesh->get_times().size() - 1)
         result += doti / 2 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
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
   //         result += doti / 3 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));
   //
   //      if (i < mesh->get_times().size() - 1)
   //         result += doti / 3 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
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
   //      result += (dot1 + dot2) / 6 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
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
         result += nrm2 / 2 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));

      if (i < mesh->get_times().size() - 1)
         result += nrm2 / 2 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
   }

   // assume that function is linear in time (consistent with crank-nicolson!)
   // and integrate that exactly (Simpson rule)
   // problem when mesh changes in time!
   //   for (size_t i = 0; i < mesh->get_times().size(); i++) {
   //      double nrm2 = mesh->get_mass_matrix(i)->matrix_norm_square(function_coefficients[i]);
   //
   //      if (i > 0)
   //         result += nrm2 / 3 * (std::abs(mesh->get_times()[i] - mesh->get_times()[i - 1]));
   //
   //      if (i < mesh->get_times().size() - 1)
   //         result += nrm2 / 3 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
   //   }
   //
   //   for (size_t i = 0; i < mesh->get_times().size() - 1; i++) {
   //      double tmp = mesh->get_mass_matrix(i)->matrix_scalar_product(function_coefficients[i],
   //            function_coefficients[i + 1]);
   //
   //      result += tmp / 3 * (std::abs(mesh->get_times()[i + 1] - mesh->get_times()[i]));
   //   }

   return std::sqrt(result);
}

template<int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name) const {
   write_pvd(path, name, name + "_prime");
}

template<int dim>
void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name, std::string name_deriv) const {
   LogStream::Prefix p("DiscFunc");

   Assert(mesh->get_times().size() < 10000, ExcNotImplemented()); // 4 digits are ok
   std::vector<std::pair<double, std::string>> times_and_names;

   Threads::TaskGroup<void> task_group;

   for (size_t i = 0; i < mesh->get_times().size(); i++) {
      const std::string filename = path + "-" + Utilities::int_to_string(i, 4) + ".vtu";
      times_and_names.push_back(std::pair<double, std::string>(mesh->get_times()[i], filename));

      task_group += Threads::new_task(&DiscretizedFunction<dim>::write_vtk, *this, name, name_deriv, filename,
            i);
   }

   task_group.join_all();

   std::ofstream pvd_output(path + ".pvd");
   deallog << "Writing " << path + ".pvd" << std::endl;
   DataOutBase::write_pvd_record(pvd_output, times_and_names);
}

template<int dim>
void DiscretizedFunction<dim>::write_vtk(const std::string name, const std::string name_deriv,
      const std::string filename, size_t i) const {
   DataOut<dim> data_out;

   data_out.attach_dof_handler(*dof_handler);
   data_out.add_data_vector(function_coefficients[i], name);

   if (store_derivative)
      data_out.add_data_vector(derivative_coefficients[i], name_deriv);

   data_out.build_patches();

   deallog << "Writing " << filename << std::endl;
   std::ofstream output(filename.c_str());
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

   return VectorTools::point_value(*dof_handler, function_coefficients[cur_time_idx], p);
}

template<int dim>
Tensor<1, dim, double> DiscretizedFunction<dim>::gradient(const Point<dim> &p,
      const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   Assert(mesh->near_enough(Function<dim>::get_time(), cur_time_idx), ExcNotImplemented());
   Assert(cur_time_idx >= 0 && cur_time_idx < mesh->get_times().size(),
         ExcIndexRange(cur_time_idx, 0, mesh->get_times().size()));

   return VectorTools::point_gradient(*dof_handler, function_coefficients[cur_time_idx], p);
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

} /* namespace forward */
} /* namespace wavepi */

