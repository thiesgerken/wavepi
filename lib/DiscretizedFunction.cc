/*
 * DiscretizedFunction.cpp
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#include "DiscretizedFunction.h"

using namespace dealii;

namespace wavepi {

   template class DiscretizedFunction<1> ;
   template class DiscretizedFunction<2> ;
   template class DiscretizedFunction<3> ;

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         const Vector<double>& function_coeff) {
      Assert(!store_derivative, ExcInvalidState());

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
   }

   // times have to be inserted in order (increasing or decreasing)!
   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         const Vector<double>& function_coeff, const Vector<double>& deriv_coeff) {
      Assert(store_derivative, ExcInvalidState());

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
      derivative_coefficients.push_back(deriv_coeff);
   }

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         Function<dim>* function) {
      Assert(!store_derivative, ExcInvalidState());

      function->set_time(time);
      Vector<double> function_coeff(dof_handler->n_dofs());
      VectorTools::interpolate(*dof_handler, *function, function_coeff);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
   }

   template<int dim>
   inline const std::vector<Vector<double> >& DiscretizedFunction<dim>::get_derivative_coefficients() const {
      return derivative_coefficients;
   }

   template<int dim>
   inline const std::vector<DoFHandler<dim> *>& DiscretizedFunction<dim>::get_dof_handlers() const {
      return dof_handlers;
   }

   template<int dim>
   inline const std::vector<Vector<double> >& DiscretizedFunction<dim>::get_function_coefficients() const {
      return function_coefficients;
   }

   template<int dim>
   inline bool DiscretizedFunction<dim>::has_derivative() const {
      return store_derivative;
   }

   template<int dim>
   inline const std::vector<double>& DiscretizedFunction<dim>::get_times() const {
      return times;
   }

   template<int dim>
   DiscretizedFunction<dim>::~DiscretizedFunction() {
   }

   template<int dim>
   DiscretizedFunction<dim>::DiscretizedFunction(bool store_derivative, int capacity)
         : store_derivative(store_derivative), cur_time_idx(0), times(), dof_handlers(), function_coefficients(), derivative_coefficients() {
      times.reserve(capacity);
      dof_handlers.reserve(capacity);
      function_coefficients.reserve(capacity);
      derivative_coefficients.reserve(store_derivative ? capacity : 1);
   }

   template<int dim>
   DiscretizedFunction<dim>::DiscretizedFunction(bool store_derivative)
         : DiscretizedFunction(store_derivative, 20) {
   }

   template<int dim>
   DiscretizedFunction<dim>::DiscretizedFunction()
         : DiscretizedFunction(false) {
   }

   template<int dim>
   DiscretizedFunction<dim> DiscretizedFunction<dim>::discretize(Function<dim>* function,
         const std::vector<double>& times, const std::vector<DoFHandler<dim>*>& handlers) {
      DiscretizedFunction<dim> fdisc(false, times.size());

      for (size_t i = 0; i < times.size(); i++)
         fdisc.push_back(handlers[i], times[i], function);

      return fdisc;
   }

   template<int dim>
   void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name) const {
      write_pvd(path, name, name + "_prime");
   }

   template<int dim>
   void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name,
         std::string name_deriv) const {
      LogStream::Prefix p("DiscFunc");

      Assert(times.size() < 10000, ExcNotImplemented()); // 4 digits are ok
      std::vector<std::pair<double, std::string>> times_and_names;

      Threads::TaskGroup<void> task_group;

      for (size_t i = 0; i < times.size(); i++) {
         const std::string filename = path + "-" + Utilities::int_to_string(i, 4) + ".vtu";
         times_and_names.push_back(std::pair<double, std::string>(times[i], filename));

         task_group += Threads::new_task(&DiscretizedFunction<dim>::write_vtk, *this, name,
               name_deriv, filename, i);
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

      data_out.attach_dof_handler(*dof_handlers[i]);
      data_out.add_data_vector(function_coefficients[i], name);

      if (store_derivative)
         data_out.add_data_vector(derivative_coefficients[i], name_deriv);

      data_out.build_patches();

      deallog << "Writing " << filename << std::endl;
      std::ofstream output(filename.c_str());
      data_out.write_vtu(output);
   }

   // tries to find a given time in the times vector (using a binary search)
   // returns the index of the nearest time, the caller has to decide whether it is good enough.
   // must not be called on a empty discretization!
   template<int dim>
   size_t DiscretizedFunction<dim>::find_time(double time, size_t low, size_t up,
         bool increasing) const {
      Assert(low <= up, ExcInternalError()); // something went wrong

      if (low >= up) // low == up or sth went wrong
         return low;

      if (low + 1 == up) {
         if (std::abs(times[low] - time) <= std::abs(times[up] - time))
            return low;
         else
            return up;
      }

      size_t middle = (low + up) / 2;
      double val = times[middle];

      if (time > val)
         if (increasing)
            return find_time(time, middle, up, increasing);
         else
            return find_time(time, low, middle, increasing);
      else if (time < val)
         if (increasing)
            return find_time(time, low, middle, increasing);
         else
            return find_time(time, middle, up, increasing);
      else
         return middle;
   }

   template<int dim>
   size_t DiscretizedFunction<dim>::find_nearest_time(double time) const {
      Assert(times.size() > 0, ExcEmptyObject());

      if (times.size() == 1)
         return 0;
      else
         return find_time(time, 0, times.size() - 1, times[1] - times[0] > 0);
   }

   template<int dim>
   bool DiscretizedFunction<dim>::near_enough(double time, size_t idx) const {
      Assert(idx >= 0 && idx < times.size(), ExcIndexRange(idx, 0, times.size()));

      if (times.size() == 1)
         return std::abs(times[idx] - time) < 1e-3;
      else if (idx > 0)
         return std::abs(times[idx] - time) < 1e-3 * std::abs(times[idx] - times[idx - 1]);
      else
         return std::abs(times[idx] - time) < 1e-3 * std::abs(times[idx + 1] - times[idx]);
   }

   template<int dim>
   size_t DiscretizedFunction<dim>::find_time(double time) const {
      size_t idx = find_nearest_time(time);

      if (!near_enough(time, idx)) {
         std::stringstream err;
         err << "requested time " << time << " not found, nearest is " << times[idx];
         Assert(false, ExcMessage(err.str()));
      }

      return idx;
   }

   template<int dim>
   void DiscretizedFunction<dim>::at(double time, const Vector<double>* &coeffs,
         const Vector<double>* &deriv_coeffs, DoFHandler<dim>* &handler) const {
      Assert(store_derivative, ExcInvalidState());
      size_t idx = find_time(time); // interpolation not implemented

      coeffs = &function_coefficients[idx];
      deriv_coeffs = &derivative_coefficients[idx];
      handler = dof_handlers[idx];
   }

   template<int dim>
   void DiscretizedFunction<dim>::at(double time, const Vector<double>* &coeffs,
         DoFHandler<dim>* &handler) const {
      size_t idx = find_time(time); // interpolation not implemented

      coeffs = &function_coefficients[idx];
      handler = dof_handlers[idx];
   }

   template<int dim>
   void DiscretizedFunction<dim>::at(double time, const Vector<double>* &coeffs) const {
      size_t idx = find_time(time); // interpolation not implemented

      coeffs = &function_coefficients[idx];
   }

   template<int dim>
   double DiscretizedFunction<dim>::value(const Point<dim> &p, const unsigned int component) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));
      Assert(near_enough(Function<dim>::get_time(), cur_time_idx), ExcNotImplemented());
      Assert(cur_time_idx >= 0 && cur_time_idx < times.size(),
            ExcIndexRange(cur_time_idx, 0, times.size()));

      return VectorTools::point_value(*dof_handlers[cur_time_idx],
            function_coefficients[cur_time_idx], p);
   }

   template<int dim>
   double DiscretizedFunction<dim>::get_time_index() const {
      return cur_time_idx;
   }

   template<int dim>
   void DiscretizedFunction<dim>::set_time(const double new_time) {
      Function<dim>::set_time(new_time);
      cur_time_idx = find_nearest_time(new_time);
   }

}
