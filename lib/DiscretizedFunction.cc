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
      assert(!store_derivative);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
   }

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         const Vector<double>& function_coeff, const Vector<double>& deriv_coeff) {
      assert(store_derivative);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
      derivative_coefficients.push_back(deriv_coeff);
   }

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         Function<dim>* function) {
      assert(!store_derivative);

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
         : store_derivative(store_derivative), times(), dof_handlers(), function_coefficients(), derivative_coefficients() {
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
      LogStream::Prefix p("DiscFunc(" + name + ")");

      assert(times.size() < 10000); // 4 digits are ok
      std::vector<std::pair<double, std::string>> times_and_names;

      for (size_t i = 0; i < times.size(); i++) {
         DataOut<dim> data_out;

         data_out.attach_dof_handler(*dof_handlers[i]);
         data_out.add_data_vector(function_coefficients[i], name);

         if (store_derivative)
            data_out.add_data_vector(derivative_coefficients[i], name_deriv);

         data_out.build_patches();

         const std::string filename = path + "-" + Utilities::int_to_string(i, 4) + ".vtu";
         std::ofstream output(filename.c_str());

         deallog << "Writing " << filename << std::endl;
         data_out.write_vtu(output);

         times_and_names.push_back(std::pair<double, std::string>(times[i], filename));
      }

      std::ofstream pvd_output(path + ".pvd");
      deallog << "Writing " << path + ".pvd" << std::endl;
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
   }

}
