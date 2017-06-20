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
      assert(!has_derivative);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
   }

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,
         const Vector<double>& function_coeff, const Vector<double>& deriv_coeff) {
      assert(has_derivative);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
      derivative_coefficients.push_back(deriv_coeff);
   }

   template<int dim>
   void DiscretizedFunction<dim>::push_back(DoFHandler<dim>* dof_handler, double time,      Function<dim>* function) {
      assert(!has_derivative);

      function->set_time(time);
      Vector<double> function_coeff(dof_handler->n_dofs());
      VectorTools::interpolate(*dof_handler, *function, function_coeff);

      dof_handlers.push_back(dof_handler);
      times.push_back(time);
      function_coefficients.push_back(function_coeff);
   }

   template<int dim>
   DiscretizedFunction<dim>::~DiscretizedFunction() {
   }

   template<int dim>
   DiscretizedFunction<dim>::DiscretizedFunction(bool store_derivative, int capacity)
         : has_derivative(store_derivative), times(), dof_handlers(), function_coefficients(), derivative_coefficients() {
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
   void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name) const {
      write_pvd(path, name, name + "_prime");
   }

   template<int dim>
   void DiscretizedFunction<dim>::write_pvd(std::string path, std::string name,
         std::string name_deriv) const {

      assert(times.size() < 10000); // 4 digits are ok
      std::vector<std::pair<double, std::string>> times_and_names;

      for (size_t i = 0; i < times.size(); i++) {
         DataOut<dim> data_out;

         data_out.attach_dof_handler(*dof_handlers[i]);
         data_out.add_data_vector(function_coefficients[i], name);

         if (has_derivative)
            data_out.add_data_vector(derivative_coefficients[i], name_deriv);

         data_out.build_patches();

         const std::string filename = path + "-" + Utilities::int_to_string(i, 4) + ".vtu";
         std::ofstream output(filename.c_str());

         std::cout << "Writing " << filename << std::endl;
         data_out.write_vtu(output);

         times_and_names.push_back(std::pair<double, std::string>(times[i], filename));
      }

      std::ofstream pvd_output(path + ".pvd");
      std::cout << "Writing " << path + ".pvd" << std::endl;
      DataOutBase::write_pvd_record(pvd_output, times_and_names);
   }

}
