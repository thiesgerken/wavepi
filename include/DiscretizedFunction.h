/*
 * DiscretizedFunction.h
 *
 *  Created on: 16.06.2017
 *      Author: thies
 */

#ifndef DISCRETIZEDFUNCTION_H_
#define DISCRETIZEDFUNCTION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <utility>
#include <fstream>
#include <iostream>

namespace wavepi {
   using namespace dealii;

   template<int dim>
   class DiscretizedFunction {
      public:
         bool has_derivative;

         std::vector<double> times;
         std::vector<DoFHandler<dim>*> dof_handlers;
         std::vector<Vector<double>> function_coefficients;
         std::vector<Vector<double>> derivative_coefficients;

         DiscretizedFunction(bool store_derivative, int capacity);
         DiscretizedFunction(bool store_derivative);
         DiscretizedFunction();

         void push_back(DoFHandler<dim>* dof_handler, double time,
               const Vector<double>& function_coeff);
         void push_back(DoFHandler<dim>* dof_handler, double time,
               const Vector<double>& function_coeff, const Vector<double>& deriv_coeff);
         void push_back(DoFHandler<dim>* dof_handler, double time,
               Function<dim>* function);

         void write_pvd(std::string path, std::string name, std::string name_deriv) const;
         void write_pvd(std::string path, std::string name) const;

         virtual ~DiscretizedFunction();
      private:

   };
}

#endif /* DISCRETIZEDFUNCTION_H_ */
