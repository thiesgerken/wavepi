/*
 * RightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef LIB_RIGHTHANDSIDE_H_
#define LIB_RIGHTHANDSIDE_H_

#include <deal.II/base/function_time.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

namespace wavepi {
   using namespace dealii;

   template<int dim>
   class RightHandSide: public FunctionTime<double> {
      public:
         RightHandSide();
         virtual ~RightHandSide();

         virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler,
               const Quadrature<dim> &q, Vector<double> &rhs) const = 0;

   };

} /* namespace wavepi */

#endif /* LIB_RIGHTHANDSIDE_H_ */
