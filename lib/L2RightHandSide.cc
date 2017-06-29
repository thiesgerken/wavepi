/*
 * L2RightHandSide.cc
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <L2RightHandSide.h>

namespace wavepi {
   using namespace dealii;

   template<int dim>
   L2RightHandSide<dim>::L2RightHandSide(Function<dim>* f)
         : base_rhs(f) {
   }

   template<int dim>
   L2RightHandSide<dim>::~L2RightHandSide() {
   }

   template<int dim>
   void L2RightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof_handler,
         const Quadrature<dim> &q, Vector<double> &rhs) const {
      base_rhs->set_time(this->get_time());

      VectorTools::create_right_hand_side(dof_handler, q, *base_rhs, rhs);
   }

   template class L2RightHandSide<1> ;
   template class L2RightHandSide<2> ;
   template class L2RightHandSide<3> ;

} /* namespace wavepi */
