/*
 * L2RightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef LIB_L2RIGHTHANDSIDE_H_
#define LIB_L2RIGHTHANDSIDE_H_

#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>

#include <RightHandSide.h>

namespace wavepi {
   using namespace dealii;

   template <int dim>
   class L2RightHandSide: public RightHandSide<dim> {
      public:
         L2RightHandSide(Function<dim>* f);
         virtual ~L2RightHandSide();

         virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler,
                      const Quadrature<dim> &q, Vector<double> &rhs) const;

      private:
         Function<dim> *base_rhs;
   };

} /* namespace wavepi */

#endif /* LIB_L2RIGHTHANDSIDE_H_ */
