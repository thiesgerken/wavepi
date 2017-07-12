/*
 * DistributionRightHandSide.h
 *
 *  Created on: 30.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DISTRIBUTIONRIGHTHANDSIDE_H_
#define FORWARD_DISTRIBUTIONRIGHTHANDSIDE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <forward/RightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;

// Element of the dual space of H^1_0, represented by L^2 scalar products (f1, v) + (f2, nabla v)
template<int dim>
class DistributionRightHandSide: public RightHandSide<dim> {
   public:

      // either of the functions may be zero
      DistributionRightHandSide(Function<dim>* f1, Function<dim>* f2);
      virtual ~DistributionRightHandSide();

      virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
            Vector<double> &rhs) const;

   private:
      Function<dim> *f1;
      Function<dim> *f2;

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DISTRIBUTIONRIGHTHANDSIDE_H_ */
