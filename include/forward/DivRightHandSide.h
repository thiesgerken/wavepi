/*
 * DivRightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DIVRIGHTHANDSIDE_H_
#define FORWARD_DIVRIGHTHANDSIDE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>

#include <forward/RightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;

// implements the H^{-1} function f=div(a \nabla u) as a possible right hand side,
// this means <f,v> = (-a\nabla u, \nabla v) [note the sign!]
template<int dim>
class DivRightHandSide: public RightHandSide<dim> {
   public:

      // optimization is used only when a _and_ u are discretized
      // if u is continuous, then u has to have an implementation of gradient
      DivRightHandSide(std::shared_ptr<Function<dim>> a, std::shared_ptr<Function<dim>> u);

      virtual ~DivRightHandSide();

      virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
            Vector<double> &rhs) const;

      inline std::shared_ptr<Function<dim> > get_a() const {
         return a;
      }

      inline void set_a(std::shared_ptr<Function<dim> > a) {
         this->a = a;
      }

      inline std::shared_ptr<Function<dim> > get_u() const {
         return u;
      }

      inline void set_u(std::shared_ptr<Function<dim> > u) {
         this->u = u;
      }

   private:
      std::shared_ptr<Function<dim>> a;
      std::shared_ptr<Function<dim>> u;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DIVRIGHTHANDSIDE_H_ */
