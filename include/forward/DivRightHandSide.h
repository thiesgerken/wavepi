/*
 * DivRightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DIVRIGHTHANDSIDE_H_
#define FORWARD_DIVRIGHTHANDSIDE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/lac/vector.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>

#include <forward/DiscretizedFunction.h>
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
      DivRightHandSide(Function<dim>* a, Function<dim>* u);
      virtual ~DivRightHandSide();

      virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
            Vector<double> &rhs) const;

   private:
      Function<dim> *a;
      Function<dim> *u;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DIVRIGHTHANDSIDE_H_ */
