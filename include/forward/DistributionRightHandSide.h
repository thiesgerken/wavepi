/*
 * DistributionRightHandSide.h
 *
 *  Created on: 30.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DISTRIBUTIONRIGHTHANDSIDE_H_
#define FORWARD_DISTRIBUTIONRIGHTHANDSIDE_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

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
