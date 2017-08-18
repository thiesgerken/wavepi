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
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

#include <forward/RightHandSide.h>

#include <memory>
#include <vector>

namespace wavepi {
namespace forward {
using namespace dealii;

// implements the H^{-1} function f=div(a \nabla u) as a possible right hand side,
// this means <f,v> = (-a\nabla u, \nabla v) [note the sign!]
template<int dim>
class DivRightHandSide: public RightHandSide<dim> {
   public:

      virtual ~DivRightHandSide() = default;

      // optimization is used only when a _and_ u are discretized
      // if u is continuous, then u has to have an implementation of gradient
      DivRightHandSide(std::shared_ptr<Function<dim>> a, std::shared_ptr<Function<dim>> u);

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

      struct AssemblyScratchData {
            AssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            AssemblyScratchData(const AssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      struct AssemblyCopyData {
            Vector<double> cell_rhs;
            std::vector<types::global_dof_index> local_dof_indices;
      };

      void copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data);

      void local_assemble_dd(const Vector<double> &a, const Vector<double> &u,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);

      void local_assemble_cc(const Function<dim> * const a, const Function<dim> * const u,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
            AssemblyCopyData &copy_data);
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DIVRIGHTHANDSIDE_H_ */
