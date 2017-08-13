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
      /**
       * Default destructor.
       */
      virtual ~DistributionRightHandSide() = default;

      // either of the functions may be zero
      DistributionRightHandSide(Function<dim>* f1, Function<dim>* f2);

      virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q, Vector<double> &rhs) const;

   private:
      struct AssemblyScratchData {
            AssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            AssemblyScratchData(const AssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      struct AssemblyCopyData {
            Vector<double> cell_rhs;
            std::vector<types::global_dof_index> local_dof_indices;
      };

      static void copy_local_to_global(Vector<double> &result, const AssemblyCopyData &copy_data);

      static void local_assemble_dc(const Vector<double> &f1, const Function<dim> * const f2,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);
      static void local_assemble_cd(const Function<dim> * const f1, const Vector<double> &f2,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);
      static void local_assemble_dd(const Vector<double> &f1, const Vector<double> &f2,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);
      static void local_assemble_cc(const Function<dim> * const f1, const Function<dim> * const f2,
            const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);

      Function<dim> *f1;
      Function<dim> *f2;

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DISTRIBUTIONRIGHTHANDSIDE_H_ */
