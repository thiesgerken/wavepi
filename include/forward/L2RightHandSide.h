/*
 * L2RightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef FORWARD_L2RIGHTHANDSIDE_H_
#define FORWARD_L2RIGHTHANDSIDE_H_

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

template <int dim>
class L2RightHandSide : public RightHandSide<dim> {
 public:
  virtual ~L2RightHandSide() = default;

  L2RightHandSide(std::shared_ptr<Function<dim>> f);

  virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
                                      Vector<double> &rhs) const;

  std::shared_ptr<Function<dim>> get_base_rhs() const;
  void set_base_rhs(std::shared_ptr<Function<dim>> base_rhs);

 private:
  std::shared_ptr<Function<dim>> base_rhs;

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
  void local_assemble(const Vector<double> &f, const typename DoFHandler<dim>::active_cell_iterator &cell,
                      AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_L2RIGHTHANDSIDE_H_ */
