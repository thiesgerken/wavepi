/*
 * L2RightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef FORWARD_L2PRODUCTRIGHTHANDSIDE_H_
#define FORWARD_L2PRODUCTRIGHTHANDSIDE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/lac/vector.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>

#include <functional>
#include <memory>

#include <base/DiscretizedFunction.h>
#include <forward/RightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

/**
 * Implements $`(-f_1*f_2, \phi_j)`$ for two discretized functions $`f_1`$ and $`f_2`$ as a right hand side.
 *
 * Please note the sign, and also be aware that the difference between this and just multiplying nodal values of $`f_1`$
 * and $`f_2`$ is very big. Do not use one for the forward problem and the other for its adjoint, or the linear
 * subproblems will diverge!
 */
template <int dim>
class L2ProductRightHandSide : public RightHandSide<dim> {
 public:
  virtual ~L2ProductRightHandSide() = default;

  L2ProductRightHandSide(std::shared_ptr<DiscretizedFunction<dim>> f1, std::shared_ptr<DiscretizedFunction<dim>> f2);

  virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
                                      Vector<double> &rhs) const;

  std::shared_ptr<DiscretizedFunction<dim>> get_func1() const;
  void set_func1(std::shared_ptr<DiscretizedFunction<dim>> func1);

  std::shared_ptr<DiscretizedFunction<dim>> get_func2() const;
  void set_func2(std::shared_ptr<DiscretizedFunction<dim>> func2);

 private:
  std::shared_ptr<DiscretizedFunction<dim>> func1;
  std::shared_ptr<DiscretizedFunction<dim>> func2;

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
  void local_assemble(const Vector<double> &f1, const Vector<double> &f2,
                      const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
                      AssemblyCopyData &copy_data);
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_L2RIGHTHANDSIDE_H_ */
