/*
 * DivRightHandSide.h
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#ifndef FORWARD_DIVRIGHTHANDSIDE_H_
#define FORWARD_DIVRIGHTHANDSIDE_H_

#include <base/DiscretizedFunction.h>
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
using namespace wavepi::base;

/**
 * implements the H^{-1} function f=div(a \nabla u) + bu as a possible right hand side,
 * this means $`<f,v> = (-a\nabla u, \nabla v) + (b u, v)`$ (note the sign)
 */
template <int dim>
class DivRightHandSide : public RightHandSide<dim> {
 public:
  virtual ~DivRightHandSide() = default;

  DivRightHandSide(std::shared_ptr<DiscretizedFunction<dim>> a, std::shared_ptr<DiscretizedFunction<dim>> b, std::shared_ptr<DiscretizedFunction<dim>> u);

  virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
                                      Vector<double> &rhs) const;

  std::shared_ptr<DiscretizedFunction<dim>> get_a() const { return a; }
  void set_a(std::shared_ptr<DiscretizedFunction<dim>> a) { this->a = a; }

  std::shared_ptr<DiscretizedFunction<dim>> get_b() const { return b; }
  void set_b(std::shared_ptr<DiscretizedFunction<dim>> b) { this->b = b; }

  std::shared_ptr<DiscretizedFunction<dim>> get_u() const { return u; }
  void set_u(std::shared_ptr<DiscretizedFunction<dim>> u) { this->u = u; }

 private:
  std::shared_ptr<DiscretizedFunction<dim>> a;
  std::shared_ptr<DiscretizedFunction<dim>> b;
  std::shared_ptr<DiscretizedFunction<dim>> u;

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

  void local_assemble(const Vector<double> &a, const Vector<double> &b, const Vector<double> &u,
                         const typename DoFHandler<dim>::active_cell_iterator &cell, AssemblyScratchData &scratch_data,
                         AssemblyCopyData &copy_data);

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_DIVRIGHTHANDSIDE_H_ */
