/*
 * VectorRightHandSide.h
 *
 *  Created on: 27.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_FORWARD_VECTORRIGHTHANDSIDE_H_
#define INCLUDE_FORWARD_VECTORRIGHTHANDSIDE_H_

#include <base/DiscretizedFunction.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <forward/RightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;
using namespace wavepi::base;

template <int dim>
class VectorRightHandSide : public RightHandSide<dim> {
 public:
  VectorRightHandSide(const std::shared_ptr<DiscretizedFunction<dim>> base);

  virtual ~VectorRightHandSide() = default;

  virtual void create_right_hand_side(const DoFHandler<dim> &dof_handler, const Quadrature<dim> &q,
                                      Vector<double> &rhs) const;

 private:
  std::shared_ptr<DiscretizedFunction<dim>> base;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_VECTORRIGHTHANDSIDE_H_ */
