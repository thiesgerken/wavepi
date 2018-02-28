/*
 * VectorRightHandSide.cpp
 *
 *  Created on: 27.02.2018
 *      Author: thies
 */

#include <forward/VectorRightHandSide.h>

namespace wavepi {
namespace forward {

template <int dim>
VectorRightHandSide<dim>::VectorRightHandSide(const std::shared_ptr<DiscretizedFunction<dim>> base) : base(base) {
  AssertThrow(base, ExcNotInitialized());
}

template <int dim>
void VectorRightHandSide<dim>::create_right_hand_side(const DoFHandler<dim> &dof_handler __attribute((unused)),
                                                      const Quadrature<dim> &q __attribute((unused)),
                                                      Vector<double> &rhs) const {
  size_t ti = base->get_mesh()->find_time(this->get_time());
  rhs       = (*base)[ti];
}

template class VectorRightHandSide<1>;
template class VectorRightHandSide<2>;
template class VectorRightHandSide<3>;

} /* namespace forward */
} /* namespace wavepi */
