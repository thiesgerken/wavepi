/*
 * RightHandSide.cpp
 *
 *  Created on: 29.06.2017
 *      Author: thies
 */

#include <forward/RightHandSide.h>

namespace wavepi {
namespace forward {

template<int dim>
RightHandSide<dim>::RightHandSide() {
}

template<int dim>
RightHandSide<dim>::~RightHandSide() {
}

template class RightHandSide<1> ;
template class RightHandSide<2> ;
template class RightHandSide<3> ;

} /* namespace forward */
} /* namespace wavepi */
