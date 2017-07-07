/*
 * WaveProblem.cpp
 *
 *  Created on: 04.07.2017
 *      Author: thies
 */

#include <inversion/WaveProblem.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

template<int dim>
inline WaveEquation<dim>& WaveProblem<dim>::get_wave_equation() {
   return wave_equation;
}

template<int dim>
WaveProblem<dim>::~WaveProblem() {
}

template<int dim>
WaveProblem<dim>::WaveProblem(WaveEquation<dim>& weq)
      : wave_equation(weq) {
}

template<int dim>
void WaveProblem<dim>::progress(const DiscretizedFunction<dim>& current_estimate,
      const DiscretizedFunction<dim>& current_residual, const DiscretizedFunction<dim>& data, int iteration_number,
      const DiscretizedFunction<dim>* exact_param) {
   deallog << "i=" << std::setw(3) << iteration_number << ": rdisc=" << current_residual.l2_norm() / data.l2_norm();

   if (exact_param != nullptr) {
      DiscretizedFunction<dim> tmp(current_estimate);
      tmp -= *exact_param;
      deallog << ", rnorm=" << current_estimate.norm() / exact_param->norm() << ", rerr="
            << tmp.l2_norm() / exact_param->l2_norm();
   } else
      deallog << ", norm=" << current_estimate.norm();

   deallog << std::endl;
}

template class WaveProblem<1> ;
template class WaveProblem<2> ;
template class WaveProblem<3> ;

} /* namespace inversion */
} /* namespace wavepi */

