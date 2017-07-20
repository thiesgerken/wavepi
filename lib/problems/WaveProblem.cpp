/*
 * WaveProblem.cpp
 *
 *  Created on: 04.07.2017
 *      Author: thies
 */

#include <deal.II/base/logstream.h>
#include <problems/WaveProblem.h>
#include <iostream>

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
bool WaveProblem<dim>::progress(InversionProgress<DiscretizedFunction<dim>, DiscretizedFunction<dim>> state) {
   deallog << "i=" << state.iteration_number << ": rdisc=" << state.current_discrepancy / state.norm_data;

   if (state.norm_exact_param > 0.0) {
      deallog << ", rnorm=" << state.norm_current_estimate / state.norm_exact_param << ", rerr="
            << state.current_error / state.norm_exact_param;
   } else
      deallog << ", norm=" << state.norm_current_estimate;

   deallog << std::endl;
   return true;
}

template class WaveProblem<1> ;
template class WaveProblem<2> ;
template class WaveProblem<3> ;

} /* namespace inversion */
} /* namespace wavepi */

