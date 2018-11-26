/*
 * WaveEquationAdjoint.h
 *
 *  Created on: 17.0.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATIONADJOINT_H_
#define FORWARD_WAVEEQUATIONADJOINT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <forward/WaveEquationBase.h>
#include <forward/WaveEquation.h>
#include <forward/AbstractEquationAdjoint.h>

#include <stddef.h>
#include <memory>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must currently be discretized on the same space-time grid!
// this is the adjoint equation when using vector norm in time and space
template<int dim>
class WaveEquationAdjoint: public AbstractEquationAdjoint<dim>, public WaveEquationBase<dim> {
public:
   WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

   WaveEquationAdjoint(const WaveEquation<dim> &wave);

   virtual ~WaveEquationAdjoint() = default;

private:
   virtual void apply_boundary_conditions_u(double time);
   virtual void apply_boundary_conditions_v(double time);
   virtual void assemble_matrices(size_t time_idx);

   using AbstractEquationAdjoint<dim>::mesh;
   using AbstractEquationAdjoint<dim>::dof_handler;
   using AbstractEquationAdjoint<dim>::constraints;
   using AbstractEquationAdjoint<dim>::system_matrix;
   using AbstractEquationAdjoint<dim>::system_rhs_u;
   using AbstractEquationAdjoint<dim>::system_rhs_v;
   using AbstractEquationAdjoint<dim>::solution_u;
   using AbstractEquationAdjoint<dim>::solution_v;
   using AbstractEquationAdjoint<dim>::matrix_A;
   using AbstractEquationAdjoint<dim>::matrix_B;
   using AbstractEquationAdjoint<dim>::matrix_C;

   using WaveEquationBase<dim>::param_c;
   using WaveEquationBase<dim>::param_nu;
   using WaveEquationBase<dim>::param_rho;
   using WaveEquationBase<dim>::param_q;

   std::shared_ptr<Function<dim>> zero;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
