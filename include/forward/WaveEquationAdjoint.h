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
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquationBase.h>

#include <stddef.h>
#include <memory>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must currently be discretized on the same space-time grid!
// this is the adjoint equation when using vector norm in time and L2 (using mass matrices) in space
template<int dim>
class WaveEquationAdjoint: public WaveEquationBase<dim> {
   public:
      WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
            std::shared_ptr<DoFHandler<dim>> dof_handler, const Quadrature<dim> quad);
      WaveEquationAdjoint(const WaveEquationAdjoint<dim>& weq);
      WaveEquationAdjoint(const WaveEquationBase<dim>& weq);

      ~WaveEquationAdjoint();

      WaveEquationAdjoint<dim>& operator=(const WaveEquationAdjoint<dim>& weq);

      DiscretizedFunction<dim> run();

   private:
      using WaveEquationBase<dim>::dof_handler;
      using WaveEquationBase<dim>::mesh;
      using WaveEquationBase<dim>::quad;
      using WaveEquationBase<dim>::theta;
      using WaveEquationBase<dim>::right_hand_side;

      void init_system();
      void setup_step(double time);
      void assemble_u(size_t i);
      void assemble_v(size_t i);
      void solve_u();
      void solve_v();

      DiscretizedFunction<dim> apply_R_transpose(const DiscretizedFunction<dim>& u) const;

      ConstraintMatrix constraints;
      SparsityPattern sparsity_pattern;

      // matrices corresponding to the operators A, B, C at the current and the last time step
      SparseMatrix<double> matrix_A;
      SparseMatrix<double> matrix_B;
      SparseMatrix<double> matrix_C;

      SparseMatrix<double> matrix_A_old;
      SparseMatrix<double> matrix_B_old;
      SparseMatrix<double> matrix_C_old;

      // solution and its derivative at the current and the last time step
      Vector<double> solution_u, solution_v;
      Vector<double> solution_u_old, solution_v_old;
      Vector<double> rhs, rhs_old;

      // space for linear systems and their right hand sides
      SparseMatrix<double> system_matrix;
      Vector<double> system_rhs;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
