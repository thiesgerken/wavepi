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
      WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh);
      WaveEquationAdjoint(const WaveEquationAdjoint<dim>& weq);
      WaveEquationAdjoint(const WaveEquationBase<dim>& weq);

      ~WaveEquationAdjoint();

      WaveEquationAdjoint<dim>& operator=(const WaveEquationAdjoint<dim>& weq);

      DiscretizedFunction<dim> run();

   private:
      using WaveEquationBase<dim>::mesh;
      using WaveEquationBase<dim>::theta;
      using WaveEquationBase<dim>::right_hand_side;
      using WaveEquationBase<dim>::param_c;
      using WaveEquationBase<dim>::param_nu;
      using WaveEquationBase<dim>::param_a;
      using WaveEquationBase<dim>::param_q;

      // move on one step (overwrite X_old with X)
      void next_step(size_t time_idx);

      // assembling steps of u and v that need to happen on the old mesh
      void assemble_u_pre(size_t time_idx);
      void assemble_v_pre(size_t time_idx);

      // move on to the mesh of the current time step,
      // interpolating system_rhs_[u,v] and tmp_u on the next mesh
      void next_mesh(size_t source_idx, size_t target_idx);

      // assemble matices of the current time step
      void assemble_matrices();

      // final assembly of rhs for u and solving for u
      void assemble_u(size_t time_idx);
      void solve_u();

      // final assembly of rhs for v and solving for v
      void assemble_v(size_t time_idx);
      void solve_v();

      DiscretizedFunction<dim> apply_R_transpose(const DiscretizedFunction<dim>& u);

      std::shared_ptr<SparsityPattern> sparsity_pattern;
      std::shared_ptr<ConstraintMatrix> constraints;

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

      // space for linear systems, their right hand sides
      // and some temporary storage between assemble and pre_assemble.
      SparseMatrix<double> system_matrix;

      Vector<double> system_rhs_u;
      Vector<double> system_rhs_v;

      Vector<double> tmp_u;
      Vector<double> tmp_v;

      std::shared_ptr<DoFHandler<dim>> dof_handler;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
