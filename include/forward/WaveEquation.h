/*
 * WaveEquation.h
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATION_H_
#define FORWARD_WAVEEQUATION_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquationBase.h>

#include <memory>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must currently be discretized on the same space-time grid!
template<int dim>
class WaveEquation: public WaveEquationBase<dim> {
   public:
      enum Direction {
         Forward, Backward
      };

      WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh);
      WaveEquation(const WaveEquation<dim>& weq);
      ~WaveEquation();

      WaveEquation<dim>& operator=(const WaveEquation<dim>& weq);

      DiscretizedFunction<dim> run();

      std::shared_ptr<Function<dim>> get_boundary_values_u() const;
      void set_boundary_values_u(std::shared_ptr<Function<dim>> boundary_values_u);

      std::shared_ptr<Function<dim>> get_boundary_values_v() const;
      void set_boundary_values_v(std::shared_ptr<Function<dim>> boundary_values_v);

      std::shared_ptr<Function<dim>> get_initial_values_u() const;
      void set_initial_values_u(std::shared_ptr<Function<dim>> initial_values_u);

      std::shared_ptr<Function<dim>> get_initial_values_v() const;
      void set_initial_values_v(std::shared_ptr<Function<dim>> initial_values_v);

      typename WaveEquation<dim>::Direction get_run_direction() const;
      void set_run_direction(typename WaveEquation<dim>::Direction run_direction);

   private:
      using WaveEquationBase<dim>::mesh;
      using WaveEquationBase<dim>::theta;
      using WaveEquationBase<dim>::right_hand_side;

      void init_system(double initial_time);
      void setup_step(double time);
      void assemble_u(double time_step);
      void assemble_v(double time_step);
      void solve_u();
      void solve_v();

      Direction run_direction = Forward;

      std::shared_ptr<Function<dim>> initial_values_u, initial_values_v;
      std::shared_ptr<Function<dim>> boundary_values_u, boundary_values_v;

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
