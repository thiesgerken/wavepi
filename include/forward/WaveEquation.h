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
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <forward/WaveEquationBase.h>

#include <memory>

/**
 * WavePI - Parameter Identification for Wave Equations
 */
namespace wavepi {

/**
 * Tools for the forward problem (solver for wave equations and adjoints, meshes, ...)
 */
namespace forward {

using namespace dealii;
using namespace wavepi::base;

// parameters and rhs must currently be discretized on the same space-time grid!
template <int dim>
class WaveEquation : public WaveEquationBase<dim> {
 public:
  enum Direction { Forward, Backward };

  WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh);
  WaveEquation(const WaveEquation<dim>& weq);
  ~WaveEquation() = default;

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
  using WaveEquationBase<dim>::param_c;
  using WaveEquationBase<dim>::param_nu;
  using WaveEquationBase<dim>::param_a;
  using WaveEquationBase<dim>::param_q;

  // initialize vectors and matrices
  void init_system(size_t first_idx);

  // move on one step (overwrite X_old with X)
  void next_step(double time);

  // assembling steps of u and v that need to happen on the old mesh
  void assemble_pre(double time_step);

  // move on to the mesh of the current time step,
  // interpolating system_rhs_[u,v] and tmp_u on the next mesh
  void next_mesh(size_t source_idx, size_t target_idx);

  // assemble matices of the current time step
  void assemble_matrices();

  // final assembly of rhs for u and solving for u
  void assemble_u(double time_step);
  void solve_u();

  // final assembly of rhs for v and solving for v
  void assemble_v(double time_step);
  void solve_v();

  /**
   * Deinitialize matrices and vectors.
   * This function should be called after computations to have minimal memory requirements when this object is not
   * currently in use.
   */
  void cleanup();

  Direction run_direction = Forward;

  std::shared_ptr<Function<dim>> initial_values_u, initial_values_v;
  std::shared_ptr<Function<dim>> boundary_values_u, boundary_values_v;

  std::shared_ptr<AffineConstraints<double>> constraints;
  std::shared_ptr<SparsityPattern> sparsity_pattern;

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

  // stuff that assemble_pre wants to pass on to assemble_u and assemble_v
  Vector<double> system_tmp1;  // X^n_1
  Vector<double> system_tmp2;  // X^n_2

  // DoFHandler for the current time step
  std::shared_ptr<DoFHandler<dim>> dof_handler;
};
} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
