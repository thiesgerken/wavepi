/*
 * WaveEquation.h
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATION_H_
#define FORWARD_WAVEEQUATION_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <utility>
#include <cmath>
#include <memory>

#include <forward/DiscretizedFunction.h>
#include <forward/MatrixCreator.h>
#include <forward/RightHandSide.h>
#include <forward/L2RightHandSide.h>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must be discretized on the same space-time grid!
template<int dim>
class WaveEquation {
   public:
      WaveEquation(DoFHandler<dim> *dof_handler, const std::vector<double> &times, const Quadrature<dim> quad);
      WaveEquation(const WaveEquation<dim>& weq);
      ~WaveEquation();

      DiscretizedFunction<dim> run();

      std::shared_ptr<Function<dim>> zero = std::make_shared<ZeroFunction<dim>>(1);
      std::shared_ptr<Function<dim>> one = std::make_shared<ConstantFunction<dim>>(1.0, 1);

      std::shared_ptr<RightHandSide<dim>> zero_rhs = std::make_shared<L2RightHandSide<dim>>(zero);

      double get_theta() const;
      void set_theta(double theta);

      const std::vector<double>& get_times() const;
      void set_times(const std::vector<double>& times);

      std::shared_ptr<Function<dim>> get_boundary_values_u() const;
      void set_boundary_values_u(std::shared_ptr<Function<dim>> boundary_values_u);

      std::shared_ptr<Function<dim>> get_boundary_values_v() const;
      void set_boundary_values_v(std::shared_ptr<Function<dim>> boundary_values_v);

      std::shared_ptr<Function<dim>> get_initial_values_u() const;
      void set_initial_values_u(std::shared_ptr<Function<dim>> initial_values_u);

      std::shared_ptr<Function<dim>> get_initial_values_v() const;
      void set_initial_values_v(std::shared_ptr<Function<dim>> initial_values_v);

      std::shared_ptr<Function<dim>> get_param_a() const;
      void set_param_a(std::shared_ptr<Function<dim>> param_a);

      std::shared_ptr<Function<dim>> get_param_c() const;
      void set_param_c(std::shared_ptr<Function<dim>> param_c);

      std::shared_ptr<Function<dim>> get_param_nu() const;
      void set_param_nu(std::shared_ptr<Function<dim>> param_nu);

      std::shared_ptr<Function<dim>> get_param_q() const;
      void set_param_q(std::shared_ptr<Function<dim>> param_q);

      std::shared_ptr<RightHandSide<dim> > get_right_hand_side() const;
      void set_right_hand_side(std::shared_ptr<RightHandSide<dim>> right_hand_side);

   private:
      void init_system();
      void setup_step(double time);
      void assemble_u(double time_step);
      void assemble_v(double time_step);
      void solve_u();
      void solve_v();

      void fill_A();
      void fill_B();
      void fill_C();

      double theta;

      DoFHandler<dim> *dof_handler;
      Quadrature<dim> quad;
      std::vector<double> times;

      std::shared_ptr<Function<dim>> initial_values_u, initial_values_v;
      std::shared_ptr<Function<dim>> boundary_values_u, boundary_values_v;
      std::shared_ptr<Function<dim>> param_c, param_nu, param_a, param_q;

      std::shared_ptr<DiscretizedFunction<dim>> param_c_disc = nullptr, param_nu_disc = nullptr;
      std::shared_ptr<DiscretizedFunction<dim>> param_a_disc = nullptr, param_q_disc = nullptr;

      std::shared_ptr<RightHandSide<dim>> right_hand_side;

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
