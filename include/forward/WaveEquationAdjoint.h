/*
 * WaveEquationAdjoint.h
 *
 *  Created on: 17.0.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATIONADJOINT_H_
#define FORWARD_WAVEEQUATIONADJOINT_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <forward/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>
#include <forward/RightHandSide.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>

#include <cmath>
#include <memory>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must currently be discretized on the same space-time grid!
template<int dim>
class WaveEquationAdjoint {
   public:
      WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh,
            std::shared_ptr<DoFHandler<dim>> dof_handler, const Quadrature<dim> quad);
      WaveEquationAdjoint(const WaveEquationAdjoint<dim>& weq);
      WaveEquationAdjoint(const WaveEquation<dim>& weq);

      ~WaveEquationAdjoint();

      WaveEquationAdjoint<dim>& operator=(const WaveEquationAdjoint<dim>& weq);

      DiscretizedFunction<dim> run();

      std::shared_ptr<Function<dim>> zero = std::make_shared<ZeroFunction<dim>>(1);
      std::shared_ptr<Function<dim>> one = std::make_shared<ConstantFunction<dim>>(1.0, 1);

      std::shared_ptr<RightHandSide<dim>> zero_rhs = std::make_shared<L2RightHandSide<dim>>(zero);

      double get_theta() const;
      void set_theta(double theta);

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

      int get_special_assembly_tactic() const;
      void set_special_assembly_tactic(int special_assembly_tactic);

      // uses special functions for matrix assembly when discretized parameters are passed, which is a lot better for P1 elements.
      // For P2 elements and 3 dimensions it actually turns out to be worse
      // (too much coupling going on, evaluating the polynomial is actually cheaper)
      // in that case, you should turn of the specialization.
      inline bool is_special_assembly_recommended() const {
         return !(quad.size() >= std::pow(2, dim) && dim >= 3);
      }

      inline bool using_special_assembly() {
         return
               special_assembly_tactic == 0 ?
                     is_special_assembly_recommended() : (special_assembly_tactic > 0);
      }

      const Quadrature<dim> get_quad() const;
      void set_quad(const Quadrature<dim> quad);
      const std::shared_ptr<DoFHandler<dim> > get_dof_handler() const;
      void set_dof_handler(const std::shared_ptr<DoFHandler<dim> > dof_handler);
      const std::shared_ptr<SpaceTimeMesh<dim> > get_mesh() const;
      void set_mesh(const std::shared_ptr<SpaceTimeMesh<dim> > mesh);

   private:
      void init_system();
      void setup_step(double time);
      void assemble_u(size_t i);
      void assemble_v(size_t i);
      void solve_u(size_t i);
      void solve_v(size_t i);

      void fill_A();
      void fill_B();
      void fill_C();

      DiscretizedFunction<dim> apply_R_transpose(const DiscretizedFunction<dim>& u) const;

      double theta;

      // treat DiscretizedFunctions as params and right hand side differently
      // < 0 -> no (better if much coupling present), > 0 -> yes, = 0 automatically (default)
      int special_assembly_tactic = 0;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<DoFHandler<dim>> dof_handler;
      Quadrature<dim> quad;

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
