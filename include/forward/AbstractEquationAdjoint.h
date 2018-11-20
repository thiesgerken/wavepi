/*
 * AbstractEquationAdjoint.h
 *
 *  Created on: 16.11.2018
 *      Author: thies
 */

#ifndef FORWARD_ABSTRACTEQUATIONADJOINT_H_
#define FORWARD_ABSTRACTEQUATIONADJOINT_H_

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <forward/RightHandSide.h>
#include <stddef.h>
#include <memory>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template<int dim>
class AbstractEquationAdjoint {
public:

   AbstractEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
         : mesh(mesh) {
   }

   virtual ~AbstractEquationAdjoint() = default;

   static void declare_parameters(ParameterHandler &prm);
   void get_parameters(ParameterHandler &prm);

   virtual DiscretizedFunction<dim> run(std::shared_ptr<RightHandSide<dim>> right_hand_side);

   inline double get_theta() const {
      return theta;
   }
   inline void set_theta(double theta) {
      this->theta = theta;
   }

   inline std::shared_ptr<SpaceTimeMesh<dim>> get_mesh() const {
      return this->mesh;
   }
   inline void set_mesh(std::shared_ptr<SpaceTimeMesh<dim>> mesh) {
      this->mesh = mesh;
   }

   inline double get_solver_tolerance() const {
      return solver_tolerance;
   }
   inline void set_solver_tolerance(double solver_tolerance) {
      this->solver_tolerance = solver_tolerance;
   }

   inline int get_solver_max_iter() const {
      return solver_max_iter;
   }
   inline void set_solver_max_iter(int solver_max_iter) {
      this->solver_max_iter = solver_max_iter;
   }

protected:
   double theta = 0.5;
   double solver_tolerance = 1e-8;
   int solver_max_iter = 10000;

   std::shared_ptr<SpaceTimeMesh<dim>> mesh;

   std::shared_ptr<AffineConstraints<double>> constraints;
   std::shared_ptr<SparsityPattern> sparsity_pattern;

   // matrices corresponding to the operators A, B, D^-1 C at the current time step
   SparseMatrix<double> matrix_A;
   SparseMatrix<double> matrix_B;
   SparseMatrix<double> matrix_C;

   // TODO: do these actually have to be saved at all?
   SparseMatrix<double> matrix_A_old;
   SparseMatrix<double> matrix_B_old;
   SparseMatrix<double> matrix_C_old;

   // solution and its derivative at the current and the last time step
   Vector<double> solution_u, solution_v;
   Vector<double> solution_u_old, solution_v_old;
   Vector<double> rhs;

   // handle to the right hand side used
   std::shared_ptr<RightHandSide<dim>> right_hand_side;

   // space for linear systems and their right hand sides
   SparseMatrix<double> system_matrix;
   Vector<double> system_rhs;

   Vector<double> system_rhs_u;
   Vector<double> system_rhs_v;

   Vector<double> tmp_u;
   Vector<double> tmp_v;

   // DoFHandler for the current time step
   std::shared_ptr<DoFHandler<dim>> dof_handler;

   // move on one step (overwrite X_old with X)
   void next_step(size_t time_idx);

   // assembling steps of u and v that need to happen on the old mesh
   void assemble_u_pre(size_t time_idx);
   void assemble_v_pre(size_t time_idx);

   // move on to the mesh of the current time step,
   // interpolating system_rhs_[u,v] and tmp_u on the next mesh
   void next_mesh(size_t source_idx, size_t target_idx);

   // assemble matrices of the current time step into matrix_A, matrix_B and matrix_C.
   // if needed, this function can also do stuff so that the functions concerning D can run faster.
   virtual void assemble_matrices(double time) = 0;

   // assemble matrices and system_rhs_u for current mesh (calls assemble_matrices)
   void assemble(double time);

   // final assembly of rhs for u
   void assemble_u(size_t time_idx);

   // modify system_matrix, system_rhs_v and solution_u for Dirichlet boundary values
   virtual void apply_boundary_conditions_u(double time) = 0;

   // solve for u
   void solve_u();

   // final assembly of rhs for v
   void assemble_v(size_t time_idx);

   // modify system_matrix, rhs and solution_v for Dirichlet boundary values
   virtual void apply_boundary_conditions_v(double time) = 0;

   // solve for v
   void solve_v();

   /**
    * Destroy matrices and vectors.
    * This function should be called after computations to have minimal memory requirements when this object is not
    * currently in use.
    */
   virtual void cleanup();

   DiscretizedFunction<dim> apply_R_transpose(const DiscretizedFunction<dim>& u);
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_ABSTRACTEQUATIONADJOINT_H_ */
