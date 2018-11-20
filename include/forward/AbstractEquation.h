/*
 * AbstractEquation.h
 *
 *  Created on: 14.11.2018
 *      Author: thies
 */

#ifndef FORWARD_ABSTRACTEQUATION_H_
#define FORWARD_ABSTRACTEQUATION_H_

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
class AbstractEquation {
public:
   enum Direction {
      Forward, Backward
   };

   AbstractEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh)
         : mesh(mesh) {
   }

   virtual ~AbstractEquation() = default;

   static void declare_parameters(ParameterHandler &prm);
   void get_parameters(ParameterHandler &prm);

   virtual DiscretizedFunction<dim> run(std::shared_ptr<RightHandSide<dim>> right_hand_side, Direction direction =
         Forward);

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

   // matrices corresponding to the operators A, B, C at the current and the last time step
   SparseMatrix<double> matrix_A;
   SparseMatrix<double> matrix_B;
   SparseMatrix<double> matrix_C;

   // solution and its derivative at the current and the last time step
   Vector<double> solution_u, solution_v;
   Vector<double> solution_u_old, solution_v_old;
   Vector<double> rhs, rhs_old;

   // handle to the right hand side used
   std::shared_ptr<RightHandSide<dim>> right_hand_side;

   // space for linear systems and their right hand sides
   SparseMatrix<double> system_matrix;
   Vector<double> system_rhs;

   // stuff that assemble_pre wants to pass on to assemble_u and assemble_v
   Vector<double> system_tmp1;  // X^n_1
   Vector<double> system_tmp2;  // X^n_2

   // DoFHandler for the current time step
   std::shared_ptr<DoFHandler<dim>> dof_handler;

   // initialize vectors and matrices
   void init_system(size_t first_idx);

   // fill solution_u and solution_v with initial values
   virtual void initial_values(double time) = 0;

   // assembling steps of u and v that need to happen on the old mesh
   void assemble_pre(double time_step);

   // move on to the mesh of the current time step,
   // interpolating system_rhs_[u,v] and tmp_u on the next mesh
   void next_mesh(size_t source_idx, size_t target_idx);

   // assemble matrices of the current time step into matrix_A, matrix_B and matrix_C.
   // if needed, this function can also do stuff so that the functions concerning D can run faster.
   virtual void assemble_matrices(double time) = 0;

   // assemble matrices and rhs for current mesh (calls assemble_matrices)
   void assemble(double time);

   // final assembly of rhs for u
   void assemble_u(double time, double time_step);

   // modify system_matrix, rhs and solution_u for Dirichlet boundary values
   virtual void apply_boundary_conditions_u(double time) = 0;

   // solve for u
   void solve_u();

   // final assembly of rhs for v
   void assemble_v(double time, double time_step);

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

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_FORWARD_ABSTRACTEQUATION_H_ */
