/*
 * WaveEquation.h
 *
 *  Created on: 05.05.2017
 *      Author: thies
 */

#ifndef INCLUDE_WAVEEQUATION_H_
#define INCLUDE_WAVEEQUATION_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

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

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <fstream>
#include <iostream>
#include <utility>

namespace wavepi {
   using namespace dealii;

   template<int dim>
   class WaveEquation {
      public:
         WaveEquation();
         void run();

         Function<dim> *initial_values_u, *initial_values_v;
         Function<dim> *boundary_values_u, *boundary_values_v;
         Function<dim> *right_hand_side;
         Function<dim> *param_c, *param_nu, *param_a, *param_q;

         double theta;
         double time_end; // = T
         double time_step; // = \Delta t

         ZeroFunction<dim> zero = ZeroFunction<dim>(1);
         ConstantFunction<dim> one = ConstantFunction<dim>(1.0, 1);
      private:
         void init_system();
         void setup_step();
         void assemble_u();
         void assemble_v();
         void solve_u();
         void solve_v();
         void output_results() const;

         Triangulation<dim> triangulation;
         FE_Q<dim> fe;
         DoFHandler<dim> dof_handler;

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

         // current time and time step
         unsigned int timestep_number;
         double time;

         // for output to paraview
         std::vector<std::pair<double, std::string>> times_and_names;
   };
}

#endif /* INCLUDE_WAVEEQUATION_H_ */
