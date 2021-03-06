/*
 * MatrixCreator.h
 *
 *  Created on: 28.06.2017
 *      Author: thies
 */

#ifndef UTIL_MATRIXCREATOR_H_
#define UTIL_MATRIXCREATOR_H_

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/vector.h>
#include <memory>
#include <vector>

#include <base/LightFunction.h>

namespace wavepi {
namespace forward {

using namespace wavepi::base;
using namespace dealii;

template<int dim>
class MatrixCreator {
public:

   /**
    * like dealii::MatrixCreator::create_laplace_matrix, but with a zero order coefficient q as well and 1/ρ instead of ρ,
    * i.e. the bilinear form (u,v) ↦ ∇u·∇v/ρ + quv.
    * ρ and q must be valid function handles.
    */
   static void create_A_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> rho, std::shared_ptr<LightFunction<dim>> q, const double time);

   /**
    * like `dealii::MatrixCreator::create_laplace_matrix`, but with a zero order coefficient q as well and 1/ρ instead of ρ,
    * i.e. the bilinear form (u,v) ↦ ∇u·∇v/ρ + quv.
    * ρ must be a valid function handle and q a discretized function on the same mesh.
    */
   static void create_A_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> rho, const Vector<double> &q, const double time);

   /**
    * like `dealii::MatrixCreator::create_laplace_matrix`, but with a zero order coefficient q as well and 1/ρ instead of ρ,
    * i.e. the bilinear form (u,v) ↦ ∇u·∇v/ρ + quv.
    * q must be a valid function handle and ρ a discretized function on the same mesh.
    */
   static void create_A_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &rho, std::shared_ptr<LightFunction<dim>> q, const double time);

   /**
    * like `dealii::MatrixCreator::create_laplace_matrix`, but with a zero order coefficient q as well and 1/ρ instead of ρ,
    * i.e. the bilinear form (u,v) ↦ ∇u·∇v/ρ + quv.
    * ρ and q are supplied as discretized FE functions  (living on the same mesh).
    */
   static void create_A_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &rho, const Vector<double> &q);

   /**
    * like `dealii::MatrixCreator::create_mass_matrix`, but with a discretized coefficient c (living on the same mesh)
    * you could just pass this coefficient to `dealii::MatrixCreator::create_mass_matrix`, but in tests this took 20x
    * longer than when using the continuous version. This implementation did it in 2x the time.
    */
   static void create_mass_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &c);

   /**
    * like `dealii::MatrixCreator::create_mass_matrix`, but with a LightFunction<dim>
     */
   static void create_mass_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> c, const double time);

   /**
    * discretizes the bilinear form (u,v) ↦ uv/(ρ c²).
    * ρ and c must be valid function handles.
    */
   static void create_C_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> rho, std::shared_ptr<LightFunction<dim>> c, const double time_rho, const double time_c);

   /**
    * discretizes the bilinear form (u,v) ↦ uv/(ρ c²).
    *  ρ must be a valid function handle and c a discretized function on the same mesh.
    */
   static void create_C_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> rho, const Vector<double> &c, const double time_rho);

   /**
    * discretizes the bilinear form (u,v) ↦ uv/(ρ c²).
    * c must be a valid function handle and ρ a discretized function on the same mesh.
    */
   static void create_C_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &rho, std::shared_ptr<LightFunction<dim>> c, const double time_c);

   /**
    * discretizes the bilinear form (u,v) ↦ uv/(ρ c²).
    * ρ and c are supplied as discretized FE functions  (living on the same mesh).
    */
   static void create_C_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &rho, const Vector<double> &c);

   /**
    * discretizes the bilinear form (u,v) ↦ uv ρ^n/ρ^{n+1}.
    * ρ is supplied as a discretized FE function.
    */
   static void create_D_intermediate_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         const Vector<double> &rho_current, const Vector<double> &rho_next);

   /**
    * discretizes the bilinear form (u,v) ↦ uv ρ^n/ρ^{n+1}.
    * ρ must be a valid function handle, which will be modified (need to change its time)
    */
   static void create_D_intermediate_matrix(std::shared_ptr<DoFHandler<dim>> dof, const Quadrature<dim> &quad, SparseMatrix<double> &matrix,
         std::shared_ptr<LightFunction<dim>> rho, double current_time, double next_time);

private:
   // scratch data for assembly that needs function values and gradients
   struct LaplaceAssemblyScratchData {
      LaplaceAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
      LaplaceAssemblyScratchData(const LaplaceAssemblyScratchData &scratch_data);
      FEValues<dim> fe_values;
   };

   // scratch data for assembly that only needs function values
   struct MassAssemblyScratchData {
      MassAssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
      MassAssemblyScratchData(const MassAssemblyScratchData &scratch_data);
      FEValues<dim> fe_values;
   };

   struct AssemblyCopyData {
      FullMatrix<double> cell_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
   };

   static void copy_local_to_global(SparseMatrix<double> &matrix, const AssemblyCopyData &copy_data);

   static void local_assemble_mass_d(const Vector<double> &c, const typename DoFHandler<dim>::active_cell_iterator &cell,
         MassAssemblyScratchData &scratch_data, AssemblyCopyData &copy_data);

   static void local_assemble_A_dc(const Vector<double> &a, const LightFunction<dim> * const q, const double time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_A_cd(const LightFunction<dim> * const a, const Vector<double> &q,const double time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_A_dd(const Vector<double> &a, const Vector<double> &q,
         const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_A_cc(const LightFunction<dim> * const a, const LightFunction<dim> * const q,const double time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, LaplaceAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_C_dc(const Vector<double> &a, const LightFunction<dim> * const c,const double c_time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_C_cd(const LightFunction<dim> * const a, const Vector<double> &c, const double a_time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_C_dd(const Vector<double> &a, const Vector<double> &c,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_C_cc(const LightFunction<dim> * const a, const LightFunction<dim> * const c, const double a_time, const double c_time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_D_intermediate_d(const Vector<double> &rho_current, const Vector<double> &rho_next,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_D_intermediate_c(const LightFunction<dim> * const rho, const double time_current, const double time_next,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

   static void local_assemble_mass_c(const LightFunction<dim> * const c, const double time,
         const typename DoFHandler<dim>::active_cell_iterator &cell, MassAssemblyScratchData &scratch_data,
         AssemblyCopyData &copy_data);

};
}  // namespace forward
} /* namespace wavepi */

#endif /* UTIL_MATRIXCREATOR_H_ */
