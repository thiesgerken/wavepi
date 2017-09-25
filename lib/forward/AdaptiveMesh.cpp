/*
 * AdaptiveMesh.cpp
 *
 *  Created on: 09.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/memory_consumption.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/cell_id.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <forward/AdaptiveMesh.h>

#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <list>

namespace wavepi {
namespace forward {

template<int dim>
AdaptiveMesh<dim>::AdaptiveMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
      std::shared_ptr<Triangulation<dim>> tria)
      : SpaceTimeMesh<dim>(times, fe, quad), initial_triangulation(tria) {
   forward_patches = std::vector < Patch > (this->length() - 1);
   backward_patches = std::vector < Patch > (this->length() - 1);

   vector_sizes = std::vector < size_t > (this->length(), 0);
   sparsity_patterns = std::vector < std::shared_ptr < SparsityPattern >> (this->length());
   mass_matrices = std::vector < std::shared_ptr<SparseMatrix<double>> > (this->length());
   constraints = std::vector < std::shared_ptr < ConstraintMatrix >> (this->length());

   working_time_idx = 0;

   working_triangulation = std::make_shared<Triangulation<dim>>();
   working_triangulation->copy_triangulation(*initial_triangulation);

   working_dof_handler = std::make_shared<DoFHandler<dim>>();
   working_dof_handler->initialize(*working_triangulation, this->fe);
}

template<int dim>
void AdaptiveMesh<dim>::reset() {
   vector_sizes = std::vector < size_t > (this->length(), 0);
   mass_matrices = std::vector < std::shared_ptr<SparseMatrix<double>> > (this->length());
   sparsity_patterns = std::vector < std::shared_ptr < SparsityPattern >> (this->length());
   constraints = std::vector < std::shared_ptr < ConstraintMatrix >> (this->length());

   working_time_idx = 0;

   working_dof_handler = std::make_shared<DoFHandler<dim>>();

   working_triangulation = std::make_shared<Triangulation<dim>>();
   working_triangulation->copy_triangulation(*initial_triangulation);

   working_dof_handler->initialize(*working_triangulation, this->fe);
}

template<int dim>
size_t AdaptiveMesh<dim>::n_dofs(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   if (vector_sizes[idx] == 0) {
      get_dof_handler(idx);
      vector_sizes[idx] = working_dof_handler->n_dofs();
   }

   return vector_sizes[idx];
}

template<int dim>
std::shared_ptr<ConstraintMatrix> AdaptiveMesh<dim>::get_constraint_matrix(size_t idx) {
   if (!constraints[idx]) {
      {
         LogStream::Prefix p("AdaptiveMesh");
         deallog << "making constraint matrix for time step " << idx << std::endl;
      }

      get_dof_handler(idx);
      constraints[idx] = std::make_shared<ConstraintMatrix>();
      DoFTools::make_hanging_node_constraints(*working_dof_handler, *constraints[idx]);
      constraints[idx]->close();
   }

   return constraints[idx];
}

template<int dim>
std::shared_ptr<SparseMatrix<double>> AdaptiveMesh<dim>::get_mass_matrix(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   if (!mass_matrices[idx]) {
      //{
      //   LogStream::Prefix p("AdaptiveMesh");
      //   deallog << "making mass matrix for time step " << idx << std::endl;
      //}

      get_sparsity_pattern(idx);
      get_dof_handler(idx);

      mass_matrices[idx] = std::make_shared<SparseMatrix<double>>(*sparsity_patterns[idx]);
      dealii::MatrixCreator::create_mass_matrix(*working_dof_handler, quad, *mass_matrices[idx]);
   }

   return mass_matrices[idx];
}

template<int dim> std::shared_ptr<SparsityPattern> AdaptiveMesh<dim>::get_sparsity_pattern(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   if (!sparsity_patterns[idx]) {
      //{
      //   LogStream::Prefix p("AdaptiveMesh");
      //   deallog << "making sparsity pattern for time step " << idx << std::endl;
      //}

      get_constraint_matrix(idx);
      get_dof_handler(idx);

      DynamicSparsityPattern dsp(working_dof_handler->n_dofs(), working_dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(*working_dof_handler, dsp, *constraints[idx], true);

      sparsity_patterns[idx] = std::make_shared<SparsityPattern>();
      sparsity_patterns[idx]->copy_from(dsp);
   }

   return sparsity_patterns[idx];
}

template<int dim>
std::shared_ptr<DoFHandler<dim> > AdaptiveMesh<dim>::get_dof_handler(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   transfer(idx);
   return working_dof_handler;
}

template<int dim>
std::shared_ptr<Triangulation<dim> > AdaptiveMesh<dim>::get_triangulation(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   transfer(idx);
   return working_triangulation;
}

template<int dim>
std::shared_ptr<DoFHandler<dim> > AdaptiveMesh<dim>::transfer(size_t source_time_index,
      size_t target_time_index, std::initializer_list<Vector<double>*> vectors) {
   Assert(source_time_index >= 0 && source_time_index < this->length(),
         ExcIndexRange(source_time_index, 0, this->length()));
   Assert(target_time_index >= 0 && target_time_index < this->length(),
         ExcIndexRange(target_time_index, 0, this->length()));

   transfer(source_time_index);

   std::vector<Vector<double>*> vecs;
   for (auto vec : vectors)
      vecs.push_back(vec);

   transfer(target_time_index, vecs);

   return working_dof_handler;
}

template<int dim>
bool AdaptiveMesh<dim>::any(const std::vector<bool> &v) const {
   for (auto b : v)
      if (b)
         return true;

   return false;
}

template<int dim>
void AdaptiveMesh<dim>::patch(const Patch &patch, std::vector<Vector<double>*> &vectors) {
   for (auto p : patch) {
      auto cells_to_refine = p.first;
      auto cells_to_coarsen = p.second;

      LogStream::Prefix lp("patch");
      deallog << "Applying patch (working_time_idx = " << working_time_idx << ") in a set of " << patch.size()
            << std::endl;

      if (vectors.size() == 0) {
         working_triangulation->load_refine_flags(cells_to_refine);
         working_triangulation->load_coarsen_flags(cells_to_coarsen);

         working_triangulation->prepare_coarsening_and_refinement();
         working_triangulation->execute_coarsening_and_refinement();

         working_dof_handler->distribute_dofs(fe);
      } else if (!any(cells_to_coarsen)) {
         SolutionTransfer<dim, Vector<double>> trans(*working_dof_handler);

         working_triangulation->load_refine_flags(cells_to_refine);
         working_triangulation->prepare_coarsening_and_refinement();
         trans.prepare_for_pure_refinement();

         working_triangulation->execute_coarsening_and_refinement();
         working_dof_handler->distribute_dofs(fe);

         for (auto vec : vectors) {
            Vector<double> tmp(*vec);
            vec->reinit(working_dof_handler->n_dofs());
            trans.refine_interpolate(tmp, *vec);
         }
      } else {
         SolutionTransfer<dim, Vector<double>> trans(*working_dof_handler);

         std::vector<Vector<double>> all_in;
         for (auto vec : vectors)
            all_in.emplace_back(*vec);

         working_triangulation->load_refine_flags(cells_to_refine);
         working_triangulation->load_coarsen_flags(cells_to_coarsen);
         working_triangulation->prepare_coarsening_and_refinement();
         trans.prepare_for_coarsening_and_refinement(all_in);

         working_triangulation->execute_coarsening_and_refinement();
         working_dof_handler->distribute_dofs(fe);

         std::vector<Vector<double>> all_out(vectors.size(), Vector<double>(working_dof_handler->n_dofs()));
         trans.interpolate(all_in, all_out);

         for (size_t i = 0; i < vectors.size(); i++)
            *vectors[i] = all_out[i];
      }
   }
}
template<int dim>
void AdaptiveMesh<dim>::transfer(size_t target_idx) {
   std::vector<Vector<double>*> vecs;
   transfer(target_idx, vecs);
}

template<int dim>
void AdaptiveMesh<dim>::transfer(size_t target_idx, std::vector<Vector<double>*> &vectors) {
   Assert(target_idx >= 0 && target_idx < this->length(), ExcIndexRange(target_idx, 0, this->length()));

   LogStream::Prefix p("AdaptiveMesh");
   LogStream::Prefix pp("transfer");

   if (target_idx != working_time_idx)
      deallog << working_time_idx << " → " << target_idx << ", taking " << vectors.size()
            << " vector(s) along" << std::endl;

   if (working_time_idx < target_idx)
      for (size_t idx = working_time_idx; idx < target_idx; idx++) {
         patch(forward_patches[idx], vectors);
         working_time_idx++;

         // hanging node constraints
         // (only needed if we coarsened any cells)
         // is this even required after each patch element? would require a lot more ConstraintMatrices ...
         // commented out because constraining in WaveEquation results in reflections from the 'refinement boundary'
         // the solution fullfills the constraints because of distribute calls in WaveEquation and WaveEquationAdjoint.
         // for (size_t i = 0; i < vectors.size(); i++)
         //   get_constraint_matrix(working_time_idx)->distribute(*vectors[i]);
      }
   else if (working_time_idx > target_idx)
      for (ssize_t idx = working_time_idx; idx > (ssize_t) target_idx; idx--) {
         patch(backward_patches[idx - 1], vectors);
         working_time_idx--;

         // hanging node constraints!
         // (only needed if we coarsened any cells)
         // is this even required after each patch element? would require a lot more ConstraintMatrices ...
         // commented out because constraining in WaveEquation results in reflections from the 'refinement boundary'
         // the solution fullfills the constraints because of distribute calls in WaveEquation and WaveEquationAdjoint.
         // for (size_t i = 0; i < vectors.size(); i++)
         //   get_constraint_matrix(working_time_idx)->distribute(*vectors[i]);
      }
}

template<int dim>
size_t AdaptiveMesh<dim>::memory_consumption() const {
   size_t mem = MemoryConsumption::memory_consumption(*working_dof_handler)
         + MemoryConsumption::memory_consumption(working_dof_handler)
         + MemoryConsumption::memory_consumption(*working_triangulation)
         + MemoryConsumption::memory_consumption(working_triangulation)
         + MemoryConsumption::memory_consumption(*initial_triangulation)
         + MemoryConsumption::memory_consumption(initial_triangulation)
         + MemoryConsumption::memory_consumption(times)
         + MemoryConsumption::memory_consumption(sparsity_patterns)
         + MemoryConsumption::memory_consumption(mass_matrices)
         + MemoryConsumption::memory_consumption(vector_sizes)
         + MemoryConsumption::memory_consumption(working_time_idx);

   for (size_t i = 0; i < this->length(); i++) {
      if (mass_matrices[i])
         mem += MemoryConsumption::memory_consumption(*mass_matrices[i]);

      if (sparsity_patterns[i])
         mem += MemoryConsumption::memory_consumption(*sparsity_patterns[i]);
   }

   return mem;
}

template<int dim>
const std::vector<Patch>& AdaptiveMesh<dim>::get_forward_patches() const {
   return forward_patches;
}

template<int dim>
void AdaptiveMesh<dim>::set_forward_patches(const std::vector<Patch>& forward_patches) {
   this->forward_patches = forward_patches;

   reset();
   generate_backward_patches();
}

template<int dim>
void AdaptiveMesh<dim>::generate_backward_patches() {
   Assert(working_time_idx == 0, ExcInternalError());

   for (size_t idx = 0; idx < this->length() - 1; idx++, working_time_idx++) {
      Patch bw_patches;

      for (auto p : forward_patches[idx]) {
         auto cells_to_refine = p.first;
         auto cells_to_coarsen = p.second;

         working_triangulation->load_refine_flags(cells_to_refine);
         working_triangulation->load_coarsen_flags(cells_to_coarsen);
         working_triangulation->prepare_coarsening_and_refinement();

         // Save cell ids of cells to be refined,
         // and parents of cells that will be coarsened (will appear multiple times!)
         std::list < CellId > ids_to_refine;
         std::list < CellId > ids_to_coarsen;

         for (auto cell : working_triangulation->active_cell_iterators())
            if (cell->coarsen_flag_set())
               ids_to_coarsen.push_back(cell->parent()->id());
            else if (cell->refine_flag_set())
               ids_to_refine.push_back(cell->id());

         working_triangulation->execute_coarsening_and_refinement();
         working_dof_handler->distribute_dofs(fe);

         auto refine_it = ids_to_refine.begin();
         auto coarsen_it = ids_to_coarsen.begin();

         // Set refinement flags and coarsening flags according to saved cell ids
         for (auto cell : working_triangulation->active_cell_iterators()) {
            auto parent_id = cell->parent()->id();

            while (refine_it != ids_to_refine.end() && *refine_it < parent_id)
               refine_it++;

            if (refine_it != ids_to_refine.end() && *refine_it == parent_id)
               cell->set_coarsen_flag();

            auto my_id = cell->id();

            while (coarsen_it != ids_to_coarsen.end() && *coarsen_it < my_id)
               coarsen_it++;

            if (coarsen_it != ids_to_coarsen.end() && *coarsen_it == my_id)
               cell->set_refine_flag();
         }

         cells_to_refine.clear();
         cells_to_coarsen.clear();

         working_triangulation->save_refine_flags(cells_to_refine);
         working_triangulation->save_coarsen_flags(cells_to_coarsen);

         bw_patches.emplace_back(cells_to_refine, cells_to_coarsen);
      }

      // patches need to be applied in reverse order
      std::reverse(bw_patches.begin(), bw_patches.end());
      backward_patches[idx] = bw_patches;
   }
}

template<int dim>
void AdaptiveMesh<dim>::refine_and_coarsen(std::vector<size_t> refine_intervals,
      std::vector<size_t> coarsen_time_points, std::vector<std::vector<bool>> refine_trias,
      std::vector<std::vector<bool>> coarsen_trias,
      std::initializer_list<DiscretizedFunction<dim>*> interpolate_vectors) {
   // TODO: implement refine_and_coarsen
   AssertThrow(false, ExcNotImplemented());
}

template class AdaptiveMesh<1> ;
template class AdaptiveMesh<2> ;
template class AdaptiveMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */
