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
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/solution_transfer.h>

#include <forward/AdaptiveMesh.h>

#include <stddef.h>
#include <initializer_list>
#include <iostream>

namespace wavepi {
namespace forward {

template<int dim>
AdaptiveMesh<dim>::AdaptiveMesh(std::vector<double> times, FE_Q<dim> fe, Quadrature<dim> quad,
      std::shared_ptr<Triangulation<dim>> tria)
      : SpaceTimeMesh<dim>(times, fe, quad), initial_triangulation(tria) {
   patches = std::vector<Patch>(this->length());
   vector_sizes = std::vector<size_t>(this->length(), 0);
   sparsity_patterns = std::vector<std::shared_ptr<SparsityPattern>>(this->length());
   mass_matrices = std::vector<std::shared_ptr<SparseMatrix<double>>>(this->length());

   working_time_idx = -1;

   working_triangulation = std::make_shared<Triangulation<dim>>();
   working_triangulation->copy_triangulation(*initial_triangulation);

   working_dof_handler = std::make_shared<DoFHandler<dim>>();
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
std::shared_ptr<SparseMatrix<double>> AdaptiveMesh<dim>::get_mass_matrix(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   if (!mass_matrices[idx]) {
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
      get_dof_handler(idx);

      DynamicSparsityPattern dsp(working_dof_handler->n_dofs(), working_dof_handler->n_dofs());
      DoFTools::make_sparsity_pattern(*working_dof_handler, dsp);

      sparsity_patterns[idx] = std::make_shared<SparsityPattern>();
      sparsity_patterns[idx]->copy_from(dsp);
   }

   return sparsity_patterns[idx];
}

template<int dim>
std::shared_ptr<DoFHandler<dim> > AdaptiveMesh<dim>::get_dof_handler(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   return transfer(idx, { });
}

template<int dim>
std::shared_ptr<Triangulation<dim> > AdaptiveMesh<dim>::get_triangulation(size_t idx) {
   Assert(idx >= 0 && idx < this->length(), ExcIndexRange(idx, 0, this->length()));

   transfer(idx, { });
   return working_triangulation;
}

template<int dim>
std::shared_ptr<DoFHandler<dim> > AdaptiveMesh<dim>::transfer(size_t source_time_index,
      size_t target_time_index, std::initializer_list<Vector<double>*> vectors) {
   LogStream::Prefix p("AdaptiveMesh");
   deallog << "Mesh transfer: " << source_time_index << " → " << target_time_index << ", taking "
         << vectors.size() << " vector(s) along" << std::endl;

   Assert(source_time_index >= 0 && source_time_index < this->length(),
         ExcIndexRange(source_time_index, 0, this->length()));
   Assert(target_time_index >= 0 && target_time_index < this->length(),
         ExcIndexRange(target_time_index, 0, this->length()));

   transfer(source_time_index, { });
   return transfer(target_time_index, vectors);
}

// transfer with source_time_index = working_time_idx
template<int dim>
std::shared_ptr<DoFHandler<dim> > AdaptiveMesh<dim>::transfer(size_t target_idx,
      std::initializer_list<Vector<double>*> vectors) {
   Assert(target_idx >= 0 && target_idx < this->length(), ExcIndexRange(target_idx, 0, this->length()));

   if (working_time_idx < target_idx) {
      AssertThrow(false, ExcNotImplemented());

      for (size_t idx = working_time_idx + 1; idx <= target_idx; idx++) {
         for (auto patch : patches[idx]) {
            auto cells_to_refine = patch.first;
            auto cells_to_coarsen = patch.second;

            if (cells_to_coarsen.size() == 0) {
               SolutionTransfer<dim, Vector<double>> trans(*working_dof_handler);

               working_triangulation->load_refine_flags(cells_to_refine);
               working_triangulation->load_coarsen_flags(cells_to_coarsen);

               bool flags_changed = working_triangulation->prepare_coarsening_and_refinement();
               Assert(!flags_changed, ExcMessage("refine flags are not complete!"));

               trans.prepare_for_pure_refinement();

               working_triangulation->execute_coarsening_and_refinement();

               working_dof_handler->distribute_dofs(fe);

               for (auto vec : vectors) {
                  Vector<double> tmp(*vec);
                  vec->reinit(working_dof_handler->n_dofs());
                  trans.refine_interpolate(tmp, *vec);
               }
            } else {
               AssertThrow(false, ExcNotImplemented());
            }
         }
      }

   } else if (working_time_idx > target_idx) {
      AssertThrow(false, ExcNotImplemented());
   }

   return working_dof_handler;
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
void AdaptiveMesh<dim>::refine_and_coarsen(std::vector<size_t> refine_intervals,
      std::vector<size_t> coarsen_time_points, std::vector<std::vector<bool>> refine_trias,
      std::vector<std::vector<bool>> coarsen_trias,
      std::initializer_list<DiscretizedFunction<dim>*> interpolate_vectors) {
   AssertThrow(false, ExcNotImplemented());
}

template class AdaptiveMesh<1> ;
template class AdaptiveMesh<2> ;
template class AdaptiveMesh<3> ;

} /* namespace forward */
} /* namespace wavepi */
