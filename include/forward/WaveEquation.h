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
#include <forward/AbstractEquation.h>

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
template<int dim>
class WaveEquation: public AbstractEquation<dim>, public WaveEquationBase<dim> {
public:
   WaveEquation(std::shared_ptr<SpaceTimeMesh<dim>> mesh);
   WaveEquation(const WaveEquation<dim> &wave);
   virtual ~WaveEquation() = default;

   virtual DiscretizedFunction<dim> run(std::shared_ptr<RightHandSide<dim>> right_hand_side,
         typename AbstractEquation<dim>::Direction direction = AbstractEquation<dim>::Forward);

   std::shared_ptr<Function<dim>> get_boundary_values_u() const {
      return boundary_values_u;
   }

   void set_boundary_values_u(std::shared_ptr<Function<dim>> boundary_values_u) {
      this->boundary_values_u = boundary_values_u;
   }

   std::shared_ptr<Function<dim>> get_boundary_values_v() const {
      return boundary_values_v;
   }

   void set_boundary_values_v(std::shared_ptr<Function<dim>> boundary_values_v) {
      this->boundary_values_v = boundary_values_v;
   }

   std::shared_ptr<Function<dim>> get_initial_values_u() const {
      return initial_values_u;
   }

   void set_initial_values_u(std::shared_ptr<Function<dim>> initial_values_u) {
      this->initial_values_u = initial_values_u;
   }

   std::shared_ptr<Function<dim>> get_initial_values_v() const {
      return initial_values_v;
   }

   void set_initial_values_v(std::shared_ptr<Function<dim>> initial_values_v) {
      this->initial_values_v = initial_values_v;
   }

protected:
   virtual void apply_boundary_conditions_u(double time);
   virtual void apply_boundary_conditions_v(double time);
   virtual void assemble_matrices(size_t time_idx);
   virtual void initial_values(double time);

   using AbstractEquation<dim>::mesh;
   using AbstractEquation<dim>::dof_handler;
   using AbstractEquation<dim>::constraints;
   using AbstractEquation<dim>::system_matrix;
   using AbstractEquation<dim>::system_rhs;
   using AbstractEquation<dim>::solution_u;
   using AbstractEquation<dim>::solution_v;
   using AbstractEquation<dim>::matrix_A;
   using AbstractEquation<dim>::matrix_B;
   using AbstractEquation<dim>::matrix_C;

   using WaveEquationBase<dim>::param_c;
   using WaveEquationBase<dim>::param_nu;
   using WaveEquationBase<dim>::param_rho;
   using WaveEquationBase<dim>::param_q;

   using WaveEquationBase<dim>::vmult_D_intermediate;
   using WaveEquationBase<dim>::vmult_C_intermediate;

   virtual void vmult_D_intermediate(const SparseMatrix<double> &mass_matrix, Vector<double>& dst,
         const Vector<double>& src) const {
      WaveEquationBase<dim>::vmult_D_intermediate(mass_matrix, dst, src);
   }

   virtual void vmult_C_intermediate(Vector<double>& dst, const Vector<double>& src) const {
      WaveEquationBase<dim>::vmult_C_intermediate(matrix_C, dst, src);
   }

   virtual void cleanup() override {
      AbstractEquation<dim>::cleanup();
      WaveEquationBase<dim>::cleanup();
   }
private:
   std::shared_ptr<Function<dim>> initial_values_u, initial_values_v;
   std::shared_ptr<Function<dim>> boundary_values_u, boundary_values_v;
};
} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
