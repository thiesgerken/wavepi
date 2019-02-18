/*
 * WaveEquationAdjoint.h
 *
 *  Created on: 17.0.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATIONADJOINT_H_
#define FORWARD_WAVEEQUATIONADJOINT_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <forward/AbstractEquationAdjoint.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationBase.h>

#include <stddef.h>
#include <memory>

namespace wavepi {
namespace forward {
using namespace dealii;

// parameters and rhs must currently be discretized on the same space-time grid!
// this is the adjoint equation when using vector norm in time and space
template <int dim>
class WaveEquationAdjoint : public AbstractEquationAdjoint<dim>, public WaveEquationBase<dim> {
 public:
  WaveEquationAdjoint(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

  WaveEquationAdjoint(const WaveEquation<dim>& wave);

  virtual ~WaveEquationAdjoint() = default;

 private:
  virtual void apply_boundary_conditions_u(double time);
  virtual void apply_boundary_conditions_v(double time);
  virtual void assemble_matrices(size_t time_idx);

  using AbstractEquationAdjoint<dim>::mesh;
  using AbstractEquationAdjoint<dim>::dof_handler;
  using AbstractEquationAdjoint<dim>::constraints;
  using AbstractEquationAdjoint<dim>::system_matrix;
  using AbstractEquationAdjoint<dim>::system_rhs_u;
  using AbstractEquationAdjoint<dim>::system_rhs_v;
  using AbstractEquationAdjoint<dim>::solution_u;
  using AbstractEquationAdjoint<dim>::solution_v;
  using AbstractEquationAdjoint<dim>::matrix_A;
  using AbstractEquationAdjoint<dim>::matrix_B;
  using AbstractEquationAdjoint<dim>::matrix_C;

  using WaveEquationBase<dim>::param_c;
  using WaveEquationBase<dim>::param_nu;
  using WaveEquationBase<dim>::param_rho;
  using WaveEquationBase<dim>::param_q;

  using WaveEquationBase<dim>::vmult_D_intermediate;
  using WaveEquationBase<dim>::vmult_D_intermediate_transpose;
  using WaveEquationBase<dim>::vmult_C_intermediate;

  virtual void vmult_D_intermediate(const SparseMatrix<double>& mass_matrix, Vector<double>& dst,
                                    const Vector<double>& src, double tolerance) const {
    WaveEquationBase<dim>::vmult_D_intermediate(mass_matrix, dst, src, tolerance);
  }
  virtual void vmult_D_intermediate_transpose(const SparseMatrix<double>& mass_matrix, Vector<double>& dst,
                                              const Vector<double>& src, double tolerance) const {
    WaveEquationBase<dim>::vmult_D_intermediate_transpose(mass_matrix, dst, src, tolerance);
  }

  virtual void vmult_C_intermediate(Vector<double>& dst, const Vector<double>& src) const {
    WaveEquationBase<dim>::vmult_C_intermediate(matrix_C, dst, src);
  }

  virtual void cleanup() override {
    AbstractEquationAdjoint<dim>::cleanup();
    WaveEquationBase<dim>::cleanup();
  }

  std::shared_ptr<Function<dim>> zero;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_WAVEEQUATION_H_ */
