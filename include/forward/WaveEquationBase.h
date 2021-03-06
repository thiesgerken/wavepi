/*
 * WaveEquationBase.h
 *
 *  Created on: 23.07.2017
 *      Author: thies
 */

#ifndef FORWARD_WAVEEQUATIONBASE_H_
#define FORWARD_WAVEEQUATIONBASE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <forward/AbstractEquation.h>
#include <forward/L2RightHandSide.h>
#include <forward/RightHandSide.h>

#include <tgmath.h>
#include <cmath>
#include <memory>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template <int dim>
class WaveEquationBase {
 public:
  // solvers for adjoint of L ∊ 𝓛(L²([0,T], L²(Ω)), L²([0,T], L²(Ω)))
  enum L2AdjointSolver { WaveEquationAdjoint, WaveEquationBackwards };

  virtual ~WaveEquationBase() = default;

  WaveEquationBase()
      : param_c(std::make_shared<ConstantFunction<dim>>(1.0)),
        param_nu(std::make_shared<ConstantFunction<dim>>(0.0)),
        param_rho(std::make_shared<ConstantFunction<dim>>(1.0)),
        param_q(std::make_shared<ConstantFunction<dim>>(0.0)),
        rho_time_dependent(false) {}

  // uses special functions for matrix assembly when discretized parameters are passed, which is a lot better for P1
  // elements. For P2 elements and 3 dimensions it actually turns out to be worse (too much coupling going on,
  // evaluating the polynomial is actually cheaper) in that case, you should turn of the specialization.
  bool is_special_assembly_recommended(std::shared_ptr<SpaceTimeMesh<dim>> mesh __attribute((unused))) const {
    return true;  // mesh->get_quadrature().size() < (1 << dim) || dim < 3;
  }

  bool using_special_assembly(std::shared_ptr<SpaceTimeMesh<dim>> mesh) {
    return special_assembly_tactic == 0 ? is_special_assembly_recommended(mesh) : (special_assembly_tactic > 0);
  }

  std::shared_ptr<LightFunction<dim>> get_param_rho() const { return param_rho; }

  void set_param_rho(std::shared_ptr<LightFunction<dim>> param_rho, bool is_time_dependent = true) {
    this->param_rho      = param_rho;
    this->param_rho_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(param_rho);

    this->rho_time_dependent = is_time_dependent;
  }

  std::shared_ptr<LightFunction<dim>> get_param_c() const { return param_c; }

  void set_param_c(std::shared_ptr<LightFunction<dim>> param_c) {
    this->param_c      = param_c;
    this->param_c_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(param_c);
  }

  std::shared_ptr<LightFunction<dim>> get_param_nu() const { return param_nu; }

  void set_param_nu(std::shared_ptr<LightFunction<dim>> param_nu) {
    this->param_nu      = param_nu;
    this->param_nu_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(param_nu);
  }

  std::shared_ptr<LightFunction<dim>> get_param_q() const { return param_q; }

  void set_param_q(std::shared_ptr<LightFunction<dim>> param_q) {
    this->param_q      = param_q;
    this->param_q_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, LightFunction<dim>>(param_q);
  }

  int get_special_assembly_tactic() const { return special_assembly_tactic; }

  void set_special_assembly_tactic(int special_assembly_tactic) {
    if (special_assembly_tactic > 0)
      this->special_assembly_tactic = 1;
    else if (special_assembly_tactic < 0)
      this->special_assembly_tactic = -1;
    else
      this->special_assembly_tactic = 0;
  }

  void fill_matrices(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx, SparseMatrix<double> &dst_A,
                     SparseMatrix<double> &dst_B, SparseMatrix<double> &dst_C);

  bool is_rho_time_dependent() const { return rho_time_dependent; }

  void set_rho_time_dependent(bool rho_time_dependent) { this->rho_time_dependent = rho_time_dependent; }

  // before mesh change, let dst <- (D^n)^{-1} D^{n-1} M^{-1} src
  // ( i.e. dst <- src for time-independent D)
  virtual void vmult_D_intermediate(const SparseMatrix<double> &mass_matrix, Vector<double> &dst,
                                    const Vector<double> &src, double tolerance) const;

  // before mesh change, let dst <- M^{-1} (D^n)^{-1} D^{n-1} src
  // ( i.e. dst <- src for time-independent D)
  virtual void vmult_D_intermediate_transpose(const SparseMatrix<double> &mass_matrix, Vector<double> &dst,
                                              const Vector<double> &src, double tolerance) const;

  // before mesh change, let dst <- (D^n)^{-1} C^{n-1} src
  // ( i.e. dst <- matrix_C * src for time-independent D)
  virtual void vmult_C_intermediate(const SparseMatrix<double> &matrix_C, Vector<double> &dst,
                                    const Vector<double> &src) const;

  virtual void cleanup() {
    matrix_C_intermediate.clear();
    matrix_D_intermediate.clear();
  }

 protected:
  void fill_A(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx, SparseMatrix<double> &destination);
  void fill_B(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx, SparseMatrix<double> &destination);
  void fill_C(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx, SparseMatrix<double> &destination);

  void fill_C_intermediate(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx);
  void fill_D_intermediate(std::shared_ptr<SpaceTimeMesh<dim>> mesh, size_t time_idx);

  // treat DiscretizedFunctions as parameters and right hand side differently
  // < 0 -> no (better if much coupling present), > 0 -> yes, = 0 automatically (default)
  int special_assembly_tactic = 0;

  std::shared_ptr<LightFunction<dim>> param_c, param_nu, param_rho, param_q;

  // filled, if the function handles above can be typecast into DiscretizedFunction<dim>
  std::shared_ptr<DiscretizedFunction<dim>> param_c_disc = nullptr, param_nu_disc = nullptr;
  std::shared_ptr<DiscretizedFunction<dim>> param_rho_disc = nullptr, param_q_disc = nullptr;

  // allows faster assembly if it is constant
  bool rho_time_dependent;

  // if needed: storage for (D^n)^{-1} D^{n-1}
  SparseMatrix<double> matrix_D_intermediate;

  // if needed: storage for (D^n)^{-1} C^{n-1}
  SparseMatrix<double> matrix_C_intermediate;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_FORWARD_WAVEEQUATIONBASE_H_ */
