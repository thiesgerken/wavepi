/*
 * WaveEquationBase.h
 *
 *  Created on: 23.07.2017
 *      Author: thies
 */

#ifndef LIB_FORWARD_WAVEEQUATIONBASE_H_
#define LIB_FORWARD_WAVEEQUATIONBASE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/sparse_matrix.h>

#include <forward/DiscretizedFunction.h>
#include <forward/L2RightHandSide.h>
#include <forward/RightHandSide.h>
#include <forward/SpaceTimeMesh.h>

#include <cmath>
#include <memory>
#include <tgmath.h>

namespace wavepi {
namespace forward {

template<int dim>
class WaveEquationBase {
   public:
      // solvers for adjoint of L : L^2 -> L^2
      enum L2AdjointSolver {
         WaveEquationAdjoint, WaveEquationBackwards
      };

      std::shared_ptr<Function<dim>> zero = std::make_shared<ZeroFunction<dim>>(1);
      std::shared_ptr<Function<dim>> one = std::make_shared<ConstantFunction<dim>>(1.0, 1);
      std::shared_ptr<RightHandSide<dim>> zero_rhs = std::make_shared<L2RightHandSide<dim>>(zero);

      virtual ~WaveEquationBase() = default;

      WaveEquationBase(std::shared_ptr<SpaceTimeMesh<dim>> mesh);

      virtual DiscretizedFunction<dim> run() = 0;

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

      // uses special functions for matrix assembly when discretized parameters are passed, which is a lot better for P1 elements.
      // For P2 elements and 3 dimensions it actually turns out to be worse
      // (too much coupling going on, evaluating the polynomial is actually cheaper)
      // in that case, you should turn of the specialization.
      inline bool is_special_assembly_recommended() const {
         return mesh->get_quadrature().size() < (1 << dim) || dim < 3;
      }

      inline bool using_special_assembly() {
         return
               special_assembly_tactic == 0 ?
                     is_special_assembly_recommended() : (special_assembly_tactic > 0);
      }

      inline std::shared_ptr<SpaceTimeMesh<dim> > get_mesh() const {
         return this->mesh;
      }

      inline void set_mesh(std::shared_ptr<SpaceTimeMesh<dim> > mesh) {
         this->mesh = mesh;
      }

      inline double get_tolerance() const {
         return tolerance;
      }

      inline void set_tolerance(double tolerance) {
         this->tolerance = tolerance;
      }

      inline std::shared_ptr<Function<dim>> get_param_a() const {
         return param_a;
      }

      inline void set_param_a(std::shared_ptr<Function<dim>> param_a) {
         this->param_a = param_a;
         this->param_a_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_a);
      }

      inline std::shared_ptr<Function<dim>> get_param_c() const {
         return param_c;
      }

      inline void set_param_c(std::shared_ptr<Function<dim>> param_c) {
         this->param_c = param_c;
         this->param_c_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_c);
      }

      inline std::shared_ptr<Function<dim>> get_param_nu() const {
         return param_nu;
      }

      inline void set_param_nu(std::shared_ptr<Function<dim>> param_nu) {
         this->param_nu = param_nu;
         this->param_nu_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_nu);
      }

      inline std::shared_ptr<Function<dim>> get_param_q() const {
         return param_q;
      }

      inline void set_param_q(std::shared_ptr<Function<dim>> param_q) {
         this->param_q = param_q;
         this->param_q_disc = std::dynamic_pointer_cast<DiscretizedFunction<dim>, Function<dim>>(param_q);
      }

      inline std::shared_ptr<RightHandSide<dim>> get_right_hand_side() const {
         return right_hand_side;
      }

      inline void set_right_hand_side(std::shared_ptr<RightHandSide<dim> > right_hand_side) {
         this->right_hand_side = right_hand_side;
      }

      inline double get_theta() const {
         return theta;
      }

      inline void set_theta(double theta) {
         this->theta = theta;
      }

      inline int get_special_assembly_tactic() const {
         return special_assembly_tactic;
      }

      inline void set_special_assembly_tactic(int special_assembly_tactic) {
         if (special_assembly_tactic > 0)
            this->special_assembly_tactic = 1;
         else if (special_assembly_tactic < 0)
            this->special_assembly_tactic = -1;
         else
            this->special_assembly_tactic = 0;
      }

   protected:
      void fill_A(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination);
      void fill_B(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination);
      void fill_C(DoFHandler<dim> &dof_handler, SparseMatrix<double>& destination);

      double theta;
      double tolerance = 1e-8;

      // treat DiscretizedFunctions as params and right hand side differently
      // < 0 -> no (better if much coupling present), > 0 -> yes, = 0 automatically (default)
      int special_assembly_tactic = 0;

      std::shared_ptr<SpaceTimeMesh<dim>> mesh;

      std::shared_ptr<Function<dim>> param_c, param_nu, param_a, param_q;

      std::shared_ptr<DiscretizedFunction<dim>> param_c_disc = nullptr, param_nu_disc = nullptr;
      std::shared_ptr<DiscretizedFunction<dim>> param_a_disc = nullptr, param_q_disc = nullptr;

      std::shared_ptr<RightHandSide<dim>> right_hand_side;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_FORWARD_WAVEEQUATIONBASE_H_ */
