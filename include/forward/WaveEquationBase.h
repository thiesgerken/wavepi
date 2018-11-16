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

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <forward/L2RightHandSide.h>
#include <forward/RightHandSide.h>
#include <forward/AbstractEquation.h>

#include <tgmath.h>
#include <cmath>
#include <memory>

namespace wavepi {
namespace forward {

using namespace dealii;
using namespace wavepi::base;

template<int dim>
class WaveEquationBase {
public:
   // solvers for adjoint of L ∊ 𝓛(L²([0,T], L²(Ω)), L²([0,T], L²(Ω)))
   enum L2AdjointSolver {
      WaveEquationAdjoint, WaveEquationBackwards
   };

   virtual ~WaveEquationBase() = default;

   WaveEquationBase()
         : param_c(std::make_shared<Functions::ConstantFunction<dim>>(1.0, 1)),
               param_nu(std::make_shared<Functions::ZeroFunction<dim>>(1)),
               param_a(std::make_shared<Functions::ConstantFunction<dim>>(1.0, 1)),
               param_q(std::make_shared<Functions::ZeroFunction<dim>>(1)) {
   }

   // uses special functions for matrix assembly when discretized parameters are passed, which is a lot better for P1
   // elements. For P2 elements and 3 dimensions it actually turns out to be worse (too much coupling going on,
   // evaluating the polynomial is actually cheaper) in that case, you should turn of the specialization.
   inline bool is_special_assembly_recommended(std::shared_ptr<SpaceTimeMesh<dim>> mesh) const {
      return mesh->get_quadrature().size() < (1 << dim) || dim < 3;
   }

   inline bool using_special_assembly(std::shared_ptr<SpaceTimeMesh<dim>> mesh) {
      return special_assembly_tactic == 0 ? is_special_assembly_recommended(mesh) : (special_assembly_tactic > 0);
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

   void fill_A(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
         SparseMatrix<double> &destination);
   void fill_B(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
         SparseMatrix<double> &destination);
   void fill_C(std::shared_ptr<SpaceTimeMesh<dim>> mesh, DoFHandler<dim> &dof_handler,
         SparseMatrix<double> &destination);
protected:
   // treat DiscretizedFunctions as parameters and right hand side differently
   // < 0 -> no (better if much coupling present), > 0 -> yes, = 0 automatically (default)
   int special_assembly_tactic = 0;

   std::shared_ptr<Function<dim>> param_c, param_nu, param_a, param_q;

   // filled, if the function handles above can be typecast into DiscretizedFunction<dim>
   std::shared_ptr<DiscretizedFunction<dim>> param_c_disc = nullptr, param_nu_disc = nullptr;
   std::shared_ptr<DiscretizedFunction<dim>> param_a_disc = nullptr, param_q_disc = nullptr;
};

} /* namespace forward */
} /* namespace wavepi */

#endif /* LIB_FORWARD_WAVEEQUATIONBASE_H_ */
