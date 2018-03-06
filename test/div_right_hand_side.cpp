/*
 * adjointness.cpp
 *
 *  Created on: 22.07.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/MacroFunctionParser.h>
#include <base/SpaceTimeMesh.h>
#include <base/Transformation.h>
#include <base/Tuple.h>
#include <base/Util.h>
#include <forward/L2RightHandSide.h>
#include <forward/VectorRightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>
#include <measurements/FieldMeasure.h>
#include <measurements/Measure.h>
#include <problems/QProblem.h>
#include <problems/WaveProblem.h>

#include <gtest/gtest.h>

#include <stddef.h>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::base;
using namespace wavepi::problems;
using namespace wavepi::measurements;

template <int dim>
class TestF : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    if (p.norm() < 0.5)
      return std::sin(this->get_time() * 2 * numbers::PI);
    else
      return 0.0;
  }
};

template <int dim>
class TestG : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    Point<dim> pc = Point<dim>::unit_vector(0);
    pc *= 0.5;

    return this->get_time() * std::sin(p.distance(pc) * 2 * numbers::PI);
  }
};

template <int dim>
void run_div_rhs_adjoint_test(int fe_order, int quad_order, int refines, int n_steps, double tol) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  std::shared_ptr<SpaceTimeMesh<dim>> mesh =
      std::make_shared<ConstantMesh<dim>>(times, FE_Q<dim>(fe_order), QGauss<dim>(quad_order), triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;
  double err_avg = 0.0;
  double err_simple;

  for (size_t i = 0; i < 11; i++) {
    std::shared_ptr<DiscretizedFunction<dim>> f, g;

    if (i == 0) {
      TestF<dim> f_cont;
      f = std::make_shared<DiscretizedFunction<dim>>(mesh, f_cont);

      TestG<dim> g_cont;
      g = std::make_shared<DiscretizedFunction<dim>>(mesh, g_cont);
    } else {
      f = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(mesh));

      // make it a bit smoother, random noise might be a bit too harsh
      f->set_norm(Norm::H1L2);
      f->dot_transform_inverse();

      g = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(mesh));

      // make it a bit smoother, random noise might be a bit too harsh
      g->set_norm(Norm::H1L2);
      g->dot_transform_inverse();
    }

    f->set_norm(Norm::Coefficients);
    *f *= 1.0 / f->norm();

    g->set_norm(Norm::Coefficients);
    *g *= 1.0 / g->norm();

    DivRightHandSide<dim> divrhs(f, f);
    DivRightHandSideAdjoint<dim> divrhs_adj(f, g);

    DiscretizedFunction<dim> sol_f(mesh, false, Norm::Coefficients);

    for (size_t i = 0; i < mesh->length(); i++) {
      divrhs.set_time(mesh->get_times()[i]);
      divrhs.create_right_hand_side(*mesh->get_dof_handler(i), mesh->get_quadrature(), sol_f[i]);
    }

    EXPECT_GT(sol_f.norm(), 0.0);

    auto adj_g = divrhs_adj.run_adjoint(mesh);
    adj_g.set_norm(Norm::Coefficients);
    EXPECT_GT(adj_g.norm(), 0.0);

    double dot_solf_g = sol_f * (*g);
    double dot_f_adjg = (*f) * adj_g;
    double fg_err     = std::abs(dot_solf_g - dot_f_adjg) / (std::abs(dot_solf_g) + 1e-300);

    if (i == 0) {
      // deallog << "simple f,g: " << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
      //         << std::endl;
      err_simple = fg_err;
      deallog << std::scientific << "        relative error for simple f,g = " << fg_err << std::endl;
    } else
      err_avg = ((i - 1) * err_avg + fg_err) / i;

    deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
            << ", rel. error = " << fg_err << std::endl;

    // EXPECT_LT(zz_err, tol);
  }

  deallog << std::scientific << "average relative error for random f,g = " << err_avg << std::endl;
  EXPECT_LT(err_simple, tol);
  EXPECT_LT(err_avg, tol);
}
}  // namespace

TEST(DivRightHandSide, Adjoint1DFE1) {
  for (int i = 3; i < 10; i++)
    run_div_rhs_adjoint_test<1>(1, 3, 6, 1 << i, 1e-1);
}

TEST(DivRightHandSide, Adjoint1DFE2) {
  for (int i = 3; i < 10; i++)
    run_div_rhs_adjoint_test<1>(2, 6, 4, 1 << i, 1e-1);
}
TEST(DivRightHandSide, Adjoint2DFE1) {
  for (int i = 3; i < 10; i++)
    run_div_rhs_adjoint_test<2>(1, 3, 5, 1 << i, 1e-1);
}

TEST(DivRightHandSide, Adjoint3DFE1) {
  for (int i = 3; i < 9; i++)
    run_div_rhs_adjoint_test<3>(1, 3, 2, 1 << i, 1e-1);
}
