/*
 * adjointness.cpp
 *
 *  Created on: 22.07.2017
 *      Author: thies
 */

#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <base/Util.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <forward/L2RightHandSide.h>
#include <forward/VectorRightHandSide.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>
#include <gtest/gtest.h>
#include <norms/H1L2.h>
#include <norms/L2L2.h>
#include <stddef.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

namespace {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::base;
using namespace wavepi;

template <int dim>
class TestF : public LightFunction<dim> {
 public:
  virtual ~TestF() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const {
    if (p.norm() < 0.5)
      return std::sin(t * 2 * numbers::PI);
    else
      return 0.0;
  }
};

template <int dim>
class TestG : public LightFunction<dim> {
 public:
  virtual ~TestG() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const {
    Point<dim> pc = Point<dim>::unit_vector(0);
    pc *= 0.5;

    return t * std::sin(p.distance(pc) * 2 * numbers::PI);
  }
};

template <int dim>
class TestH : public LightFunction<dim> {
 public:
  virtual ~TestH() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const { return p.norm() * t; }
};

template <int dim>
class TestC : public LightFunction<dim> {
 public:
  virtual ~TestC() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const { return p.norm() * t + 1.0; }
};

template <int dim>
class TestRho : public LightFunction<dim> {
 public:
  virtual ~TestRho() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const { return p.norm() + t + 1.0; }
};

template <int dim>
class TestNu : public LightFunction<dim> {
 public:
  virtual ~TestNu() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const {
    if (t > 1.0) return 0.0;

    return std::abs(p[0]) * t;
  }
};

template <int dim>
class TestQ : public LightFunction<dim> {
 public:
  virtual ~TestQ() = default;
  virtual double evaluate(const Point<dim> &p, const double t) const {
    return p.norm() < 0.5 ? std::sin(t / 2 * 2 * numbers::PI) : 0.0;
  }

  static const Point<dim> q_position;
};

template <>
const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template <>
const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template <>
const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

template <int dim>
void run_wave_adjoint_test(int fe_order, int quad_order, int refines, int n_steps,
                           typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver, bool trivial, double tol,
                           std::shared_ptr<std::ofstream> log = nullptr) {
  AssertThrow(adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint ||
                  adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards,
              ExcInternalError());

  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / (n_steps - 1);
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  std::shared_ptr<SpaceTimeMesh<dim>> mesh =
      std::make_shared<ConstantMesh<dim>>(times, FE_Q<dim>(fe_order), QGauss<dim>(quad_order), triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  WaveEquation<dim> wave_eq(mesh);
  if (!trivial) {
    wave_eq.set_param_rho(std::make_shared<TestRho<dim>>());
    wave_eq.set_param_c(std::make_shared<TestC<dim>>());
    wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
    wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());
  }

  WaveEquationAdjoint<dim> wave_eq_adj(wave_eq);

  bool use_adj   = adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint;
  double err_avg = 0.0;
  double err_simple;

  for (size_t i = 0; i < 1 + 5; i++) {
    std::shared_ptr<DiscretizedFunction<dim>> f, g;

    if (i == 0) {
      TestF<dim> f_cont;
      f = std::make_shared<DiscretizedFunction<dim>>(mesh, f_cont);

      TestG<dim> g_cont;
      g = std::make_shared<DiscretizedFunction<dim>>(mesh, g_cont);
    } else {
      f = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(mesh));
      g = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(mesh));

      if (n_steps > 7) {
        // make f and g a slightly smoother, random noise might be a too harsh

        f->set_norm(std::make_shared<norms::H1L2<dim>>(0.5));
        f->dot_transform_inverse();

        g->set_norm(std::make_shared<norms::H1L2<dim>>(0.5));
        g->dot_transform_inverse();
      }
    }

    f->set_norm(std::make_shared<norms::L2L2<dim>>());
    *f *= 1.0 / f->norm();

    g->set_norm(std::make_shared<norms::L2L2<dim>>());
    *g *= 1.0 / g->norm();

    DiscretizedFunction<dim> sol_f = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(f), WaveEquation<dim>::Forward);
    sol_f.throw_away_derivative();
    sol_f.set_norm(std::make_shared<norms::L2L2<dim>>());
    EXPECT_GT(sol_f.norm(), 0.0);

    auto g_time_mass = std::make_shared<DiscretizedFunction<dim>>(*g);
    g_time_mass->set_norm(std::make_shared<norms::L2L2<dim>>());

    DiscretizedFunction<dim> adj_g(mesh);
    if (use_adj) {
      // if L2RightHandSide would be used, we would also need a call to solve_mass
      g_time_mass->dot_transform();

      adj_g = wave_eq_adj.run(std::make_shared<VectorRightHandSide<dim>>(g_time_mass));

      // wave_eq_adj does everything except the multiplication with the mass matrix (to allow for optimization)
      adj_g.set_norm(std::make_shared<norms::L2L2<dim>>());
      adj_g.dot_mult_mass_and_transform_inverse();
    } else {
      // dot_transforms not needed here, wave_eq backwards should be the L^2([0,T], L^2)-Adjoint

      adj_g = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(g_time_mass), WaveEquation<dim>::Backward);
      adj_g.throw_away_derivative();
      adj_g.set_norm(std::make_shared<norms::L2L2<dim>>());
    }

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

    // deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
    //        << ", rel. error = " << fg_err << std::endl;

    // EXPECT_LT(zz_err, tol);
  }

  deallog << std::scientific << "average relative error for random f,g = " << err_avg << std::endl;

  double h = dealii::GridTools::maximal_cell_diameter(*mesh->get_triangulation(0));

  if (log)
    *log << std::scientific << mesh->length() << " " << dt << " " << refines << " " << h << " " << err_avg << std::endl;

  EXPECT_LT(err_simple, tol);
  EXPECT_LT(err_avg, tol);
}
}  // namespace

// TEST(WaveEquationAdjointness, Adjoint1DFE1) {
//   for (int i = 3; i < 10; i++)
//     run_wave_adjoint_test<1>(1, 5, 6, 1 << i, WaveEquationBase<1>::WaveEquationAdjoint, false, 1e-4);
// }

// TEST(WaveEquationAdjointness, Adjoint1DFE2) {
//   for (int i = 3; i < 10; i++)
//     run_wave_adjoint_test<1>(2, 5, 4, 1 << i, WaveEquationBase<1>::WaveEquationAdjoint, false, 1e-4);
// }

TEST(WaveEquationAdjointness, Backwards2DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_Backwards2DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 5, 5, 1 << i, WaveEquationBase<2>::WaveEquationBackwards, false, 1e+1, f);
}

TEST(WaveEquationAdjointness, BackwardsTrivial2DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_BackwardsTrivial2DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 5, 5, 1 << i, WaveEquationBase<2>::WaveEquationBackwards, true, 1e+1, f);
}

TEST(WaveEquationAdjointness, Adjoint2DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_Adjoint2DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 5, 5, 1 << i, WaveEquationBase<2>::WaveEquationAdjoint, false, 1e-4, f);
}

TEST(WaveEquationAdjointness, AdjointTrivial2DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_AdjointTrivial2DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 5, 5, 1 << i, WaveEquationBase<2>::WaveEquationAdjoint, true, 1e-4, f);
}

// TEST(WaveEquationAdjointness, Adjoint2DFE2) {
//   auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_Adjoint2DFE2.dat", std::ios_base::trunc);
//   ASSERT_TRUE(*f) << "could not open file for output";

//   for (int i = 3; i < 10; i++)
//     run_wave_adjoint_test<2>(2, 5, 5, 1 << i, WaveEquationBase<2>::WaveEquationAdjoint, false, 1e-4, f);
// }

TEST(WaveEquationAdjointness, Adjoint3DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_Adjoint3DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 5, 3, 1 << i, WaveEquationBase<3>::WaveEquationAdjoint, false, 1e-4, f);
}

TEST(WaveEquationAdjointness, Backwards3DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_Backwards3DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 5, 3, 1 << i, WaveEquationBase<3>::WaveEquationBackwards, false, 1e+1, f);
}

TEST(WaveEquationAdjointness, AdjointTrivial3DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_AdjointTrivial3DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 5, 3, 1 << i, WaveEquationBase<3>::WaveEquationAdjoint, true, 1e-4, f);
}

TEST(WaveEquationAdjointness, BackwardsTrivial3DFE1) {
  auto f = std::make_shared<std::ofstream>("./WaveEquationAdjointness_BackwardsTrivial3DFE1.dat", std::ios_base::trunc);
  ASSERT_TRUE(*f) << "could not open file for output";

  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 5, 3, 1 << i, WaveEquationBase<3>::WaveEquationBackwards, true, 1e+1, f);
}