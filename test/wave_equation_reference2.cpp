/*
 * wave_equation_reference2.cpp
 *
 *  Created on: 27.05.2019
 *      Author: thies
 */

#include <base/AdaptiveMesh.h>
#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
#include <base/Util.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <gtest/gtest.h>
#include <norms/L2L2.h>
#include <stddef.h>
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
    double tmp2 = 1;
    for (size_t i = 0; i < dim; i++)
      tmp2 *= sin(p[i]);

    // 1/rho (u'/c^2)'
    double tmp =
        tmp2 * (sin(t) - (1 + t) * cos(t)) / ((1 + t) * (1 + t) * (1 + t) * (1 + p[0]) * (1 + p[0]) * (1 + p[1]));

    // div (nabla u / rho)

    double fac = cos(t) / ((1 + t) * (1 + p[1]));
    tmp -= -dim * fac * tmp2;

    double tmp3 = cos(p[1]);
    for (size_t i = 0; i < dim; i++)
      if (i != 1) tmp3 *= sin(p[i]);

    tmp -= -1 * fac * tmp3 / (1 + p[1]);

    return tmp;
  }
};

template <int dim>
class TestFOnlyTime : public LightFunction<dim> {
 public:
  virtual ~TestFOnlyTime() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const {
    double tmp2 = 1;
    for (size_t i = 0; i < dim; i++)
      tmp2 *= sin(p[i]);

    // 1/rho (u'/c^2)'
    double tmp =
        tmp2 * (sin(t) - (1 + t) * cos(t)) / ((1 + t) * (1 + t) * (1 + t));

    // div (nabla u / rho)

    double fac = cos(t) / (1 + t);
    tmp -= -dim * fac * tmp2;

    return tmp;
  }
};

template <int dim>
class TestC : public LightFunction<dim> {
 public:
  virtual ~TestC() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const { return sqrt(1 + t) * (1 + p[0]); }
};

template <int dim>
class TestRho : public LightFunction<dim> {
 public:
  virtual ~TestRho() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const { return (1 + t) * (1 + p[1]); }
};

template <int dim>
class TestCOnlyTime : public LightFunction<dim> {
 public:
  virtual ~TestCOnlyTime() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const { return sqrt(1 + t); }
};

template <int dim>
class TestRhoOnlyTime : public LightFunction<dim> {
 public:
  virtual ~TestRhoOnlyTime() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const { return 1 + t; }
};

template <int dim>
class TestU : public LightFunction<dim> {
 public:
  virtual ~TestU() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const {
    double tmp = 1;

    for (size_t i = 0; i < dim; i++)
      tmp *= sin(p[i]);

    return cos(t) * tmp;
  }
};

template <int dim>
class TestV : public LightFunction<dim> {
 public:
  virtual ~TestV() = default;

  virtual double evaluate(const Point<dim> &p, const double t) const {
    double tmp = 1;

    for (size_t i = 0; i < dim; i++)
      tmp *= sin(p[i]);

    return -sin(t) * tmp;
  }
};

template <int dim>
void run_reference_test2(std::shared_ptr<SpaceTimeMesh<dim>> mesh, bool expect = true, bool save = false) {
  deallog << std::endl << "----------  n_dofs(0): " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << mesh->get_times().size() << "  ----------" << std::endl;

  WaveEquation<dim> wave_eq(mesh);

  wave_eq.set_param_rho(std::make_shared<TestRho<dim>>());
  wave_eq.set_param_c(std::make_shared<TestC<dim>>());

  auto u_cont = std::make_shared<TestU<dim>>();
  auto v_cont = std::make_shared<TestV<dim>>();

  wave_eq.set_initial_values_u(u_cont);
  wave_eq.set_initial_values_v(v_cont);

  DiscretizedFunction<dim> solu = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
  DiscretizedFunction<dim> solv = solu.derivative();
  solu.throw_away_derivative();

  solu.set_norm(std::make_shared<norms::L2L2<dim>>());
  solv.set_norm(std::make_shared<norms::L2L2<dim>>());

  DiscretizedFunction<dim> u_disc(mesh, *u_cont);
  DiscretizedFunction<dim> v_disc(mesh, *v_cont);

  u_disc.set_norm(std::make_shared<norms::L2L2<dim>>());
  v_disc.set_norm(std::make_shared<norms::L2L2<dim>>());

  DiscretizedFunction<dim> tmp(solu);
  tmp -= u_disc;
  double err_u = tmp.norm() / u_disc.norm();

  tmp = solv;
  tmp -= v_disc;
  double err_v = tmp.norm() / v_disc.norm();

  if (expect) {
    EXPECT_LT(err_u, 1e-1);
    EXPECT_LT(err_v, 1e-1);
  }

  if (save) {
    solu.write_pvd("./", "solu", "u");
    u_disc.write_pvd("./", "refu", "uref");

    DiscretizedFunction<dim> tmp(solu);
    tmp -= u_disc;
    tmp.write_pvd("./", "diff", "udiff");
  }

  deallog << std::scientific << " rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl;
}
}  // namespace

template <int dim>
void run_reference_test2_constant(int fe_order, int quad_order, int refines, int steps, bool expect = true,
                                  bool save = false) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_end   = 3.0;
  double t_start = 0.0, dt = t_end / (steps-1);
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  run_reference_test2<dim>(mesh, expect, save);
}

TEST(WaveEquation, ReferenceTestParameters2DFE1) {
//   for (int steps = 6; steps <= 128; steps = (int) (steps * 1.41))
   //  run_reference_test2_constant<2>(1, 4, 7, steps, steps >= 64);

  for (int refine = 1; refine <= 8; refine++)
    run_reference_test2_constant<2>(1, 4, refine, 128, false);
}

TEST(WaveEquation, ReferenceTestParameters3DFE1) {
  for (int steps = 8; steps <= 32; steps *= 2)
    run_reference_test2_constant<3>(1, 3, 3, steps, steps >= 32);

  for (int refine = 2; refine >= 0; refine--)
    run_reference_test2_constant<3>(1, 3, refine, 32, false);
}
