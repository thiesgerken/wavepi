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
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <forward/L2RightHandSide.h>
#include <forward/WaveEquation.h>
#include <gtest/gtest.h>
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
    double tmp = tmp2 * (sin(t) - (1 + t) * cos(t)) / ((1 + t) * (1 + t) * (1 + t));

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
void run_reference_test2(std::shared_ptr<SpaceTimeMesh<dim>> mesh, int refines, bool expect = true, bool save = false,
                         std::shared_ptr<std::ofstream> log = nullptr, int precondition = -1) {
  deallog << std::endl << "----------  n_dofs(0): " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << mesh->get_times().size() << "  ----------" << std::endl;

  WaveEquation<dim> wave_eq(mesh);
  wave_eq.set_precondition_max_age(precondition);

  wave_eq.set_param_rho(std::make_shared<TestRho<dim>>());
  wave_eq.set_param_c(std::make_shared<TestC<dim>>());

  auto u_cont = std::make_shared<TestU<dim>>();
  auto v_cont = std::make_shared<TestV<dim>>();

  wave_eq.set_initial_values_u(u_cont);
  wave_eq.set_initial_values_v(v_cont);

  Timer timer;
  timer.start();
  DiscretizedFunction<dim> solu = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()));
  timer.stop();

  DiscretizedFunction<dim> solv = solu.derivative();
  solu.throw_away_derivative();

  solu.set_norm(std::make_shared<norms::L2L2<dim>>());
  solv.set_norm(std::make_shared<norms::L2L2<dim>>());

  double err_u       = norms::L2L2<dim>::absolute_error(solu, *u_cont);
  err_u /= solu.norm();

  double err_v       = norms::L2L2<dim>::absolute_error(solv, *v_cont);
  err_v /= solv.norm();

  DiscretizedFunction<dim> u_disc(mesh, *u_cont);
  u_disc.set_norm(std::make_shared<norms::L2L2<dim>>());
  DiscretizedFunction<dim> tmp(solu);
  tmp -= u_disc;
  double err_u_disc = tmp.norm() / u_disc.norm();

  DiscretizedFunction<dim> v_disc(mesh, *v_cont);
  v_disc.set_norm(std::make_shared<norms::L2L2<dim>>());
  tmp = solv;
  tmp -= v_disc;
  double err_v_disc = tmp.norm() / v_disc.norm();

  if (expect) {
    EXPECT_LT(err_u, 1e-1);
    EXPECT_LT(err_v, 1e-1);
  }

  if (save) {
    DiscretizedFunction<dim> u_disc2(mesh, *u_cont);

    DiscretizedFunction<dim> tmp(solu);
    tmp -= u_disc2;

    u_disc2.write_pvd("./", "refu", "uref");
    solu.write_pvd("./", "solu", "u");
    tmp.write_pvd("./", "diff", "udiff");
  }

  deallog << std::scientific << " rerr(u) = " << err_u << ", rerr(v) = " << err_v << ", rerr_disc(u) = " << err_u_disc << ", rerr_disc(v) = " << err_v_disc << ", cpu time = " << std::fixed << std::setprecision(2) << timer.cpu_time() << std::endl;

  double dt = mesh->get_time(1) - mesh->get_time(0);
  double h  = dealii::GridTools::maximal_cell_diameter(*mesh->get_triangulation(0));

  if (log)
    *log << std::scientific << mesh->length() << " " << dt << " " << refines << " " << h << " " << err_u << " " << err_v
         << " " << std::fixed << std::setprecision(2) << timer.cpu_time() << std::endl;
}

template <int dim>
void run_reference_test2_constant(int fe_order, int quad_order, int refines, int steps, bool expect = true,
                                  bool save = false, std::shared_ptr<std::ofstream> log = nullptr,
                                  int precondition = -1) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_end   = 3.0;
  double t_start = 0.0, dt = t_end / (steps - 1);
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  return run_reference_test2<dim>(mesh, refines, expect, save, log, precondition);
}

TEST(WaveEquation, ReferenceTestParameters2DFE1) {
  auto file_time = std::make_shared<std::ofstream>("./ReferenceTestParameters2DFE1_time.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_time) << "could not open file for output";

  for (int steps = 6; steps <= 128; steps = (int)(steps * 1.41))
    run_reference_test2_constant<2>(1, 5, 8, steps, steps >= 64, false, file_time);
  file_time->close();

  auto file_space = std::make_shared<std::ofstream>("./ReferenceTestParameters2DFE1_space.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_space) << "could not open file for output";

  for (int refine = 1; refine <= 8; refine++)
    run_reference_test2_constant<2>(1, 5, refine, 128, refine >= 4, false, file_space);
  file_space->close();
}

TEST(WaveEquation, ReferenceTestParameters2DFE2) {
  auto file_space = std::make_shared<std::ofstream>("./ReferenceTestParameters2DFE2_space.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_space) << "could not open file for output";

  for (int refine = 1; refine <= 5; refine++)
    run_reference_test2_constant<2>(2, 5, refine, 1024, refine >= 4, false, file_space);
  file_space->close();
}

TEST(WaveEquation, ReferenceTestParameters3DFE1) {
  auto file_time = std::make_shared<std::ofstream>("./ReferenceTestParameters3DFE1_time.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_time) << "could not open file for output";

  for (int steps = 4; steps <= 15; steps = (int)(steps * 1.41))
    run_reference_test2_constant<3>(1, 5, 5, steps, steps >= 64, false, file_time);
  file_time->close();

  auto file_space = std::make_shared<std::ofstream>("./ReferenceTestParameters3DFE1_space.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_space) << "could not open file for output";

  for (int refine = 1; refine <= 5; refine++)
    run_reference_test2_constant<3>(1, 5, refine, 128, refine >= 4, false, file_space);
  file_space->close();
}

TEST(WaveEquation, ReferenceTestParameters3DFE2) {
  auto file_space = std::make_shared<std::ofstream>("./ReferenceTestParameters3DFE2_space.dat", std::ios_base::trunc);
  ASSERT_TRUE(*file_space) << "could not open file for output";

  for (int refine = 1; refine <= 4; refine++)
    run_reference_test2_constant<3>(2, 5, refine, 128, refine >= 4, false, file_space);
  file_space->close();
}

TEST(WaveEquation, PreconditionTest2DFE1) {
  for (int i = 0; i <= 9; i++) {
    int max_age;
    if (i == 0)
      max_age = -1;
    else if (i == 1)
      max_age = 0;
    else
      max_age = 1 << (i - 2);

    run_reference_test2_constant<2>(1, 5, 6, 128, false, false, nullptr, max_age);
    deallog << " precon = " << max_age << std::endl;
  }
}

TEST(WaveEquation, PreconditionTest2DFE2) {
  for (int i = 0; i <= 9; i++) {
    int max_age;
    if (i == 0)
      max_age = -1;
    else if (i == 1)
      max_age = 0;
    else
      max_age = 1 << (i - 2);

    run_reference_test2_constant<2>(2, 5, 6, 128, false, false, nullptr, max_age);
    deallog << " precon = " << max_age << std::endl;
  }
}
}  // namespace