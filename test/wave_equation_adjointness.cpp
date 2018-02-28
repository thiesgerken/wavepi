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
class TestF2 : public Function<dim> {
 public:
  TestF2() : Function<dim>() {}
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if ((this->get_time() <= 0.5) && (p.distance(actor_position) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
    else
      return 0.0;
  }

 private:
  static const Point<dim> actor_position;
};

template <>
const Point<1> TestF2<1>::actor_position = Point<1>(1.0);
template <>
const Point<2> TestF2<2>::actor_position = Point<2>(1.0, 0.5);
template <>
const Point<3> TestF2<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

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
class TestH : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return p.norm() * this->get_time();
  }
};

template <int dim>
double rho(const Point<dim> &p, double t) {
  return p.norm() + t + 1.0;
}

template <int dim>
double c_squared(const Point<dim> &p, double t) {
  double tmp = p.norm() * t + 1.0;

  return tmp * tmp;
}

template <int dim>
class TestC : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return 1.0 / (rho(p, this->get_time()) * c_squared(p, this->get_time()));
  }
};

template <int dim>
class TestA : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return 1.0 / rho(p, this->get_time());
  }
};

template <int dim>
class TestNu : public Function<dim> {
 public:
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    if (this->get_time() > 1.0) return 0.0;

    return std::abs(p[0]) * this->get_time();
  }
};

template <int dim>
class TestQ : public Function<dim> {
 public:
  TestQ() : Function<dim>() {}
  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));

    return p.norm() < 0.5 ? std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
  }

  static const Point<dim> q_position;
};

template <>
const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template <>
const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template <>
const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

// template <int dim>
// void run_l2_q_adjoint_test(int fe_order, int quad_order, int refines, int n_steps,
//                           typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver, bool set_nu, double tol) {
//  AssertThrow(adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint ||
//                  adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards,
//              ExcInternalError());
//
//  auto triangulation = std::make_shared<Triangulation<dim>>();
//  GridGenerator::hyper_cube(*triangulation, -1, 1);
//  Util::set_all_boundary_ids(*triangulation, 0);
//  triangulation->refine_global(refines);
//
//  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
//  std::vector<double> times;
//
//  for (size_t i = 0; t_start + i * dt <= t_end; i++)
//    times.push_back(t_start + i * dt);
//
//  std::shared_ptr<SpaceTimeMesh<dim>> mesh =
//      std::make_shared<ConstantMesh<dim>>(times, FE_Q<dim>(fe_order), QGauss<dim>(quad_order), triangulation);
//
//  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
//  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;
//
//  WaveEquation<dim> wave_eq(mesh);
//  wave_eq.set_param_a(std::make_shared<TestA<dim>>());
//  wave_eq.set_param_c(std::make_shared<TestC<dim>>());
//  wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
//
//  if (set_nu) wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());
//
//  WaveEquationAdjoint<dim> wave_eq_adj(wave_eq);
//
//  bool use_adj = adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint;
//
//  TestF<dim> f_cont;
//  auto f = std::make_shared<DiscretizedFunction<dim>>(mesh, f_cont);
//  f->set_norm(Norm::L2L2);
//
//  wave_eq.set_run_direction(WaveEquation<dim>::Forward);
//  wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
//  DiscretizedFunction<dim> sol_f = wave_eq.run();
//  sol_f.set_norm(Norm::L2L2);
//  EXPECT_GT(sol_f.norm(), 0.0);
//
//  auto f_time_mass = std::make_shared<DiscretizedFunction<dim>>(*f);
//  f_time_mass->set_norm(Norm::L2L2);
//  f_time_mass->dot_solve_mass_and_transform();
//
//  DiscretizedFunction<dim> adj_f(mesh);
//  if (use_adj) {
//    wave_eq_adj.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f_time_mass));
//    adj_f = wave_eq_adj.run();
//  } else {
//    wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f_time_mass));
//    wave_eq.set_run_direction(WaveEquation<dim>::Backward);
//    adj_f = wave_eq.run();
//    adj_f.throw_away_derivative();
//  }
//
//  adj_f.set_norm(Norm::L2L2);
//  adj_f.dot_mult_mass_and_transform_inverse();
//  EXPECT_GT(adj_f.norm(), 0.0);
//
//  TestG<dim> g_cont;
//  auto g = std::make_shared<DiscretizedFunction<dim>>(mesh, g_cont);
//  g->set_norm(Norm::L2L2);
//
//  wave_eq.set_run_direction(WaveEquation<dim>::Forward);
//  wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g));
//  DiscretizedFunction<dim> sol_g = wave_eq.run();
//  sol_g.set_norm(Norm::L2L2);
//  EXPECT_GT(sol_g.norm(), 0.0);
//
//  auto g_time_mass = std::make_shared<DiscretizedFunction<dim>>(*g);
//  g_time_mass->set_norm(Norm::L2L2);
//  g_time_mass->dot_solve_mass_and_transform();
//
//  DiscretizedFunction<dim> adj_g(mesh);
//  if (use_adj) {
//    wave_eq_adj.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g_time_mass));
//    adj_g = wave_eq_adj.run();
//  } else {
//    wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g_time_mass));
//    wave_eq.set_run_direction(WaveEquation<dim>::Backward);
//    adj_g = wave_eq.run();
//    adj_g.throw_away_derivative();
//  }
//
//  adj_g.set_norm(Norm::L2L2);
//  adj_g.dot_mult_mass_and_transform_inverse();
//  EXPECT_GT(adj_g.norm(), 0.0);
//
//  auto z = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(*g, 1));
//  z->set_norm(Norm::L2L2);
//
//  wave_eq.set_run_direction(WaveEquation<dim>::Forward);
//  wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(z));
//  DiscretizedFunction<dim> sol_z = wave_eq.run();
//  sol_z.set_norm(Norm::L2L2);
//  EXPECT_GT(sol_z.norm(), 0.0);
//
//  auto z_time_mass = std::make_shared<DiscretizedFunction<dim>>(*z);
//  z_time_mass->set_norm(Norm::L2L2);
//  z_time_mass->dot_solve_mass_and_transform();
//
//  DiscretizedFunction<dim> adj_z(mesh);
//  if (use_adj) {
//    wave_eq_adj.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(z_time_mass));
//    adj_z = wave_eq_adj.run();
//  } else {
//    wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(z_time_mass));
//    wave_eq.set_run_direction(WaveEquation<dim>::Backward);
//    adj_z = wave_eq.run();
//    adj_z.throw_away_derivative();
//  }
//
//  adj_z.set_norm(Norm::L2L2);
//  adj_z.dot_mult_mass_and_transform_inverse();
//  EXPECT_GT(adj_z.norm(), 0.0);
//
//  double dot_solf_f = sol_f * (*f);
//  double dot_f_adjf = (*f) * adj_f;
//  double ff_err     = std::abs(dot_solf_f - dot_f_adjf) / (std::abs(dot_solf_f) + 1e-300);
//
//  deallog << std::scientific << "(Lf, f) = " << dot_solf_f << ", (f, L*f) = " << dot_f_adjf
//          << ", rel. error = " << ff_err << std::endl;
//
//  double dot_solg_g = sol_g * (*g);
//  double dot_g_adjg = (*g) * adj_g;
//  double gg_err     = std::abs(dot_solg_g - dot_g_adjg) / (std::abs(dot_solg_g) + 1e-300);
//
//  deallog << std::scientific << "(Lg, g) = " << dot_solg_g << ", (g, L*g) = " << dot_g_adjg
//          << ", rel. error = " << gg_err << std::endl;
//
//  double dot_solg_f = sol_g * (*f);
//  double dot_g_adjf = (*g) * adj_f;
//  double gf_err     = std::abs(dot_solg_f - dot_g_adjf) / (std::abs(dot_solg_f) + 1e-300);
//
//  deallog << std::scientific << "(Lg, f) = " << dot_solg_f << ", (g, L*f) = " << dot_g_adjf
//          << ", rel. error = " << gf_err << std::endl;
//
//  double dot_solf_g = sol_f * (*g);
//  double dot_f_adjg = (*f) * adj_g;
//  double fg_err     = std::abs(dot_solf_g - dot_f_adjg) / (std::abs(dot_solf_g) + 1e-300);
//
//  deallog << std::scientific << "(Lf, g) = " << dot_solf_g << ", (f, L*g) = " << dot_f_adjg
//          << ", rel. error = " << fg_err << std::endl;
//
//  double dot_solz_z = sol_z * (*z);
//  double dot_z_adjz = (*z) * adj_z;
//  double zz_err     = std::abs(dot_solz_z - dot_z_adjz) / (std::abs(dot_solz_z) + 1e-300);
//
//  deallog << std::scientific << "(Lz, z) = " << dot_solz_z << ", (z, L*z) = " << dot_z_adjz
//          << ", rel. error = " << zz_err << std::endl
//          << std::endl;
//
//  DiscretizedFunction<dim> u(sol_z);  // sol_f
//
//  DiscretizedFunction<dim> mf(*f);
//  mf.pointwise_multiplication(u);
//
//  DiscretizedFunction<dim> mg(*g);
//  mg.pointwise_multiplication(u);
//
//  DiscretizedFunction<dim> mz(*z);
//  mz.pointwise_multiplication(u);
//
//  DiscretizedFunction<dim> madj_f(*f);
//  madj_f.dot_transform();
//  madj_f.pointwise_multiplication(u);
//  madj_f.dot_transform_inverse();
//  EXPECT_GT(madj_f.norm(), 0.0);
//
//  DiscretizedFunction<dim> madj_g(*g);
//  madj_g.dot_transform();
//  madj_g.pointwise_multiplication(u);
//  madj_g.dot_transform_inverse();
//  EXPECT_GT(madj_g.norm(), 0.0);
//
//  DiscretizedFunction<dim> madj_z(*z);
//  madj_z.dot_transform();
//  madj_z.pointwise_multiplication(u);
//  madj_z.dot_transform_inverse();
//  EXPECT_GT(madj_z.norm(), 0.0);
//
//  double dot_mulf_f  = mf * (*f);
//  double dot_f_madjf = (*f) * madj_f;
//  double mff_err     = std::abs(dot_mulf_f - dot_f_madjf) / (std::abs(dot_mulf_f) + 1e-300);
//
//  deallog << std::scientific << "(Mf, f) = " << dot_mulf_f << ", (f, M*f) = " << dot_f_madjf
//          << ", rel. error = " << mff_err << std::endl;
//
//  double dot_mulg_g  = mg * (*g);
//  double dot_g_madjg = (*g) * madj_g;
//  double mgg_err     = std::abs(dot_mulg_g - dot_g_madjg) / (std::abs(dot_mulg_g) + 1e-300);
//
//  deallog << std::scientific << "(Mg, g) = " << dot_mulg_g << ", (g, M*g) = " << dot_g_madjg
//          << ", rel. error = " << mgg_err << std::endl;
//
//  double dot_mulg_f  = mg * (*f);
//  double dot_g_madjf = (*g) * madj_f;
//  double mgf_err     = std::abs(dot_mulg_f - dot_g_madjf) / (std::abs(dot_mulg_f) + 1e-300);
//
//  deallog << std::scientific << "(Mg, f) = " << dot_mulg_f << ", (g, M*f) = " << dot_g_madjf
//          << ", rel. error = " << mgf_err << std::endl;
//
//  double dot_mulf_g  = mf * (*g);
//  double dot_f_madjg = (*f) * madj_g;
//  double mfg_err     = std::abs(dot_mulf_g - dot_f_madjg) / (std::abs(dot_mulf_g) + 1e-300);
//
//  deallog << std::scientific << "(Mf, g) = " << dot_mulf_g << ", (f, M*g) = " << dot_f_madjg
//          << ", rel. error = " << mfg_err << std::endl;
//
//  double dot_mulz_z  = mz * (*z);
//  double dot_z_madjz = (*z) * madj_z;
//  double mzz_err     = std::abs(dot_mulz_z - dot_z_madjz) / (std::abs(dot_mulz_z) + 1e-300);
//
//  deallog << std::scientific << "(Mz, z) = " << dot_mulz_z << ", (z, M*z) = " << dot_z_madjz
//          << ", rel. error = " << mzz_err << std::endl
//          << std::endl;
//
//  // test concatenation of both (if they are implemented as above)
//  DiscretizedFunction<dim> estimate(mesh);
//  estimate.set_norm(Norm::L2L2);
//
//  std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> measures;
//  measures.push_back(std::make_shared<FieldMeasure<dim>>(mesh, Norm::L2L2));
//
//  std::map<std::string, double> consts;
//  std::vector<std::shared_ptr<Function<dim>>> pulses;
//  pulses.push_back(std::make_shared<MacroFunctionParser<dim>>("if(norm{x|y|z} < 0.2, sin(t), 0.0)", consts));
//
//  QProblem<dim, DiscretizedFunction<dim>> problem(wave_eq, pulses, measures,
//  std::make_shared<IdentityTransform<dim>>(),
//                                                  adjoint_solver);
//  auto data_current = problem.forward(estimate);
//  auto A            = problem.derivative(estimate);
//
//  Tuple<DiscretizedFunction<dim>> Tf(*f);
//  Tuple<DiscretizedFunction<dim>> Tg(*g);
//  Tuple<DiscretizedFunction<dim>> Tz(*z);
//
//  auto Af(A->forward(*f));
//  auto Aadjf(A->adjoint(Tf));
//  EXPECT_GT(Af.norm(), 0.0);
//  EXPECT_GT(Aadjf.norm(), 0.0);
//
//  auto Ag(A->forward(*g));
//  auto Aadjg(A->adjoint(Tg));
//  EXPECT_GT(Ag.norm(), 0.0);
//  EXPECT_GT(Aadjg.norm(), 0.0);
//
//  auto Az(A->forward(*z));
//  auto Aadjz(A->adjoint(Tz));
//  EXPECT_GT(Az.norm(), 0.0);
//  EXPECT_GT(Aadjz.norm(), 0.0);
//
//  double dot_Af_f    = Af * (Tf);
//  double dot_f_Aadjf = (*f) * Aadjf;
//  double Aff_err     = std::abs(dot_Af_f - dot_f_Aadjf) / (std::abs(dot_Af_f) + 1e-300);
//
//  deallog << std::scientific << "(Af, f) = " << dot_Af_f << ", (f, A*f) = " << dot_f_Aadjf
//          << ", rel. error = " << Aff_err << std::endl;
//
//  double dot_Ag_g    = Ag * (Tg);
//  double dot_g_Aadjg = (*g) * Aadjg;
//  double Agg_err     = std::abs(dot_Ag_g - dot_g_Aadjg) / (std::abs(dot_Ag_g) + 1e-300);
//
//  deallog << std::scientific << "(Ag, g) = " << dot_Ag_g << ", (g, A*g) = " << dot_g_Aadjg
//          << ", rel. error = " << Agg_err << std::endl;
//
//  double dot_Ag_f    = Ag * (Tf);
//  double dot_g_Aadjf = (*g) * Aadjf;
//  double Agf_err     = std::abs(dot_Ag_f - dot_g_Aadjf) / (std::abs(dot_Ag_f) + 1e-300);
//
//  deallog << std::scientific << "(Ag, f) = " << dot_Ag_f << ", (g, A*f) = " << dot_g_Aadjf
//          << ", rel. error = " << Agf_err << std::endl;
//
//  double dot_Af_g    = Af * (Tg);
//  double dot_f_Aadjg = (*f) * Aadjg;
//  double Afg_err     = std::abs(dot_Af_g - dot_f_Aadjg) / (std::abs(dot_Af_g) + 1e-300);
//
//  deallog << std::scientific << "(Af, g) = " << dot_Af_g << ", (f, A*g) = " << dot_f_Aadjg
//          << ", rel. error = " << Afg_err << std::endl;
//
//  double dot_Az_z    = Az * (Tz);
//  double dot_z_Aadjz = (*z) * Aadjz;
//  double Azz_err     = std::abs(dot_Az_z - dot_z_Aadjz) / (std::abs(dot_Az_z) + 1e-300);
//
//  deallog << std::scientific << "(Az, z) = " << dot_Az_z << ", (z, A*z) = " << dot_z_Aadjz
//          << ", rel. error = " << Azz_err << std::endl;
//
//  EXPECT_LT(ff_err, tol);
//  EXPECT_LT(gg_err, tol);
//  EXPECT_LT(gf_err, tol);
//  EXPECT_LT(fg_err, tol);
//  // EXPECT_LT(zz_err, tol);
//
//  EXPECT_LT(mff_err, tol);
//  EXPECT_LT(mgg_err, tol);
//  EXPECT_LT(mgf_err, tol);
//  EXPECT_LT(mfg_err, tol);
//  // EXPECT_LT(mzz_err, tol);
//
//  EXPECT_LT(Aff_err, tol);
//  EXPECT_LT(Agg_err, tol);
//  EXPECT_LT(Agf_err, tol);
//  EXPECT_LT(Afg_err, tol);
//  // EXPECT_LT(Azz_err, tol);
//}

template <int dim>
void run_wave_adjoint_test(int fe_order, int quad_order, int refines, int n_steps,
                           typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver, bool set_nu, double tol) {
  AssertThrow(adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint ||
                  adjoint_solver == WaveEquationBase<dim>::WaveEquationBackwards,
              ExcInternalError());

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

  WaveEquation<dim> wave_eq(mesh);
  wave_eq.set_param_a(std::make_shared<TestA<dim>>());
  wave_eq.set_param_c(std::make_shared<TestC<dim>>());
  wave_eq.set_param_q(std::make_shared<TestQ<dim>>());

  if (set_nu) wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

  WaveEquationAdjoint<dim> wave_eq_adj(wave_eq);

  bool use_adj   = adjoint_solver == WaveEquationBase<dim>::WaveEquationAdjoint;
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

    f->set_norm(Norm::L2L2);
    *f *= 1.0 / f->norm();

    g->set_norm(Norm::L2L2);
    *g *= 1.0 / g->norm();

    wave_eq.set_run_direction(WaveEquation<dim>::Forward);
    wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f));
    DiscretizedFunction<dim> sol_f = wave_eq.run();
    sol_f.set_norm(Norm::L2L2);
    EXPECT_GT(sol_f.norm(), 0.0);

    auto g_time_mass = std::make_shared<DiscretizedFunction<dim>>(*g);
    g_time_mass->set_norm(Norm::L2L2);

    DiscretizedFunction<dim> adj_g(mesh);
    if (use_adj) {
      // if L2RightHandSide would be used, we would also need a call to solve_mass
      g_time_mass->dot_transform();

      wave_eq_adj.set_right_hand_side(std::make_shared<VectorRightHandSide<dim>>(g_time_mass));
      adj_g = wave_eq_adj.run();

      // wave_eq_adj does everything except the multiplication with the mass matrix (to allow for optimization)
      adj_g.set_norm(Norm::L2L2);
      adj_g.dot_mult_mass_and_transform_inverse();
    } else {
      // dot_transforms not needed here, wave_eq backwards should be the L^2([0,T], L^2)-Adjoint

      wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(g_time_mass));
      wave_eq.set_run_direction(WaveEquation<dim>::Backward);
      adj_g = wave_eq.run();
      adj_g.throw_away_derivative();
      adj_g.set_norm(Norm::L2L2);
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
  EXPECT_LT(err_simple, tol);
  EXPECT_LT(err_avg, tol);
}
}  // namespace

TEST(WaveEquationAdjointness, Backwards1DFE1) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<1>(1, 3, 6, 1 << i, WaveEquationBase<1>::WaveEquationBackwards, false, 1e-1);
}

TEST(WaveEquationAdjointness, Adjoint1DFE1) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<1>(1, 3, 6, 1 << i, WaveEquationBase<1>::WaveEquationAdjoint, false, 1e-1);
}

TEST(WaveEquationAdjointness, Backwards1DFE2) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<1>(2, 6, 4, 1 << i, WaveEquationBase<1>::WaveEquationBackwards, false, 1e-1);
}

TEST(WaveEquationAdjointness, Adjoint1DFE2) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<1>(2, 6, 4, 1 << i, WaveEquationBase<1>::WaveEquationAdjoint, false, 1e-1);
}

TEST(WaveEquationAdjointness, Backwards2DFE1) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 3, 5, 1 << i, WaveEquationBase<2>::WaveEquationBackwards, false, 1e-1);
}

TEST(WaveEquationAdjointness, BackwardsNu2DFE1) {
  // this is more for demonstration purposes that backwards does not work well in this case

  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 3, 5, 1 << i, WaveEquationBase<2>::WaveEquationBackwards, true, 1e-1);
}

TEST(WaveEquationAdjointness, Adjoint2DFE1) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 3, 5, 1 << i, WaveEquationBase<2>::WaveEquationAdjoint, false, 1e-1);
}

TEST(WaveEquationAdjointness, AdjointNu2DFE1) {
  for (int i = 3; i < 10; i++)
    run_wave_adjoint_test<2>(1, 3, 5, 1 << i, WaveEquationBase<2>::WaveEquationAdjoint, true, 1e-1);
}

TEST(WaveEquationAdjointness, Adjoint3DFE1) {
  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 3, 2, 1 << i, WaveEquationBase<3>::WaveEquationAdjoint, false, 1e-1);
}

TEST(WaveEquationAdjointness, Backwards3DFE1) {
  for (int i = 3; i < 9; i++)
    run_wave_adjoint_test<3>(1, 3, 2, 1 << i, WaveEquationBase<3>::WaveEquationBackwards, false, 1e-1);
}
