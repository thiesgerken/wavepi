/*
 * wave_equation.cpp
 *
 *  Created on: 11.07.2017
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

template<int dim>
class TestF: public LightFunction<dim> {
public:
   virtual ~TestF() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      if ((t <= 1) && (p.norm() < 0.5))
         return std::sin(t * 2 * numbers::PI);
      else
         return 0.0;
   }
};

template<int dim>
class TestG: public LightFunction<dim> {
public:
   virtual ~TestG() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      Point<dim> pc = Point<dim>::unit_vector(0);
      pc *= 0.5;

      if (std::abs(this->get_time() - 1.0) < 0.5 && (p.distance(pc) < 0.5))
         return std::sin(this->get_time() * 1 * numbers::PI);
      else
         return 0.0;
   }
};

template<int dim>
class TestH: public LightFunction<dim> {
public:
   virtual ~TestH() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      return p.norm() * this->get_time();
   }
};

template<int dim>
double rho(const Point<dim> &p, double t) {
   return p.norm() + t + 1.0;
}

template<int dim>
double c_squared(const Point<dim> &p, double t) {
   double tmp = p.norm() * t + 1.0;

   return tmp * tmp;
}

template<int dim>
class TestC: public LightFunction<dim> {
public:
   virtual ~TestC() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      return 1.0 / (rho(p, t) * c_squared(p, t));
   }
};

template<int dim>
class TestRho: public LightFunction<dim> {
public:
   virtual ~TestRho() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      return rho(p, t);
   }
};

template<int dim>
class TestNu: public LightFunction<dim> {
public:
   virtual ~TestNu() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      // do not just change this, reference_test_nu needs this
      return p.norm() * t;
   }
};

template<int dim>
class RhsTestNu: public LightFunction<dim> {
public:
   virtual ~RhsTestNu() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      // do not just change this, reference_test_nu needs this
      return nu.value(p, t) * v->value(p, t);
   }

   RhsTestNu(const std::shared_ptr<LightFunction<dim>> &v)
         : v(v) {
      AssertThrow(v, ExcNotInitialized());
   }

private:
   std::shared_ptr<LightFunction<dim>> v;  // pointer to u'
   TestNu<dim> nu;
};

template<int dim>
class TestQ: public LightFunction<dim> {
public:
   virtual ~TestQ() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const override {
      return p.norm() < 0.5 ? std::sin(t / 2 * 2 * numbers::PI) : 0.0;
   }

   static const Point<dim> q_position;
};

template<>
const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<>
const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<>
const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

template<int dim>
class DiscretizedFunctionDisguise: public LightFunction<dim> {
public:
   DiscretizedFunctionDisguise(std::shared_ptr<DiscretizedFunction<dim>> base)
         : base(base) {
   }
   virtual ~DiscretizedFunctionDisguise() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      return base->value(p, t);
   }
private:
   std::shared_ptr<DiscretizedFunction<dim>> base;
};

// checks, whether the matrix assembly of discretized parameters works correct
// (by supplying DiscretizedFunctions and DiscretizedFunctionDisguises)
template<int dim>
void run_discretized_test(int fe_order, int quad_order, int refines) {
   Timer timer;

   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

   WaveEquation<dim> wave_eq(mesh);

   /* continuous */

   wave_eq.set_param_rho(std::make_shared<TestRho<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   timer.restart();
   DiscretizedFunction<dim> sol_cont = wave_eq.run(
         std::make_shared<L2RightHandSide<dim>>(std::make_shared<TestF<dim>>()), WaveEquation<dim>::Forward);
   sol_cont.set_norm(std::make_shared<norms::L2L2<dim>>());

   timer.stop();
   deallog << "continuous params: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_cont.norm(), 0.0);

   /* discretized */

   TestC<dim> c;
   auto c_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, c);
   wave_eq.set_param_c(c_disc);

   TestRho<dim> rho;
   auto rho_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, rho);
   wave_eq.set_param_rho(rho_disc);

   TestQ<dim> q;
   auto q_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, q);
   wave_eq.set_param_q(q_disc);

   TestNu<dim> nu;
   auto nu_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, nu);
   wave_eq.set_param_nu(nu_disc);

   TestF<dim> f;
   auto f_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, f);

   timer.restart();
   DiscretizedFunction<dim> sol_disc = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(f_disc),
         WaveEquation<dim>::Forward);
   sol_disc.set_norm(std::make_shared<norms::L2L2<dim>>());
   timer.stop();
   deallog << "all discretized: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc.norm(), 0.0);

   /* discretized, q disguised */

   auto c_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(c_disc);
   auto rho_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(rho_disc);
   auto q_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(q_disc);
   auto nu_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(nu_disc);
   auto f_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(f_disc);

   wave_eq.set_param_q(q_disguised);

   timer.restart();
   DiscretizedFunction<dim> sol_disc_except_q = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(f_disc),
         WaveEquation<dim>::Forward);
   sol_disc_except_q.set_norm(std::make_shared<norms::L2L2<dim>>());
   timer.stop();
   deallog << "all discretized, q disguised: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc_except_q.norm(), 0.0);

   /* discretized, a disguised */

   wave_eq.set_param_rho(rho_disguised);
   wave_eq.set_param_q(q_disc);

   timer.restart();
   DiscretizedFunction<dim> sol_disc_except_a = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(f_disc),
         WaveEquation<dim>::Forward);
   sol_disc_except_a.set_norm(std::make_shared<norms::L2L2<dim>>());
   timer.stop();
   deallog << "all discretized, a disguised: " << std::fixed << timer.wall_time() << " s of wall time" << std::endl;
   EXPECT_GT(sol_disc_except_a.norm(), 0.0);

   /* disguised */

   wave_eq.set_param_nu(nu_disguised);
   wave_eq.set_param_q(q_disguised);
   wave_eq.set_param_rho(rho_disguised);
   wave_eq.set_param_c(c_disguised);

   timer.restart();
   DiscretizedFunction<dim> sol_disguised = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(f_disguised),
         WaveEquation<dim>::Forward);
   sol_disguised.set_norm(std::make_shared<norms::L2L2<dim>>());
   timer.stop();
   deallog << "all discretized and disguised as continuous: " << std::fixed << timer.wall_time() << " s of wall time"
         << std::endl << std::endl;
   EXPECT_GT(sol_disguised.norm(), 0.0);

   /* results */
   DiscretizedFunction<dim> tmp(sol_disc_except_q);
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_q = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (q disguised): " << std::scientific
         << err_disguised_vs_disc_except_q << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_q, 1e-6);

   tmp = sol_disc_except_a;
   tmp -= sol_disguised;
   double err_disguised_vs_disc_except_a = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and discrete (a disguised): " << std::scientific
         << err_disguised_vs_disc_except_a << std::endl;
   EXPECT_LT(err_disguised_vs_disc_except_a, 1e-6);

   tmp = sol_disc;
   tmp -= sol_disguised;
   double err_disguised_vs_disc = tmp.norm() / sol_disguised.norm();

   deallog << "rel. error between disguised discrete and full discrete: " << std::scientific << err_disguised_vs_disc
         << std::endl << std::endl;
   EXPECT_LT(err_disguised_vs_disc, 1e-6);
}

// product of sines in space to have dirichlet b.c. in [0,pi], times a sum of sine and cosine in time.
// its time derivative is the same function with C[1] = C[0]*norm(k), C[0] = -C[1]*norm(k)
template<int dim>
class SeparationAnsatz: public LightFunction<dim> {
public:
   virtual ~SeparationAnsatz() = default;

   virtual double evaluate(const Point<dim> &p, const double t) const {
      double res = 1;

      for (size_t i = 0; i < dim; i++)
         res *= std::sin(k[i] * p[i]);

      res *= constants[0] * std::sin(std::sqrt(k.square()) * t) + constants[1] * std::cos(std::sqrt(k.square()) * t);
      return res;
   }

   SeparationAnsatz(Point<dim, int> k, Point<2> constants)
         : k(k), constants(constants) {
   }

private:
   Point<dim, int> k;
   Point<2> constants;
};

template<int dim>
void run_reference_test(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Point<dim, int> k, Point<2> constants, bool expect =
      true, bool save = false) {
   deallog << std::endl << "----------  n_dofs(0): " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << mesh->get_times().size() << "  ----------" << std::endl;

   WaveEquation<dim> wave_eq(mesh);

   Point<2> derivative_constants;
   derivative_constants[0] = -constants[1] * std::sqrt(k.square());
   derivative_constants[1] = constants[0] * std::sqrt(k.square());

   auto u = std::make_shared<SeparationAnsatz<dim>>(k, constants);
   auto v = std::make_shared<SeparationAnsatz<dim>>(k, derivative_constants);

   wave_eq.set_initial_values_u(u);
   wave_eq.set_initial_values_v(v);

   DiscretizedFunction<dim> solu = wave_eq.run(
         std::make_shared<L2RightHandSide<dim>>(std::make_shared<Functions::ZeroFunction<dim>>(1)),
         WaveEquation<dim>::Forward);
   DiscretizedFunction<dim> solv = solu.derivative();
   solu.throw_away_derivative();

   solu.set_norm(std::make_shared<norms::L2L2<dim>>());
   solv.set_norm(std::make_shared<norms::L2L2<dim>>());

   DiscretizedFunction<dim> refu(mesh, *u);
   DiscretizedFunction<dim> refv(mesh, *v);

   refu.set_norm(std::make_shared<norms::L2L2<dim>>());
   refv.set_norm(std::make_shared<norms::L2L2<dim>>());

   DiscretizedFunction<dim> tmp(solu);
   tmp -= refu;
   double err_u = tmp.norm() / refu.norm();

   tmp = solv;
   tmp -= refv;
   double err_v = tmp.norm() / refv.norm();

   if (expect) {
      EXPECT_LT(err_u, 1e-1);
      EXPECT_LT(err_v, 1e-1);
   }

   if (save) {
      solu.write_pvd("./", "solu", "u");
      refu.write_pvd("./", "refu", "uref");

      DiscretizedFunction<dim> tmp(solu);
      tmp -= refu;
      tmp.write_pvd("./", "diff", "udiff");
   }

   deallog << std::scientific << "forward : rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl;

   solu = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(std::make_shared<Functions::ZeroFunction<dim>>(1)),
         WaveEquation<dim>::Backward);
   solv = solu.derivative();
   solu.throw_away_derivative();

   solu.set_norm(std::make_shared<norms::L2L2<dim>>());
   solv.set_norm(std::make_shared<norms::L2L2<dim>>());

   tmp = solu;
   tmp -= refu;
   err_u = tmp.norm() / refu.norm();

   tmp = solv;
   tmp -= refv;
   err_v = tmp.norm() / refv.norm();

   if (expect) {
      EXPECT_LT(err_u, 1e-1);
      EXPECT_LT(err_v, 1e-1);
   }

   deallog << std::scientific << "backward: rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl << std::endl;
}
}  // namespace

template<int dim>
void run_reference_test_constant(int fe_order, int quad_order, int refines, Point<dim, int> k, Point<2> constants,
      double t_end, int steps, bool expect = true, bool save = false) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, dt = t_end / steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

   run_reference_test<dim>(mesh, k, constants, expect, save);
}

template<int dim>
void run_reference_test_nu(int fe_order, int quad_order, int refines, Point<dim, int> k, Point<2> constants,
      double t_end, int steps, bool expect = true, bool save = false) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, dt = t_end / steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);
   deallog << std::endl << "----------  n_dofs(0): " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << mesh->get_times().size() << "  ----------" << std::endl;

   Point<2> derivative_constants;
   derivative_constants[0] = -constants[1] * std::sqrt(k.square());
   derivative_constants[1] = constants[0] * std::sqrt(k.square());

   auto u = std::make_shared<SeparationAnsatz<dim>>(k, constants);
   auto v = std::make_shared<SeparationAnsatz<dim>>(k, derivative_constants);

   WaveEquation<dim> wave_eq(mesh);
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   wave_eq.set_initial_values_u(u);
   wave_eq.set_initial_values_v(v);

   auto x = std::make_shared<RhsTestNu<dim>>(v);

   DiscretizedFunction<dim> solu = wave_eq.run(std::make_shared<L2RightHandSide<dim>>(x), WaveEquation<dim>::Forward);
   DiscretizedFunction<dim> solv = solu.derivative();
   solu.throw_away_derivative();

   solu.set_norm(std::make_shared<norms::L2L2<dim>>());
   solv.set_norm(std::make_shared<norms::L2L2<dim>>());

   DiscretizedFunction<dim> refu(mesh, *u);
   DiscretizedFunction<dim> refv(mesh, *v);

   refu.set_norm(std::make_shared<norms::L2L2<dim>>());
   refv.set_norm(std::make_shared<norms::L2L2<dim>>());

   DiscretizedFunction<dim> tmp(solu);
   tmp -= refu;
   double err_u = tmp.norm() / refu.norm();

   tmp = solv;
   tmp -= refv;
   double err_v = tmp.norm() / refv.norm();

   if (expect) {
      EXPECT_LT(err_u, 1e-1);
      EXPECT_LT(err_v, 1e-1);
   }

   if (save) {
      solu.write_pvd("./", "solu", "u");
      refu.write_pvd("./", "refu", "uref");

      DiscretizedFunction<dim> tmp(solu);
      tmp -= refu;
      tmp.write_pvd("./", "diff", "udiff");
   }

   deallog << std::scientific << "forward : rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl;

   // backward does not work for nu > 0
   /*
    wave_eq.set_run_direction(WaveEquation<dim>::Backward);
    solu = wave_eq.run();
    solv = solu.derivative();
    solu.throw_away_derivative();

    solu.set_norm(Norm::L2L2);
    solv.set_norm(Norm::L2L2);

    tmp = solu;
    tmp -= refu;
    err_u = tmp.norm() / refu.norm();

    tmp = solv;
    tmp -= refv;
    err_v = tmp.norm() / refv.norm();

    if (expect) {
    EXPECT_LT(err_u, 1e-1);
    EXPECT_LT(err_v, 1e-1);
    }

    deallog << std::scientific << "backward: rerr(u) = " << err_u << ", rerr(v) = " << err_v << std::endl << std::endl;
    */
}

template<int dim>
void run_reference_test_adaptive(int fe_order, int quad_order, int refines, Point<dim, int> k, Point<2> constants,
      double t_end, int steps, bool expect = true, bool save = false) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, dt = t_end / steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto mesh = std::make_shared<AdaptiveMesh<dim>>(times, fe, quad, triangulation);

   // flag some cells for refinement, and refine them in some step
   for (auto cell : triangulation->active_cell_iterators())
      if (cell->center()[0] > numbers::PI / 2) cell->set_refine_flag();

   std::vector<bool> ref;
   std::vector<bool> coa;

   triangulation->save_refine_flags(ref);
   triangulation->save_coarsen_flags(coa);

   for (auto cell : triangulation->active_cell_iterators())
      cell->clear_refine_flag();

   std::vector<Patch> patches = mesh->get_forward_patches();
   patches[steps / 4].emplace_back(ref, coa);
   mesh->set_forward_patches(patches);

   mesh->get_dof_handler(0);

   run_reference_test<dim>(mesh, k, constants, expect, save);
}

template<int dim>
void run_reference_test_refined(int fe_order, int quad_order, int refines, Point<dim, int> k, Point<2> constants,
      double t_end, int steps, bool expect = true, bool save = false) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, 0.0, numbers::PI);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   // flag some cells for refinement and refine them
   for (auto cell : triangulation->active_cell_iterators())
      if (cell->center()[1] > numbers::PI / 2) cell->set_refine_flag();

   triangulation->execute_coarsening_and_refinement();

   double t_start = 0.0, dt = t_end / steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   FE_Q<dim> fe(fe_order);
   Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

   auto mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

   run_reference_test<dim>(mesh, k, constants, expect, save);
}

TEST(WaveEquation, DiscretizedParameters1DFE1) {
   run_discretized_test<1>(1, 3, 8);
}

TEST(WaveEquation, DiscretizedParameters1DFE2) {
   run_discretized_test<1>(2, 4, 8);
}

TEST(WaveEquation, DiscretizedParameters2DFE1) {
   run_discretized_test<2>(1, 3, 3);
}

TEST(WaveEquation, DiscretizedParameters2DFE2) {
   run_discretized_test<2>(2, 4, 3);
}

TEST(WaveEquation, DiscretizedParameters3DFE1) {
   run_discretized_test<3>(1, 3, 1);
}

TEST(WaveEquation, ReferenceTest1DFE1) {
   for (int steps = 128; steps <= 1024; steps *= 2)
      run_reference_test_constant<1>(1, 3, 10, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 7; refine >= 1; refine--)
      run_reference_test_constant<1>(1, 3, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 1024, false);
}

TEST(WaveEquation, ReferenceTest1DFE2) {
   for (int steps = 16; steps <= 128; steps *= 2)
      run_reference_test_constant<1>(2, 4, 7, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 6; refine >= 1; refine--)
      run_reference_test_constant<1>(2, 4, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 128, false);
}

TEST(WaveEquation, ReferenceTest2DFE1) {
   for (int steps = 16; steps <= 256; steps *= 2)
      run_reference_test_constant<2>(1, 3, 5, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64);

   for (int refine = 5; refine >= 1; refine--)
      run_reference_test_constant<2>(1, 3, refine, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, 256,
            false);
}

TEST(WaveEquation, ReferenceTestAdaptive2DFE1) {
   for (int steps = 16; steps <= 256; steps *= 2)
      run_reference_test_adaptive<2>(1, 3, 5, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64, false);

   for (int refine = 5; refine >= 1; refine--)
      run_reference_test_adaptive<2>(1, 3, refine, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, 256,
            false);
}

TEST(WaveEquation, ReferenceTestRefined2DFE1) {
   for (int steps = 16; steps <= 256; steps *= 2)
      run_reference_test_refined<2>(1, 3, 5, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps,
            steps >= 64, false);

   for (int refine = 5; refine >= 1; refine--)
      run_reference_test_refined<2>(1, 3, refine, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, 256, false);
}

TEST(WaveEquation, ReferenceTest3DFE1) {
   for (int steps = 8; steps <= 32; steps *= 2)
      run_reference_test_constant<3>(1, 3, 3, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, steps,
            steps >= 32);

   for (int refine = 2; refine >= 0; refine--)
      run_reference_test_constant<3>(1, 3, refine, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, 32,
            false);
}

TEST(WaveEquation, ReferenceTestNu1DFE1) {
   for (int steps = 128; steps <= 1024; steps *= 2)
      run_reference_test_nu<1>(1, 3, 10, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps, steps >= 64);

   for (int refine = 7; refine >= 1; refine--)
      run_reference_test_nu<1>(1, 3, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 1024, false);
}

TEST(WaveEquation, ReferenceTestNu1DFE2) {
   for (int steps = 16; steps <= 128; steps *= 2)
      run_reference_test_nu<1>(2, 4, 7, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps, steps >= 64);

   for (int refine = 6; refine >= 1; refine--)
      run_reference_test_nu<1>(2, 4, refine, Point<1, int>(2), Point<2>(1.0, 1.5), 2 * numbers::PI, 128, false);
}

TEST(WaveEquation, ReferenceTestNu2DFE1) {
   for (int steps = 16; steps <= 512; steps *= 2)
      run_reference_test_nu<2>(1, 3, 6, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, steps, steps >= 64);

   for (int refine = 5; refine >= 1; refine--)
      run_reference_test_nu<2>(1, 3, refine, Point<2, int>(1, 2), Point<2>(1.0, 1.5), 2 * numbers::PI, 256, false);
}

TEST(WaveEquation, ReferenceTestNu3DFE1) {
   for (int steps = 8; steps <= 32; steps *= 2)
      run_reference_test_nu<3>(1, 3, 3, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, steps,
            steps >= 32);

   for (int refine = 2; refine >= 0; refine--)
      run_reference_test_nu<3>(1, 3, refine, Point<3, int>(1, 2, 3), Point<2>(0.7, 1.2), 2 * numbers::PI, 32, false);
}
