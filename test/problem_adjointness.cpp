/*
 * problem_adjointness.cpp
 *
 *  Created on: 06.03.2018
 *      Author: thies
 */

#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/MacroFunctionParser.h>
#include <base/Norm.h>
#include <base/SpaceTimeMesh.h>
#include <base/Transformation.h>
#include <base/Tuple.h>
#include <base/Util.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationBase.h>
#include <gtest/gtest.h>
#include <measurements/FieldMeasure.h>
#include <measurements/Measure.h>
#include <norms/H1H1.h>
#include <norms/H1L2.h>
#include <norms/L2L2.h>
#include <problems/AProblem.h>
#include <problems/CProblem.h>
#include <problems/NuProblem.h>
#include <problems/QProblem.h>
#include <stddef.h>
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
using namespace wavepi;

template<int dim>
class TestF: public Function<dim> {
public:
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      if (p.norm() < 0.5)
         return std::sin(this->get_time() * 2 * numbers::PI);
      else
         return 0.0;
   }
};

template<int dim>
class TestF2: public Function<dim> {
public:
   TestF2()
         : Function<dim>() {
   }
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

template<>
const Point<1> TestF2<1>::actor_position = Point<1>(1.0);
template<>
const Point<2> TestF2<2>::actor_position = Point<2>(1.0, 0.5);
template<>
const Point<3> TestF2<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template<int dim>
class TestG: public Function<dim> {
public:
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      Point<dim> pc = Point<dim>::unit_vector(0);
      pc *= 0.5;

      return this->get_time() * std::sin(p.distance(pc) * 2 * numbers::PI);
   }
};

template<int dim>
class TestH: public Function<dim> {
public:
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

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
class TestC: public Function<dim> {
public:
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return 1.0 / (rho(p, this->get_time()) * c_squared(p, this->get_time()));
   }
};

template<int dim>
class TestA: public Function<dim> {
public:
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return 1.0 / rho(p, this->get_time());
   }
};

template<int dim>
class TestNu: public Function<dim> {
public:
   double value(const Point<dim> &p __attribute__((unused)), const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      if (this->get_time() > 1.0) return 0.0;

      return std::abs(p[0]) * this->get_time();
   }
};

template<int dim>
class TestQ: public Function<dim> {
public:
   TestQ()
         : Function<dim>() {
   }
   double value(const Point<dim> &p, const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return p.norm() < 0.5 ? std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
   }

   static const Point<dim> q_position;
};

template<int dim>
class TestEstimate: public Function<dim> {
public:
   TestEstimate()
         : Function<dim>() {
   }
   double value(const Point<dim> &p __attribute__((unused)), const unsigned int component = 0) const {
      Assert(component == 0, ExcIndexRange(component, 0, 1));

      return 2;
   }
};

template<>
const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<>
const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<>
const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

template<int dim, typename ProblemType>
void run_adjoint_test(int fe_order, int quad_order, int refines, int n_steps,
      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_domain,
      std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm_codomain, double tol) {
   auto triangulation = std::make_shared<Triangulation<dim>>();
   GridGenerator::hyper_cube(*triangulation, -1, 1);
   Util::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(refines);

   double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, FE_Q<dim>(fe_order),
         QGauss<dim>(quad_order), triangulation);

   deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
   deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

   WaveEquation<dim> wave_eq(mesh);
   wave_eq.set_param_a(std::make_shared<TestA<dim>>());
   wave_eq.set_param_c(std::make_shared<TestC<dim>>());
   wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
   wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

   TestEstimate<dim> est_cont;
   DiscretizedFunction<dim> estimate(mesh, est_cont);
   estimate.set_norm(norm_domain);

   std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> measures;
   measures.push_back(std::make_shared<FieldMeasure<dim>>(mesh, norm_codomain));

   std::map<std::string, double> consts;
   std::vector<std::shared_ptr<Function<dim>>> pulses;
   pulses.push_back(std::make_shared<MacroFunctionParser<dim>>("if(norm{x|y|z} < 0.2, sin(t), 0.0)", consts));

   ProblemType problem(wave_eq, pulses, measures, std::make_shared<IdentityTransform<dim>>(), 0);
   problem.set_adjoint_solver(WaveEquationBase<dim>::WaveEquationAdjoint);

   problem.set_norm_domain(norm_domain);
   problem.set_norm_codomain(norm_codomain);

   auto data_current = problem.forward(estimate);  // have to run forward at least once
   auto A = problem.derivative(estimate);

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
         f->set_norm(std::make_shared<norms::H1L2<dim>>(0.5));
         f->dot_transform_inverse();

         g = std::make_shared<DiscretizedFunction<dim>>(DiscretizedFunction<dim>::noise(mesh));

         // make it a bit smoother, random noise might be a bit too harsh
         g->set_norm(std::make_shared<norms::H1L2<dim>>(0.5));
         g->dot_transform_inverse();
      }

      // normalize both f and g (not necessary)
      f->set_norm(norm_domain);
      *f *= 1.0 / f->norm();

      g->set_norm(norm_codomain);
      *g *= 1.0 / g->norm();
      Tuple<DiscretizedFunction<dim>> Tg(*g);

      auto Af = A->forward(*f);
      EXPECT_GT(Af.norm(), 0.0);

      auto Astarg = A->adjoint(Tg);
      EXPECT_GT(Astarg.norm(), 0.0);

      double dot_solf_g = Af * Tg;
      double dot_f_adjg = (*f) * Astarg;
      double fg_err = std::abs(dot_solf_g - dot_f_adjg) / (std::abs(dot_solf_g) + 1e-300);

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

/* Q */

TEST(ProblemAdjointness, AdjointQ1DFE1) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, QProblem<1, DiscretizedFunction<1>>>(1, 3, 6, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 6; refine >= 1; refine--)
      run_adjoint_test<1, QProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointQ1DFE2) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, QProblem<1, DiscretizedFunction<1>>>(2, 6, 4, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<1, QProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointQ2DFE1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, QProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, QProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointQ2DFE1H1H1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, QProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, QProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointQ3DFE1) {
   for (int i = 3; i < 5; i++)
      run_adjoint_test<3, QProblem<3, DiscretizedFunction<3>>>(1, 3, 2, 1 << i, std::make_shared<norms::L2L2<3>>(),
            std::make_shared<norms::L2L2<3>>(), 1e-1);
}

/* A */

TEST(ProblemAdjointness, AdjointA1DFE1) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, AProblem<1, DiscretizedFunction<1>>>(1, 3, 6, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 6; refine >= 1; refine--)
      run_adjoint_test<1, AProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointA1DFE2) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, AProblem<1, DiscretizedFunction<1>>>(2, 6, 4, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<1, AProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointA2DFE1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, AProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, AProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointA2DFE1H1H1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, AProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, AProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointA3DFE1) {
   for (int i = 3; i < 5; i++)
      run_adjoint_test<3, AProblem<3, DiscretizedFunction<3>>>(1, 3, 2, 1 << i, std::make_shared<norms::L2L2<3>>(),
            std::make_shared<norms::L2L2<3>>(), 1e-1);
}

/* Nu */

TEST(ProblemAdjointness, AdjointNu1DFE1) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, NuProblem<1, DiscretizedFunction<1>>>(1, 3, 6, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 6; refine >= 1; refine--)
      run_adjoint_test<1, NuProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9,
            std::make_shared<norms::L2L2<1>>(), std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointNu1DFE2) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, NuProblem<1, DiscretizedFunction<1>>>(2, 6, 4, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<1, NuProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9,
            std::make_shared<norms::L2L2<1>>(), std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointNu2DFE1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, NuProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, NuProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8,
            std::make_shared<norms::L2L2<2>>(), std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointNu2DFE1H1H1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, NuProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, NuProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointNu3DFE1) {
   for (int i = 3; i < 5; i++)
      run_adjoint_test<3, NuProblem<3, DiscretizedFunction<3>>>(1, 3, 2, 1 << i, std::make_shared<norms::L2L2<3>>(),
            std::make_shared<norms::L2L2<3>>(), 1e-1);
}

/* C */

TEST(ProblemAdjointness, AdjointC1DFE1) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, CProblem<1, DiscretizedFunction<1>>>(1, 3, 6, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 6; refine >= 1; refine--)
      run_adjoint_test<1, CProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointC1DFE2) {
   for (int i = 3; i < 10; i++)
      run_adjoint_test<1, CProblem<1, DiscretizedFunction<1>>>(2, 6, 4, 1 << i, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<1, CProblem<1, DiscretizedFunction<1>>>(1, 3, refine, 1 << 9, std::make_shared<norms::L2L2<1>>(),
            std::make_shared<norms::L2L2<1>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointC2DFE1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, CProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, CProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8, std::make_shared<norms::L2L2<2>>(),
            std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointC2DFE1H1H1) {
   for (int i = 3; i < 9; i++)
      run_adjoint_test<2, CProblem<2, DiscretizedFunction<2>>>(1, 3, 5, 1 << i,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);

   for (int refine = 4; refine >= 1; refine--)
      run_adjoint_test<2, CProblem<2, DiscretizedFunction<2>>>(1, 3, refine, 1 << 8,
            std::make_shared<norms::H1H1<2>>(0.5, 0.5), std::make_shared<norms::L2L2<2>>(), 1e-1);
}

TEST(ProblemAdjointness, AdjointC3DFE1) {
   for (int i = 3; i < 5; i++)
      run_adjoint_test<3, CProblem<3, DiscretizedFunction<3>>>(1, 3, 2, 1 << i, std::make_shared<norms::L2L2<3>>(),
            std::make_shared<norms::L2L2<3>>(), 1e-1);
}
