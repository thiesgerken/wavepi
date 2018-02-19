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
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>
#include <measurements/GridPointMeasure.h>
#include <measurements/Measure.h>
#include <measurements/PointMeasure.h>
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
using namespace wavepi::base;
using namespace wavepi::measurements;

// tests whether mass matrix operations are inverse to each other
template <int dim>
void run_grid_measure_adjoint_test(int fe_order, int quad_order, int refines, int n_steps) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 2.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  std::vector<double> mtimes;
  for (size_t i = 0; i < 10; i++)
    mtimes.push_back(i * 0.1);

  std::vector<std::vector<double>> spatial_points;
  for (size_t d = 0; d < dim; d++) {
    std::vector<double> tmp;

    for (size_t i = 0; i < 10; i++)
      tmp.push_back((i + 1.0) / 2.0 * 0.1);

    spatial_points.push_back(tmp);
  }

  auto grid = std::make_shared<SpaceTimeGrid<dim>>(mtimes, spatial_points);
  auto my_measure =
      std::make_shared<PointMeasure<dim>>(grid, std::make_shared<typename PointMeasure<dim>::HatShape>(), 0.1, 0.1);

  double tol = 1e-06;

  for (int i = 0; i < 10; i++) {
    DiscretizedFunction<dim> f = DiscretizedFunction<dim>::noise(mesh);
    f.set_norm(Norm::L2L2);

    MeasuredValues<dim> g = MeasuredValues<dim>::noise(grid);

    auto Psif    = my_measure->evaluate(f);
    auto PsiAdjg = my_measure->adjoint(g);
    PsiAdjg.set_norm(Norm::L2L2);

    double dot_Psif_g    = Psif * g;
    double dot_f_Psiadjg = f * PsiAdjg;
    double mfg_err       = std::abs(dot_Psif_g - dot_f_Psiadjg) / (std::abs(dot_Psif_g) + 1e-300);

    deallog << std::scientific << "(Ψf, g) = " << dot_Psif_g << ", (f, Ψ*g) = " << dot_f_Psiadjg
            << ", rel. error = " << mfg_err << std::endl;

    EXPECT_LT(mfg_err, tol);
  }

  deallog << std::endl;
}

}  // namespace

TEST(Measurements, PointMeasureAdjointness1DFE1) {
  run_grid_measure_adjoint_test<1>(1, 3, 10, 128);
  run_grid_measure_adjoint_test<1>(1, 4, 9, 256);
}
TEST(Measurements, PointMeasureAdjointness1DFE2) {
  run_grid_measure_adjoint_test<1>(2, 4, 7, 128);
  run_grid_measure_adjoint_test<1>(2, 4, 7, 256);
}

TEST(Measurements, PointMeasureAdjointness2DFE1) {
  run_grid_measure_adjoint_test<2>(1, 3, 5, 128);
  run_grid_measure_adjoint_test<2>(1, 4, 4, 256);
}

TEST(Measurements, PointMeasureAdjointness2DFE2) {
  run_grid_measure_adjoint_test<2>(2, 4, 4, 128);
  run_grid_measure_adjoint_test<2>(2, 4, 4, 256);
}

TEST(Measurements, PointMeasureAdjointness3DFE1) {
  run_grid_measure_adjoint_test<3>(1, 3, 2, 128);
  run_grid_measure_adjoint_test<3>(1, 4, 2, 256);
}
