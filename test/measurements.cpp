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

#include <base/AdaptiveMesh.h>
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
#include <measurements/ConvolutionMeasure.h>
#include <measurements/DeltaMeasure.h>
#include <measurements/Measure.h>
#include <measurements/SensorDistribution.h>
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

enum class MeasureType { convolution = 1, delta };

template <int dim>
void run_sensor_measure_adjoint_test(MeasureType measure_type, int fe_order, int quad_order, int refines, int n_steps) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 1.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  const size_t PPD = 2;

  std::vector<double> mtimes(PPD, 0.0);
  for (size_t i = 0; i < mtimes.size(); i++)
    mtimes[i] = (i + 1.0) / (mtimes.size() + 1.0);

  std::vector<std::vector<double>> spatial_points;
  for (size_t d = 0; d < dim; d++) {
    std::vector<double> tmp(PPD, 0.0);

    for (size_t i = 0; i < tmp.size(); i++)
      tmp[i] = ((i + 1.0) / (tmp.size() + 1.0)) * 2.0 - 1.0;

    spatial_points.push_back(tmp);
  }

  auto grid = std::make_shared<GridDistribution<dim>>(mtimes, spatial_points);
  std::shared_ptr<Measure<DiscretizedFunction<dim>, SensorValues<dim>>> measure;

  if (measure_type == MeasureType::convolution)
    measure = std::make_shared<ConvolutionMeasure<dim>>(
        mesh, grid, Norm::L2L2, std::make_shared<typename ConvolutionMeasure<dim>::HatShape>(), 0.1, 0.2);
  else if (measure_type == MeasureType::delta)
    measure = std::make_shared<DeltaMeasure<dim>>(mesh, grid, Norm::L2L2);

  double tol = 1e-06;

  for (int i = 0; i < 10; i++) {
    DiscretizedFunction<dim> f = DiscretizedFunction<dim>::noise(mesh);
    f.set_norm(Norm::L2L2);

    SensorValues<dim> g = SensorValues<dim>::noise(grid);

    Timer eval_timer;
    eval_timer.start();
    auto Psif = measure->evaluate(f);
    eval_timer.stop();

    Timer adj_timer;
    adj_timer.start();
    auto PsiAdjg = measure->adjoint(g);
    AssertThrow(PsiAdjg.get_norm() == Norm::L2L2, ExcInternalError());
    adj_timer.stop();

    double dot_Psif_g    = Psif * g;
    double dot_f_Psiadjg = f * PsiAdjg;
    double mfg_err       = std::abs(dot_Psif_g - dot_f_Psiadjg) / (std::abs(dot_Psif_g) + 1e-300);

    deallog << "wall time evaluate: " << std::fixed << eval_timer.wall_time() << " s" << std::endl;
    deallog << "wall time adjoint: " << std::fixed << adj_timer.wall_time() << " s" << std::endl;

    deallog << std::scientific << "(Ψf, g) = " << dot_Psif_g << ", (f, Ψ*g) = " << dot_f_Psiadjg
            << ", rel. error = " << mfg_err << std::endl;

    EXPECT_LT(mfg_err, tol);
  }

  deallog << std::endl;
}

// tests whether the specialized implementation of delta_measure for constant meshes yields the same result as for
// general meshes
template <int dim>
void run_delta_measure_implementation_test(int fe_order, int quad_order, int refines, int n_steps) {
  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -1, 1);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(refines);

  double t_start = 0.0, t_end = 1.0, dt = t_end / n_steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(fe_order);
  Quadrature<dim> quad = QGauss<dim>(quad_order);  // exact in poly degree 2n-1 (needed: fe_dim^3)

  std::shared_ptr<SpaceTimeMesh<dim>> mesh  = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);
  std::shared_ptr<SpaceTimeMesh<dim>> amesh = std::make_shared<AdaptiveMesh<dim>>(times, fe, quad, triangulation);

  deallog << std::endl << "----------  n_dofs / timestep: " << mesh->get_dof_handler(0)->n_dofs();
  deallog << ", n_steps: " << times.size() << "  ----------" << std::endl;

  const size_t PPD = 10;

  std::vector<double> mtimes(PPD, 0.0);
  for (size_t i = 0; i < mtimes.size(); i++)
    mtimes[i] = (i + 1.0) / (mtimes.size() + 1.0);

  std::vector<std::vector<double>> spatial_points;
  for (size_t d = 0; d < dim; d++) {
    std::vector<double> tmp(PPD, 0.0);

    for (size_t i = 0; i < tmp.size(); i++)
      tmp[i] = ((i + 1.0) / (tmp.size() + 1.0)) * 2.0 - 1.0;

    spatial_points.push_back(tmp);
  }

  auto grid     = std::make_shared<GridDistribution<dim>>(mtimes, spatial_points);
  auto measure  = std::make_shared<DeltaMeasure<dim>>(mesh, grid, Norm::L2L2);
  auto ameasure = std::make_shared<DeltaMeasure<dim>>(amesh, grid, Norm::L2L2);

  double tol = 1e-06;

  for (int i = 0; i < 10; i++) {
    DiscretizedFunction<dim> f = DiscretizedFunction<dim>::noise(mesh);
    f.set_norm(Norm::L2L2);

    DiscretizedFunction<dim> af = DiscretizedFunction<dim>(amesh);
    for (size_t ti = 0; ti < mesh->length(); ti++)
      af[ti] = f[ti];

    SensorValues<dim> g = SensorValues<dim>::noise(grid);

    Timer timer;
    timer.restart();
    auto Psif = measure->evaluate(f);
    timer.stop();
    deallog << "wall time evaluate (ConstantMesh): " << std::fixed << timer.wall_time() << " s" << std::endl;

    timer.restart();
    auto aPsif = ameasure->evaluate(af);
    timer.stop();
    deallog << "wall time evaluate (AdaptiveMesh): " << std::fixed << timer.wall_time() << " s" << std::endl;

    timer.restart();
    auto PsiAdjg = measure->adjoint(g);
    AssertThrow(PsiAdjg.get_norm() == Norm::L2L2, ExcInternalError());
    timer.stop();
    deallog << "wall time adjoint (ConstantMesh): " << std::fixed << timer.wall_time() << " s" << std::endl;

    timer.restart();
    auto aPsiAdjg = ameasure->adjoint(g);
    AssertThrow(aPsiAdjg.get_norm() == Norm::L2L2, ExcInternalError());
    timer.stop();
    deallog << "wall time adjoint (AdaptiveMesh): " << std::fixed << timer.wall_time() << " s" << std::endl;

    DiscretizedFunction<dim> PsiAdjg_copy = DiscretizedFunction<dim>(amesh, false, PsiAdjg.get_norm());
    for (size_t ti = 0; ti < mesh->length(); ti++)
      PsiAdjg_copy[ti] = PsiAdjg[ti];

    double psi_err = aPsif.relative_error(Psif);
    deallog << std::scientific << "error [ Ψf (ConstantMesh - Ψf (AdaptiveMesh) ] = " << psi_err << std::endl;

    EXPECT_LT(psi_err, tol);

    double psiadj_err = aPsiAdjg.relative_error(PsiAdjg_copy);
    deallog << std::scientific << "error [ Ψ*f (ConstantMesh - Ψ*f (AdaptiveMesh) ] = " << psiadj_err << std::endl;

    EXPECT_LT(psiadj_err, tol);
  }

  deallog << std::endl;
}

}  // namespace

TEST(Measurements, ConvolutionMeasureAdjointness1DFE1) {
  run_sensor_measure_adjoint_test<1>(MeasureType::convolution, 1, 4, 9, 256);
}

TEST(Measurements, ConvolutionMeasureAdjointness1DFE2) {
  run_sensor_measure_adjoint_test<1>(MeasureType::convolution, 2, 4, 7, 128);
}

TEST(Measurements, ConvolutionMeasureAdjointness2DFE1) {
  run_sensor_measure_adjoint_test<2>(MeasureType::convolution, 1, 4, 5, 256);
}

TEST(Measurements, ConvolutionMeasureAdjointness2DFE2) {
  run_sensor_measure_adjoint_test<2>(MeasureType::convolution, 2, 4, 5, 128);
}

TEST(Measurements, ConvolutionMeasureAdjointness3DFE1) {
  run_sensor_measure_adjoint_test<3>(MeasureType::convolution, 1, 4, 4, 128);
}

TEST(Measurements, DeltaMeasureAdjointness1DFE1) {
  run_sensor_measure_adjoint_test<1>(MeasureType::delta, 1, 4, 9, 256);
}

TEST(Measurements, DeltaMeasureAdjointness1DFE2) {
  run_sensor_measure_adjoint_test<1>(MeasureType::delta, 2, 4, 7, 128);
}

TEST(Measurements, DeltaMeasureAdjointness2DFE1) {
  run_sensor_measure_adjoint_test<2>(MeasureType::delta, 1, 4, 5, 256);
}

TEST(Measurements, DeltaMeasureAdjointness2DFE2) {
  run_sensor_measure_adjoint_test<2>(MeasureType::delta, 2, 4, 5, 128);
}

TEST(Measurements, DeltaMeasureAdjointness3DFE1) {
  run_sensor_measure_adjoint_test<3>(MeasureType::delta, 1, 4, 4, 128);
}

TEST(Measurements, DeltaMeasureImplementation1DFE1) { run_delta_measure_implementation_test<1>(1, 4, 9, 256); }

TEST(Measurements, DeltaMeasureImplementation1DFE2) { run_delta_measure_implementation_test<1>(2, 4, 7, 128); }

TEST(Measurements, DeltaMeasureImplementation2DFE1) { run_delta_measure_implementation_test<2>(1, 4, 5, 256); }

TEST(Measurements, DeltaMeasureImplementation2DFE2) { run_delta_measure_implementation_test<2>(2, 4, 5, 128); }

TEST(Measurements, DeltaMeasureImplementation3DFE1) { run_delta_measure_implementation_test<3>(1, 4, 3, 128); }
