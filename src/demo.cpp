/*
 * demo.cpp
 *
 *  Created on: 25.09.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <base/ConstantMesh.h>
#include <base/DiscretizedFunction.h>
#include <base/Util.h>
#include <forward/WaveEquation.h>

#include <stddef.h>
#include <ctgmath>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

using namespace dealii;
using namespace wavepi;
using namespace wavepi::forward;
using namespace wavepi::base;

template <int dim>
class DemoF : public Function<dim> {
 public:
  static const Point<dim> center;

  double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if (p.distance(center) < 0.75)
      return std::sin(this->get_time() * 2 * numbers::PI);
    else
      return 0.0;
  }
};

template <int dim>
class DemoC : public Function<dim> {
 public:
  static const Point<dim> center;

  virtual ~DemoC() = default;

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    if (p.distance(center) < 2)
      return 1.0 / (4.0 + (1.0 - std::pow(p.distance(center) / 2, 4)) * 12.0 *
                              std::pow(std::sin(this->get_time() / 2.5 * numbers::PI), 2));
    else
      return 1.0 / (4.0 + 0.0);
  }
};

template <int dim>
class DemoWaveSpeed : public Function<dim> {
 public:
  DemoC<dim> base;

  virtual ~DemoWaveSpeed() = default;

  virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
    return 1.0 / std::sqrt(base.value(p, component));
  }

  virtual void set_time(double t) {
    Function<dim>::set_time(t);
    base.set_time(t);
  }
};

template <>
const Point<2> DemoF<2>::center = Point<2>(1.0, 0.0);
template <>
const Point<2> DemoC<2>::center = Point<2>(0.0, 2.5);

template <int dim>
void demo() {
  std::ofstream logout("wavepi_demo.log", std::ios_base::trunc);
  deallog.attach(logout);
  deallog.depth_console(3);
  deallog.depth_file(100);
  deallog.precision(3);
  deallog.pop();

  auto triangulation = std::make_shared<Triangulation<dim>>();
  GridGenerator::hyper_cube(*triangulation, -5.0, 5.0);
  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(6);

  double t_end = 10;
  int steps    = t_end * 64;

  double t_start = 0.0, dt = t_end / steps;
  std::vector<double> times;

  for (size_t i = 0; t_start + i * dt <= t_end; i++)
    times.push_back(t_start + i * dt);

  FE_Q<dim> fe(1);
  Quadrature<dim> quad = QGauss<dim>(3);

  auto mesh = std::make_shared<ConstantMesh<dim>>(times, fe, quad, triangulation);

  WaveEquation<dim> wave_eq(mesh);
  wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(std::make_shared<DemoF<dim>>()));

  DemoC<dim> demo_c_cont;
  auto demo_c = std::make_shared<DiscretizedFunction<dim>>(mesh, demo_c_cont);
  wave_eq.set_param_c(demo_c);

  auto sol = wave_eq.run();
  sol.write_pvd("./", "demo_u", "u");
  demo_c->write_pvd("./", "demo_c", "c");

  DemoWaveSpeed<dim> demo_wave_speed_cont;
  auto demo_wave_speed = std::make_shared<DiscretizedFunction<dim>>(mesh, demo_wave_speed_cont);
  demo_wave_speed->write_pvd("./", "demo_wave_speed", "wave speed");
}

int main(int argc, char *argv[]) {
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv);

  try {
    demo<2>();
  } catch (std::exception &exc) {
    std::cerr << "Exception on processing: " << exc.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception!" << std::endl;
    return 1;
  }

  return 0;
}
