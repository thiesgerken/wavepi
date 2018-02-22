/*
 * WavePI.cpp
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <base/ConstantMesh.h>
#include <base/Transformation.h>
#include <base/Util.h>
#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/PostProcessor.h>
#include <inversion/REGINN.h>
#include <inversion/Regularization.h>
#include <measurements/ConvolutionMeasure.h>
#include <measurements/DeltaMeasure.h>
#include <measurements/SensorDistribution.h>
#include <measurements/SensorValues.h>
#include <problems/AProblem.h>
#include <problems/CProblem.h>
#include <problems/NuProblem.h>
#include <problems/QProblem.h>

#include <WavePI.h>

#include <stddef.h>
#include <tgmath.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::base;
using namespace wavepi::inversion;
using namespace wavepi::measurements;
using namespace wavepi::problems;

template <int dim, typename Meas>
WavePI<dim, Meas>::WavePI(std::shared_ptr<SettingsManager> cfg) : cfg(cfg) {
  measures.clear();
  for (auto config_idx : cfg->configs)
    measures.push_back(get_measure(config_idx));

  for (auto expr : cfg->exprs_rhs)
    pulses.push_back(std::make_shared<MacroFunctionParser<dim>>(expr, cfg->constants_for_exprs));

  AssertThrow(pulses.size() == measures.size(), ExcInternalError());

  DiscretizedFunction<dim>::h1l2_alpha = cfg->norm_h1l2_alpha;
  DiscretizedFunction<dim>::h2l2_alpha = cfg->norm_h2l2_alpha;
  DiscretizedFunction<dim>::h2l2_beta  = cfg->norm_h2l2_beta;

  initial_guess = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_initial_guess, cfg->constants_for_exprs);

  param_a  = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_a, cfg->constants_for_exprs);
  param_nu = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_nu, cfg->constants_for_exprs);
  param_c  = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_c, cfg->constants_for_exprs);
  param_q  = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_q, cfg->constants_for_exprs);
}

template <int dim, typename Meas>
Point<dim> WavePI<dim, Meas>::make_point(double x, double y, double z) {
  switch (dim) {
    case 1:
      return Point<dim>(x);
    case 2:
      return Point<dim>(x, y);
    case 3:
      return Point<dim>(x, y, z);
  }
}

#define get_measure_tuple(DIM)                                                                                       \
  template <>                                                                                                        \
  std::shared_ptr<Measure<DiscretizedFunction<DIM>, SensorValues<DIM>>> WavePI<DIM, SensorValues<DIM>>::get_measure( \
      size_t config_idx) {                                                                                           \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM);                                                        \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA);                                                   \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(config_idx, 1));       \
    std::shared_ptr<SensorDistribution<DIM>> sensor_distribution;                                                    \
    if (cfg->sensor_distributions[config_idx] == SettingsManager::SensorDistribution::grid) {                        \
      auto my_distribution = std::make_shared<GridDistribution<DIM>>();                                              \
      my_distribution->get_parameters(*cfg->prm);                                                                    \
      sensor_distribution = my_distribution;                                                                         \
    } else                                                                                                           \
      AssertThrow(false, ExcInternalError());                                                                        \
    std::shared_ptr<Measure<Param, SensorValues<DIM>>> measure;                                                      \
    if (cfg->measures[config_idx] == SettingsManager::Measure::convolution) {                                        \
      auto my_measure = std::make_shared<ConvolutionMeasure<DIM>>(sensor_distribution);                              \
      my_measure->get_parameters(*cfg->prm);                                                                         \
      measure = my_measure;                                                                                          \
    } else if (cfg->measures[config_idx] == SettingsManager::Measure::delta) {                                       \
      measure = std::make_shared<DeltaMeasure<DIM>>(sensor_distribution);                                            \
    } else                                                                                                           \
      AssertThrow(false, ExcInternalError());                                                                        \
    cfg->prm->leave_subsection();                                                                                    \
    cfg->prm->leave_subsection();                                                                                    \
    cfg->prm->leave_subsection();                                                                                    \
    return measure;                                                                                                  \
  }

get_measure_tuple(1)      //
    get_measure_tuple(2)  //
    get_measure_tuple(3)  //
#define get_measure_cont(D)                                                                                    \
  template <>                                                                                                  \
  std::shared_ptr<Measure<DiscretizedFunction<D>, DiscretizedFunction<D>>>                                     \
  WavePI<D, DiscretizedFunction<D>>::get_measure(size_t config_idx) {                                          \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM);                                                  \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA);                                             \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(config_idx, 1)); \
    std::shared_ptr<Measure<Param, DiscretizedFunction<D>>> measure;                                           \
    if (cfg->measures[config_idx] == SettingsManager::Measure::identical) {                                    \
      measure = std::make_shared<IdenticalMeasure<DiscretizedFunction<D>>>();                                  \
    } else                                                                                                     \
      AssertThrow(false, ExcInternalError());                                                                  \
    cfg->prm->leave_subsection();                                                                              \
    cfg->prm->leave_subsection();                                                                              \
    cfg->prm->leave_subsection();                                                                              \
    return measure;                                                                                            \
  }

    get_measure_cont(1)  //
    get_measure_cont(2)  //
    get_measure_cont(3)  //

    template <int dim, typename Meas>
    void WavePI<dim, Meas>::initialize_mesh() {
  LogStream::Prefix p("initialize_mesh");

  auto triangulation = std::make_shared<Triangulation<dim>>();

  if (cfg->shape == SettingsManager::MeshShape::hyper_cube)
    GridGenerator::hyper_cube(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
  else if (cfg->shape == SettingsManager::MeshShape::hyper_L)
    GridGenerator::hyper_L(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
  else if (cfg->shape == SettingsManager::MeshShape::hyper_ball) {
    Point<dim> center =
        make_point(cfg->shape_options["center_x"], cfg->shape_options["center_y"], cfg->shape_options["center_z"]);

    GridGenerator::hyper_ball(*triangulation, center, cfg->shape_options["radius"]);
  } else if (cfg->shape == SettingsManager::MeshShape::cheese) {
    std::vector<unsigned int> holes({2, 1});

    if (dim == 3) holes.push_back(1);

    GridGenerator::cheese(*triangulation, holes);
    dealii::GridTools::scale(cfg->shape_options["scale"], *triangulation);
  } else
    AssertThrow(false, ExcInternalError());

  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(cfg->initial_refines);

  mesh = std::make_shared<ConstantMesh<dim>>(cfg->times, FE_Q<dim>(cfg->fe_degree), QGauss<dim>(cfg->quad_order),
                                             triangulation);

  // auto a_mesh = std::make_shared<AdaptiveMesh<dim>>(cfg->times, FE_Q<dim>(cfg->fe_degree),
  //      QGauss<dim>(cfg->quad_order), triangulation);
  //
  // mesh = a_mesh;

  //// TEST: flag some cells for refinement, and refine them in some step
  //{
  //   LogStream::Prefix pd("TEST");
  //   for (auto cell : triangulation->active_cell_iterators())
  //      if (cell->center()[1] > 0)
  //         cell->set_refine_flag();
  //
  //   std::vector<bool> ref;
  //   std::vector<bool> coa;
  //
  //   triangulation->save_refine_flags(ref);
  //   triangulation->save_coarsen_flags(coa);
  //
  //   std::vector<Patch> patches = a_mesh->get_forward_patches();
  //   patches[cfg->initial_time_steps / 2].emplace_back(ref, coa);
  //
  //   a_mesh->set_forward_patches(patches);
  //
  //   mesh->get_dof_handler(0);
  //}

  deallog << "Number of active cells: " << triangulation->n_active_cells() << std::endl;
  deallog << "Number of degrees of freedom in the first spatial mesh: " << mesh->get_dof_handler(0)->n_dofs()
          << std::endl;
  deallog << "cell diameters: minimal = " << dealii::GridTools::minimal_cell_diameter(*triangulation);
  deallog << ", maximal = " << dealii::GridTools::maximal_cell_diameter(*triangulation) << std::endl;
  deallog << "dt: " << cfg->dt << std::endl;

  double mem = 0;
  for (size_t i = 0; i < mesh->length(); i++)
    mem += 8 * mesh->n_dofs(i);

  // (in case of !store_derivative)
  deallog << "expected size of a DiscretizedFunction<dim>: " << mem / (1024 * 1024) << " MiB" << std::endl;
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::initialize_problem() {
  LogStream::Prefix p("initialize_problem");

  wave_eq = std::make_shared<WaveEquation<dim>>(mesh);

  wave_eq->set_param_a(param_a);
  wave_eq->set_param_c(param_c);
  wave_eq->set_param_q(param_q);
  wave_eq->set_param_nu(param_nu);
  wave_eq->get_parameters(*cfg->prm);

  if (cfg->transform == SettingsManager::TransformType::identity)
    transform = std::make_shared<IdentityTransform<dim>>();
  else if (cfg->transform == SettingsManager::TransformType::log) {
    auto tmp = std::make_shared<LogTransform<dim>>();
    tmp->get_parameters(*cfg->prm);
    transform = tmp;
  } else
    AssertThrow(false, ExcInternalError());

  switch (cfg->problem_type) {
    case SettingsManager::ProblemType::L2Q:
      /* Reconstruct q */
      param_exact = wave_eq->get_param_q();
      problem     = std::make_shared<QProblem<dim, Meas>>(*wave_eq, pulses, measures, transform);
      break;
    case SettingsManager::ProblemType::L2C:
      /* Reconstruct c */
      param_exact = wave_eq->get_param_c();
      problem     = std::make_shared<CProblem<dim, Meas>>(*wave_eq, pulses, measures, transform);
      break;
    case SettingsManager::ProblemType::L2Nu:
      /* Reconstruct nu */
      param_exact = wave_eq->get_param_nu();
      problem     = std::make_shared<NuProblem<dim, Meas>>(*wave_eq, pulses, measures, transform);
      break;
    case SettingsManager::ProblemType::L2A:
      /* Reconstruct a */
      param_exact = wave_eq->get_param_a();
      problem     = std::make_shared<AProblem<dim, Meas>>(*wave_eq, pulses, measures, transform);
      break;
    default:
      AssertThrow(false, ExcInternalError());
  }

  // transform param_exact
  param_exact_untransformed = param_exact;
  param_exact               = transform->transform(param_exact);

  problem->set_norm_domain(cfg->norm_domain);
  problem->set_norm_codomain(cfg->norm_codomain);
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::generate_data() {
  LogStream::Prefix p("generate_data");
  LogStream::Prefix pp("run");  // make logs of forward operator appear in the right level

  DiscretizedFunction<dim> param_exact_disc(mesh, *param_exact);
  param_exact_disc.set_norm(cfg->norm_domain);
  Tuple<Meas> data_exact = problem->forward(param_exact_disc);

  double data_exact_norm = data_exact.norm();

  // in itself not wrong, but makes relative errors and noise levels meaningless.
  AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

  data = std::make_shared<Tuple<Meas>>(data_exact);
  data->add(1.0, Tuple<Meas>::noise(data_exact, cfg->epsilon * data_exact_norm));
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::run() {
  Timer timer_total;
  timer_total.start();

  initialize_mesh();
  initialize_problem();
  generate_data();

  std::shared_ptr<Regularization<Param, Tuple<Meas>, Exact>> regularization;

  deallog.push("Initial Guess");
  auto initial_guess_transformed = transform->transform(initial_guess);
  auto initial_guess_discretized = std::make_shared<Param>(mesh, *initial_guess_transformed);

  // make sure that the initial guess has the right norm
  initial_guess_discretized->set_norm(cfg->norm_domain);

  deallog.pop();

  cfg->prm->enter_subsection(SettingsManager::KEY_INVERSION);
  switch (cfg->method) {
    case SettingsManager::NonlinearMethod::REGINN:
      regularization =
          std::make_shared<REGINN<Param, Tuple<Meas>, Exact>>(problem, initial_guess_discretized, *cfg->prm);
      break;
    case SettingsManager::NonlinearMethod::NonlinearLandweber:
      regularization = std::make_shared<NonlinearLandweber<Param, Tuple<Meas>, Exact>>(
          problem, initial_guess_discretized, *cfg->prm);
      break;
    default:
      AssertThrow(false, ExcInternalError());
  }
  cfg->prm->leave_subsection();

  // Output only for master node
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    regularization->add_listener(std::make_shared<OutputProgressListener<dim, Tuple<Meas>>>(*cfg->prm, transform));
    regularization->add_listener(std::make_shared<StatOutputProgressListener<Param, Tuple<Meas>, Exact>>(*cfg->prm));
  }

  regularization->add_listener(std::make_shared<GenericInversionProgressListener<Param, Tuple<Meas>, Exact>>("i"));
  regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Tuple<Meas>, Exact>>());
  regularization->add_listener(std::make_shared<BoundCheckProgressListener<dim, Tuple<Meas>>>(*cfg->prm));
  regularization->add_listener(std::make_shared<WatchdogProgressListener<Param, Tuple<Meas>, Exact>>(*cfg->prm));

  regularization->add_post_processor(std::make_shared<BoundEnforcingPostProcessor<dim>>(*cfg->prm));

  cfg->log_parameters();

  auto reconstruction = regularization->invert(*data, cfg->tau * cfg->epsilon * data->norm(), param_exact);

  // transform back and output errors in the untransformed setting
  transform->transform_inverse(reconstruction);

  double norm_exact = 0.0;
  double err        = reconstruction.absolute_error(*param_exact_untransformed, &norm_exact);

  if (norm_exact > 1e-16)
    deallog << "Relative error of the reconstruction" << err / norm_exact << std::endl;
  else
    deallog << "Absolute error of the reconstruction" << err << std::endl;

  if (problem->get_statistics()) {
    auto stats = problem->get_statistics();
    LogStream::Prefix p("stats");

    deallog << "forward              : " << stats->calls_forward << " calls, average "
            << stats->time_forward / stats->calls_forward << " s per call" << std::endl;
    deallog << "linearization forward: " << stats->calls_linearization_forward << " calls, average "
            << stats->time_linearization_forward / stats->calls_linearization_forward << " s per call" << std::endl;
    deallog << "linearization adjoint: " << stats->calls_linearization_adjoint << " calls, average "
            << stats->time_linearization_adjoint / stats->calls_linearization_adjoint << " s per call" << std::endl;
    deallog << "measure forward      : " << stats->calls_measure_forward << " calls, average "
            << stats->time_measure_forward / stats->calls_measure_forward << " s per call" << std::endl;
    deallog << "measure adjoint      : " << stats->calls_measure_adjoint << " calls, average "
            << stats->time_measure_adjoint / stats->calls_measure_adjoint << " s per call" << std::endl;

    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1)
      deallog << "mpi communication    : " << stats->time_communication << " s, average "
              << stats->time_communication /
                     (stats->calls_forward + stats->calls_linearization_forward + stats->calls_linearization_adjoint)
              << " s per pde solution" << std::endl;

    deallog << "total wall time      : " << (int)std::floor(timer_total.wall_time() / 60) << "min "
            << (int)std::floor(std::fmod(timer_total.wall_time(), 60)) << "s" << std::endl;
  }
}

template class WavePI<1, DiscretizedFunction<1>>;
template class WavePI<1, SensorValues<1>>;

template class WavePI<2, DiscretizedFunction<2>>;
template class WavePI<2, SensorValues<2>>;

template class WavePI<3, DiscretizedFunction<3>>;
template class WavePI<3, SensorValues<3>>;

} /* namespace wavepi */
