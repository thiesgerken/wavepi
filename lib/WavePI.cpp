/*
 * WavePI.cpp
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#include <WavePI.h>
#include <base/ConstantMesh.h>
#include <base/Norm.h>
#include <base/Transformation.h>
#include <base/Util.h>
#include <deal.II/base/exceptions.h>
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
#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/PostProcessor.h>
#include <inversion/REGINN.h>
#include <inversion/Regularization.h>
#include <measurements/ConvolutionMeasure.h>
#include <measurements/CubeBoundaryDistribution.h>
#include <measurements/DeltaMeasure.h>
#include <measurements/FieldMeasure.h>
#include <measurements/GridDistribution.h>
#include <measurements/MaskedFieldMeasure.h>
#include <measurements/SensorDistribution.h>
#include <measurements/SensorValues.h>
#include <norms/H1H1.h>
#include <norms/H1L2.h>
#include <norms/H2L2.h>
#include <norms/H2L2PlusL2H1.h>
#include <norms/L2Coefficients.h>
#include <norms/L2L2.h>
#include <norms/LPWrapper.h>
#include <problems/CProblem.h>
#include <problems/ConstantRhoProblem.h>
#include <problems/ConstantCProblem.h>
#include <problems/NuProblem.h>
#include <problems/QProblem.h>
#include <problems/RhoProblem.h>
#include <iostream>
#include <string>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::base;
using namespace wavepi;
using namespace wavepi::inversion;
using namespace wavepi::measurements;
using namespace wavepi::problems;

template <int dim, typename Meas>
WavePI<dim, Meas>::WavePI(std::shared_ptr<SettingsManager> cfg) : cfg(cfg) {
  norm_vector       = std::make_shared<norms::L2Coefficients<dim>>();
  norm_l2l2         = std::make_shared<norms::L2L2<dim>>();
  norm_h1l2         = std::make_shared<norms::H1L2<dim>>(cfg->norm_h1l2_alpha);
  norm_h2l2         = std::make_shared<norms::H2L2<dim>>(cfg->norm_h2l2_alpha, cfg->norm_h2l2_beta);
  norm_h1h1         = std::make_shared<norms::H1H1<dim>>(cfg->norm_h1h1_alpha, cfg->norm_h1h1_gamma);
  norm_h2l2plusl2h1 = std::make_shared<norms::H2L2PlusL2H1<dim>>(
      cfg->norm_h2l2plusl2h1_alpha, cfg->norm_h2l2plusl2h1_beta, cfg->norm_h2l2plusl2h1_gamma);

  switch (cfg->norm_codomain) {
    case SettingsManager::NormType::vector:
      norm_codomain = norm_vector;
      break;
    case SettingsManager::NormType::l2l2:
      norm_codomain = norm_l2l2;
      break;
    case SettingsManager::NormType::h1l2:
      norm_codomain = norm_h1l2;
      break;
    case SettingsManager::NormType::h2l2:
      norm_codomain = norm_h2l2;
      break;
    case SettingsManager::NormType::h1h1:
      norm_codomain = norm_h1h1;
      break;
    case SettingsManager::NormType::h2l2plusl2h1:
      norm_codomain = norm_h2l2plusl2h1;
      break;
    default:
      AssertThrow(false, ExcMessage("unknown norm of codomain"));
  }

  switch (cfg->norm_domain) {
    case SettingsManager::NormType::vector:
      norm_domain = norm_vector;
      break;
    case SettingsManager::NormType::l2l2:
      norm_domain = norm_l2l2;
      break;
    case SettingsManager::NormType::h1l2:
      norm_domain = norm_h1l2;
      break;
    case SettingsManager::NormType::h2l2:
      norm_domain = norm_h2l2;
      break;
    case SettingsManager::NormType::h1h1:
      norm_domain = norm_h1h1;
      break;
    case SettingsManager::NormType::h2l2plusl2h1:
      norm_domain = norm_h2l2plusl2h1;
      break;
    default:
      AssertThrow(false, ExcMessage("WavePI:: unknown norm of codomain"))
  }

  if (cfg->norm_domain_enable_wrapping)
    norm_domain = std::make_shared<norms::LPWrapper<dim>>(norm_domain, cfg->norm_domain_p);

  initial_guess = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_initial_guess, cfg->constants_for_exprs);

  param_rho = MacroFunctionParser<dim>::parse(cfg->expr_param_rho, cfg->constants_for_exprs, cfg->shape_scale);
  param_nu  = MacroFunctionParser<dim>::parse(cfg->expr_param_nu, cfg->constants_for_exprs, cfg->shape_scale);
  param_c   = MacroFunctionParser<dim>::parse(cfg->expr_param_c, cfg->constants_for_exprs, cfg->shape_scale);
  param_q   = MacroFunctionParser<dim>::parse(cfg->expr_param_q, cfg->constants_for_exprs, cfg->shape_scale);

  param_background = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_background, cfg->constants_for_exprs);
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

template <int dim, typename Meas>
std::shared_ptr<Tuple<DiscretizedFunction<dim>>> WavePI<dim, Meas>::interpolate_field(
    std::shared_ptr<SpaceTimeMesh<dim>> target_mesh, std::shared_ptr<Tuple<DiscretizedFunction<dim>>> data) const {
  auto data_new = std::make_shared<Tuple<DiscretizedFunction<dim>>>();

  for (size_t i = 0; i < data->size(); i++)
    data_new->push_back(DiscretizedFunction<dim>(target_mesh, (*data)[i], (*data)[i].get_norm()));

  return data_new;
}

#define get_measure_meas(DIM)                                                                                        \
  template <>                                                                                                        \
  std::shared_ptr<Measure<DiscretizedFunction<DIM>, SensorValues<DIM>>> WavePI<DIM, SensorValues<DIM>>::get_measure( \
      size_t config_idx, std::shared_ptr<SpaceTimeMesh<DIM>> mesh,                                                   \
      std::shared_ptr<Norm<DiscretizedFunction<DIM>>> norm) {                                                        \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM);                                                        \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA);                                                   \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(config_idx, 1));       \
    std::shared_ptr<SensorDistribution<DIM>> sensor_distribution;                                                    \
    if (cfg->sensor_distributions[config_idx] == SettingsManager::SensorDistribution::grid) {                        \
      auto my_distribution = std::make_shared<GridDistribution<DIM>>();                                              \
      my_distribution->get_parameters(*cfg->prm);                                                                    \
      sensor_distribution = my_distribution;                                                                         \
    } else if (cfg->sensor_distributions[config_idx] == SettingsManager::SensorDistribution::cube_boundary) {        \
      auto my_distribution = std::make_shared<CubeBoundaryDistribution<DIM>>();                                      \
      my_distribution->get_parameters(*cfg->prm);                                                                    \
      sensor_distribution = my_distribution;                                                                         \
    } else                                                                                                           \
      AssertThrow(false, ExcInternalError());                                                                        \
    std::shared_ptr<Measure<Param, SensorValues<DIM>>> measure;                                                      \
    if (cfg->measures[config_idx] == SettingsManager::Measure::convolution) {                                        \
      auto my_measure = std::make_shared<ConvolutionMeasure<DIM>>(mesh, sensor_distribution, norm);                  \
      my_measure->get_parameters(*cfg->prm);                                                                         \
      measure = my_measure;                                                                                          \
    } else if (cfg->measures[config_idx] == SettingsManager::Measure::delta) {                                       \
      measure = std::make_shared<DeltaMeasure<DIM>>(mesh, sensor_distribution, norm);                                \
    } else                                                                                                           \
      AssertThrow(false, ExcInternalError());                                                                        \
    cfg->prm->leave_subsection();                                                                                    \
    cfg->prm->leave_subsection();                                                                                    \
    cfg->prm->leave_subsection();                                                                                    \
    return measure;                                                                                                  \
  }

get_measure_meas(1)
    //
    get_measure_meas(2)
    //
    get_measure_meas(3)
//

#define get_measure_cont(D)                                                                                             \
  template <>                                                                                                           \
  std::shared_ptr<Measure<DiscretizedFunction<D>, DiscretizedFunction<D>>>                                              \
  WavePI<D, DiscretizedFunction<D>>::get_measure(size_t config_idx, std::shared_ptr<SpaceTimeMesh<D>> mesh,             \
                                                 std::shared_ptr<Norm<DiscretizedFunction<D>>> norm) {                  \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM);                                                           \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA);                                                      \
    cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(config_idx, 1));          \
    std::shared_ptr<Measure<Param, DiscretizedFunction<D>>> measure;                                                    \
    if (cfg->measures[config_idx] == SettingsManager::Measure::field)                                                   \
      measure = std::make_shared<FieldMeasure<D>>(mesh, norm);                                                          \
    else if (cfg->measures[config_idx] == SettingsManager::Measure::masked_field) {                                     \
      auto mask      = std::make_shared<MacroFunctionParser<D>>(cfg->expr_masks[config_idx], cfg->constants_for_exprs); \
      auto mask_disc = std::make_shared<DiscretizedFunction<D>>(mesh, *mask, norm);                                     \
      measure        = std::make_shared<MaskedFieldMeasure<D>>(mesh, norm, mask_disc);                                  \
    } else                                                                                                              \
      AssertThrow(false, ExcInternalError());                                                                           \
    cfg->prm->leave_subsection();                                                                                       \
    cfg->prm->leave_subsection();                                                                                       \
    cfg->prm->leave_subsection();                                                                                       \
    return measure;                                                                                                     \
  }

        get_measure_cont(1)
    //
    get_measure_cont(2)
    //
    get_measure_cont(3)
//

#define interpolate_data_cont(D)                                                                            \
  template <>                                                                                               \
  void WavePI<D, DiscretizedFunction<D>>::interpolate_data(std::shared_ptr<SpaceTimeMesh<D>> target_mesh) { \
    this->data = interpolate_field(target_mesh, data);                                                      \
  }

        interpolate_data_cont(1)
    //
    interpolate_data_cont(2)
    //
    interpolate_data_cont(3)
//

// do nothing, just keep the sensor values
#define interpolate_data_meas(D)                                                                  \
  template <>                                                                                     \
  void WavePI<D, SensorValues<D>>::interpolate_data(std::shared_ptr<SpaceTimeMesh<D>> target_mesh \
                                                    __attribute((unused))) {}

        interpolate_data_meas(1)
    //
    interpolate_data_meas(2)
    //
    interpolate_data_meas(3)
    //

    template <int dim, typename Meas>
    std::shared_ptr<SpaceTimeMesh<dim>> WavePI<dim, Meas>::initialize_mesh(size_t additional_refines,
                                                                           size_t additional_fe_degrees) const {
  LogStream::Prefix p("initialize_mesh");

  std::shared_ptr<Triangulation<dim>> triangulation = std::make_shared<Triangulation<dim>>();

  if (cfg->shape == SettingsManager::MeshShape::hyper_cube)
    GridGenerator::hyper_cube(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
  else if (cfg->shape == SettingsManager::MeshShape::hyper_l)
    GridGenerator::hyper_L(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
  else if (cfg->shape == SettingsManager::MeshShape::hyper_ball) {
    Point<dim> center =
        make_point(cfg->shape_options["center_x"], cfg->shape_options["center_y"], cfg->shape_options["center_z"]);

    GridGenerator::hyper_ball(*triangulation, center, cfg->shape_options["radius"]);
  } else if (cfg->shape == SettingsManager::MeshShape::cheese) {
    std::vector<unsigned int> holes({1, 1});

    if (dim == 3) holes.push_back(1);

    GridGenerator::cheese(*triangulation, holes);
    dealii::GridTools::scale(cfg->shape_options["scale"], *triangulation);
  } else
    AssertThrow(false, ExcInternalError());

  Util::set_all_boundary_ids(*triangulation, 0);
  triangulation->refine_global(cfg->initial_refines + additional_refines);

  std::vector<double> times = cfg->times;

  for (size_t i = 0; i < additional_refines; i++) {
    std::vector<double> new_times;
    new_times.reserve(2 * times.size());

    for (size_t j = 0; j < times.size(); j++) {
      new_times.push_back(times[j]);
      if (j + 1 < times.size()) new_times.push_back(0.5 * (times[j] + times[j + 1]));
    }

    times = new_times;
  }

  std::shared_ptr<SpaceTimeMesh<dim>> mesh = std::make_shared<ConstantMesh<dim>>(
      times, FE_Q<dim>(cfg->fe_degree + additional_fe_degrees), QGauss<dim>(cfg->quad_order), triangulation);

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
  deallog << "dt: " << cfg->dt / (1 << additional_refines) << std::endl;

  double mem = 0;
  for (size_t i = 0; i < mesh->length(); i++)
    mem += 8 * mesh->n_dofs(i);

  // (in case of !store_derivative, twice this amount otherwise)
  deallog << "expected size of a DiscretizedFunction<" << dim << ">: " << mem / (1024 * 1024) << " MiB" << std::endl;

  auto pattern = mesh->get_sparsity_pattern(0);
  deallog << "Sparsity pattern nnz=" << pattern->n_nonzero_elements() << "="
          << pattern->n_nonzero_elements() / (double)mesh->get_dof_handler(0)->n_dofs()
          << "*n_dofs, bandwidth=" << pattern->bandwidth() << ", max entries per row=" << pattern->max_entries_per_row()
          << std::endl;

  // TODO: also output sparsity pattern

  return mesh;
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::initialize_problem() {
  LogStream::Prefix p("initialize_problem");

  // initialize measurement configurations
  measures.clear();
  pulses.clear();

  for (auto config_idx : cfg->configs)
    measures.push_back(get_measure(config_idx, mesh, norm_codomain));

  for (auto expr : cfg->exprs_rhs)
    pulses.push_back(std::make_shared<MacroFunctionParser<dim>>(expr, cfg->constants_for_exprs));

  AssertThrow(pulses.size() == measures.size(), ExcInternalError());

  wave_eq = std::make_shared<WaveEquation<dim>>(mesh);

  // (the exact parameter is inserted later again as a discretized variant)
  wave_eq->set_param_rho(param_rho, cfg->rho_dynamic);
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
  } else if (cfg->transform == SettingsManager::TransformType::artanh) {
    auto tmp = std::make_shared<ArtanhTransform<dim>>();
    tmp->get_parameters(*cfg->prm);
    transform = tmp;
  } else
    AssertThrow(false, ExcInternalError());

  deallog << "discretizing background parameter" << std::endl;

  // WORKAROUND: force initialization of muparser object
  // (discretizing directly breaks stuff [segfault, bad allocs], maybe due to threading.)
  param_background->evaluate(Point<dim>::unit_vector(1), 0.0);
  auto param_background_discretized = std::make_shared<Param>(mesh, *param_background, norm_domain);

  deallog << "constructing problem" << std::endl;

  switch (cfg->problem_type) {
    case SettingsManager::ProblemType::q:
      /* Reconstruct q */
      param_exact = wave_eq->get_param_q();
      problem =
          std::make_shared<QProblem<dim, Meas>>(*wave_eq, pulses, measures, transform, param_background_discretized);
      break;
    case SettingsManager::ProblemType::c:
      /* Reconstruct c */
      param_exact = wave_eq->get_param_c();
      problem =
          std::make_shared<CProblem<dim, Meas>>(*wave_eq, pulses, measures, transform, param_background_discretized);
      break;
    case SettingsManager::ProblemType::nu:
      /* Reconstruct nu */
      param_exact = wave_eq->get_param_nu();
      problem =
          std::make_shared<NuProblem<dim, Meas>>(*wave_eq, pulses, measures, transform, param_background_discretized);
      break;
    case SettingsManager::ProblemType::rho:
      /* Reconstruct rho */
      param_exact = wave_eq->get_param_rho();
      problem =
          std::make_shared<RhoProblem<dim, Meas>>(*wave_eq, pulses, measures, transform, param_background_discretized);
      break;
    case SettingsManager::ProblemType::rho_constant:
      /* Reconstruct rho */
      param_exact = wave_eq->get_param_rho();
      problem     = std::make_shared<ConstantRhoProblem<dim, Meas>>(*wave_eq, pulses, measures, transform,
                                                                param_background_discretized);
      break;
    case SettingsManager::ProblemType::c_constant:
      /* Reconstruct rho */
      param_exact = wave_eq->get_param_c();
      problem     = std::make_shared<ConstantCProblem<dim, Meas>>(*wave_eq, pulses, measures, transform,
                                                                param_background_discretized);
      break;      
    default:
      AssertThrow(false, ExcInternalError())
  }

  // transform param_exact
  param_exact_untransformed = param_exact;
  param_exact               = transform->transform(param_exact);

  problem->set_norm_domain(norm_domain);
  problem->set_norm_codomain(norm_codomain);
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::synthesize_data() {
  LogStream::Prefix p("generate_data");
  LogStream::Prefix pp("run");  // make logs of forward operator appear in the right level

  deallog << "Discretizing exact parameter" << std::endl;
  param_exact_disc = std::make_shared<DiscretizedFunction<dim>>(mesh, *param_exact);
  param_exact_disc->set_norm(norm_domain);

  deallog << "Computing exact data" << std::endl;
  Tuple<Meas> data_exact = problem->forward(*param_exact_disc);

  double data_exact_norm = data_exact.norm();

  // in itself not wrong, but makes relative errors and noise levels meaningless.
  AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

  data = std::make_shared<Tuple<Meas>>(data_exact);
  data->add(1.0, Tuple<Meas>::noise(data_exact, cfg->epsilon * data_exact_norm));

#ifdef WAVEPI_MPI
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1) {
    size_t mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if (mpi_rank == 0)
      deallog << "Distributing noisy data to other nodes" << std::endl;
    else
      deallog << "Rank " << mpi_rank << " waiting for data from root node" << std::endl;

    for (size_t i = 0; i < data->size(); i++)
      (*data)[i].mpi_bcast(0);

    deallog << "Distribution of data complete." << std::endl;
  }
#endif
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::log_error(DiscretizedFunction<dim>& reconstruction,
                                  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm,
                                  DiscretizedFunction<dim>& exact) {
  reconstruction.set_norm(norm);

  double norm_exact = 0.0;
  double err        = reconstruction.absolute_error(exact, &norm_exact);

  if (norm_exact > 1e-16)
    deallog << "Relative " << norm->name() << " error of the reconstruction: " << err / norm_exact << std::endl;
  else
    deallog << "Absolute " << norm->name() << " error of the reconstruction: " << err << std::endl;

  reconstruction.set_norm(norm_domain);
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::log_error_initial(DiscretizedFunction<dim>& reconstruction_minus_initial,
                                          std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm,
                                          DiscretizedFunction<dim>& exact_minus_initial) {
  reconstruction_minus_initial.set_norm(norm);

  double norm_exact = 0.0;
  double err        = reconstruction_minus_initial.absolute_error(exact_minus_initial, &norm_exact);

  if (norm_exact > 1e-16)
    deallog << "Relative " << norm->name() << " error of (reconstruction - initial guess): " << err / norm_exact
            << std::endl;
  else
    deallog << "Absolute " << norm->name() << " error of (reconstruction - initial guess): " << err << std::endl;

  reconstruction_minus_initial.set_norm(norm_domain);
}

template <int dim, typename Meas>
void WavePI<dim, Meas>::run() {
  Timer timer_data, timer_mesh_problem;

  if (cfg->synthesis_additional_refines > 0 || cfg->synthesis_additional_fe_degrees > 0) {
    timer_data.start();
    deallog << "Generating mesh for data synthesis" << std::endl;
    this->mesh = initialize_mesh(cfg->synthesis_additional_refines, cfg->synthesis_additional_fe_degrees);

    deallog << "Initializing problem for data synthesis" << std::endl;
    initialize_problem();

    deallog << "Generating synthetic data on fine mesh" << std::endl;
    synthesize_data();
    timer_data.stop();

    timer_mesh_problem.start();
    deallog << "Generating mesh" << std::endl;
    auto inversion_mesh = initialize_mesh();
    timer_mesh_problem.stop();

    timer_data.start();
    deallog << "Interpolating data from fine mesh" << std::endl;
    interpolate_data(inversion_mesh);
    timer_data.stop();

    timer_mesh_problem.start();
    deallog << "Initializing problem" << std::endl;
    this->mesh = inversion_mesh;
    initialize_problem();
    timer_mesh_problem.stop();
  } else {
    timer_mesh_problem.start();
    this->mesh = initialize_mesh();

    deallog << "Initializing problem" << std::endl;
    initialize_problem();
    timer_mesh_problem.stop();

    deallog << "Generating synthetic data (inverse crime)" << std::endl;
    timer_data.start();
    synthesize_data();
    timer_data.stop();
  }

  // we do not want the data synthesis to be a part of the stats
  problem->reset_statistics();

  deallog << "wall time for data synthesis        : " << Util::format_duration(timer_data.wall_time()) << std::endl;
  deallog << "wall time for mesh and problem init : " << Util::format_duration(timer_mesh_problem.wall_time())
          << std::endl;

  Timer timer_inversion;
  timer_inversion.start();

  std::shared_ptr<Regularization<Param, Tuple<Meas>, Exact>> regularization;

  deallog.push("Initial Guess");
  auto initial_guess_transformed = transform->transform(initial_guess);
  auto initial_guess_discretized = std::make_shared<Param>(mesh, *initial_guess_transformed);

  // make sure that the initial guess has the right norm
  initial_guess_discretized->set_norm(norm_domain);

  deallog.pop();

  cfg->prm->enter_subsection(SettingsManager::KEY_INVERSION);
  switch (cfg->method) {
    case SettingsManager::NonlinearMethod::reginn:
      regularization =
          std::make_shared<REGINN<Param, Tuple<Meas>, Exact>>(problem, initial_guess_discretized, *cfg->prm);
      break;
    case SettingsManager::NonlinearMethod::nonlinear_landweber:
      regularization = std::make_shared<NonlinearLandweber<Param, Tuple<Meas>, Exact>>(
          problem, initial_guess_discretized, *cfg->prm);
      break;
    default:
      AssertThrow(false, ExcInternalError());
  }
  cfg->prm->leave_subsection();

#ifdef WAVEPI_MPI
  size_t mpi_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
#else
  size_t mpi_rank = 0;
#endif

  if (mpi_rank == 0) {
    regularization->add_listener(std::make_shared<OutputProgressListener<dim, Tuple<Meas>>>(*cfg->prm, transform));
    regularization->add_listener(std::make_shared<StatOutputProgressListener<Param, Tuple<Meas>, Exact>>(*cfg->prm));
  }

  regularization->add_listener(std::make_shared<GenericInversionProgressListener<Param, Tuple<Meas>, Exact>>("i"));
  regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Tuple<Meas>, Exact>>());
  regularization->add_listener(std::make_shared<BoundCheckProgressListener<dim, Tuple<Meas>, Exact>>(*cfg->prm));
  regularization->add_listener(std::make_shared<WatchdogProgressListener<Param, Tuple<Meas>, Exact>>(*cfg->prm));

  regularization->add_post_processor(std::make_shared<BoundEnforcingPostProcessor<dim>>(*cfg->prm));

  cfg->log_parameters();

#ifdef WAVEPI_MPI
  if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > pulses.size())
    deallog << "WARNING: More MPI processes than subproblems" << std::endl;
  else if (pulses.size() % Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) != 0)
    deallog << "WARNING: n_mpi_processes % pulses.size() != 0" << std::endl;
#endif

  // do the inversion
  auto reconstruction = regularization->invert(*data, cfg->tau * cfg->epsilon * data->norm(), param_exact_disc);

  // transform back and output errors in the untransformed setting
  reconstruction = transform->transform_inverse(reconstruction);

  Param param_exact_untrans_disc(mesh, *param_exact_untransformed);
  param_exact_untrans_disc.set_norm(reconstruction.get_norm());

  log_error(reconstruction, norm_domain, param_exact_untrans_disc);

  deallog << "reconstruction error with respect to other norms: " << std::endl;

  log_error(reconstruction, norm_vector, param_exact_untrans_disc);
  log_error(reconstruction, norm_l2l2, param_exact_untrans_disc);
  log_error(reconstruction, norm_h1l2, param_exact_untrans_disc);
  log_error(reconstruction, norm_h1h1, param_exact_untrans_disc);
  log_error(reconstruction, norm_h2l2, param_exact_untrans_disc);

  // same for reconstruction - initial guess
  Param initial_guess_untrans_disc(mesh, *initial_guess);
  initial_guess_untrans_disc.set_norm(reconstruction.get_norm());

  reconstruction -= initial_guess_untrans_disc;
  param_exact_untrans_disc -= initial_guess_untrans_disc;

  log_error_initial(reconstruction, norm_domain, param_exact_untrans_disc);
  log_error_initial(reconstruction, norm_vector, param_exact_untrans_disc);
  log_error_initial(reconstruction, norm_l2l2, param_exact_untrans_disc);
  log_error_initial(reconstruction, norm_h1l2, param_exact_untrans_disc);
  log_error_initial(reconstruction, norm_h1h1, param_exact_untrans_disc);
  log_error_initial(reconstruction, norm_h2l2, param_exact_untrans_disc);

  timer_inversion.stop();

  if (problem->get_statistics()) {
    auto stats = problem->get_statistics();
    deallog << "Statistics: " << std::endl;

    deallog << "forward         : " << stats->calls_forward << " calls, avg "
            << Util::format_duration(stats->time_forward / stats->calls_forward) << " per call, " << std::fixed
            << std::setprecision(2) << (stats->time_forward / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;

    deallog << " lin forward     : " << stats->calls_linearization_forward << " calls, avg "
            << Util::format_duration(stats->time_linearization_forward / stats->calls_linearization_forward)
            << " per call, " << std::fixed << std::setprecision(2)
            << (stats->time_linearization_forward / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;

    deallog << " lin adjoint     : " << stats->calls_linearization_adjoint << " calls, avg "
            << Util::format_duration(stats->time_linearization_adjoint / stats->calls_linearization_adjoint)
            << " per call, " << std::fixed << std::setprecision(2)
            << (stats->time_linearization_adjoint / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;

    deallog << " measure forward : " << stats->calls_measure_forward << " calls, avg "
            << Util::format_duration(stats->time_measure_forward / stats->calls_measure_forward) << " per call, "
            << std::fixed << std::setprecision(2) << (stats->time_measure_forward / timer_inversion.wall_time() * 100)
            << "% of total time" << std::endl;

    deallog << " measure adjoint : " << stats->calls_measure_adjoint << " calls, avg "
            << Util::format_duration(stats->time_measure_adjoint / stats->calls_measure_adjoint) << " per call, "
            << std::fixed << std::setprecision(2) << (stats->time_measure_adjoint / timer_inversion.wall_time() * 100)
            << "% of total time" << std::endl;

    deallog << " duality         : " << stats->calls_duality << " calls, avg "
            << Util::format_duration(stats->time_duality / stats->calls_duality) << " per call, " << std::fixed
            << std::setprecision(2) << (stats->time_duality / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;

    deallog << " IO              : " << Util::format_duration(stats->time_io) << " ≈ " << std::fixed
            << std::setprecision(2) << (stats->time_io / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;

    deallog << " post-processing : " << Util::format_duration(stats->time_postprocessing) << " ≈ " << std::fixed
            << std::setprecision(2) << (stats->time_postprocessing / timer_inversion.wall_time() * 100)
            << "% of total time" << std::endl;

#ifdef WAVEPI_MPI
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1) {
      deallog << " Additional time needed for MPI communication:" << std::endl;
      deallog << "  forward         : average "
              << Util::format_duration(stats->time_forward_communication / stats->calls_forward) << " per call, "
              << std::fixed << std::setprecision(2)
              << (stats->time_forward_communication / timer_inversion.wall_time() * 100) << "% of total time"
              << std::endl;
      deallog << "  lin forward     : avg "
              << Util::format_duration(stats->time_linearization_forward_communication /
                                       stats->calls_linearization_forward)
              << " per call, " << std::fixed << std::setprecision(2)
              << (stats->time_linearization_forward_communication / timer_inversion.wall_time() * 100)
              << "% of total time" << std::endl;
      deallog << "  lin adjoint     : avg "
              << Util::format_duration(stats->time_linearization_adjoint_communication /
                                       stats->calls_linearization_adjoint)
              << " per call, " << std::fixed << std::setprecision(2)
              << (stats->time_linearization_adjoint_communication / timer_inversion.wall_time() * 100)
              << "% of total time" << std::endl;
    }
#endif

    double unaccounted_time = timer_inversion.wall_time();
    unaccounted_time -= stats->time_duality;
    unaccounted_time -= stats->time_forward;
    unaccounted_time -= stats->time_linearization_forward;
    unaccounted_time -= stats->time_linearization_adjoint;
    unaccounted_time -= stats->time_measure_forward;
    unaccounted_time -= stats->time_measure_adjoint;
    unaccounted_time -= stats->time_io;
    unaccounted_time -= stats->time_postprocessing;

#ifdef WAVEPI_MPI
    if (Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) > 1) {
      // (might not be zero if timers have never been started)

      unaccounted_time -= stats->time_forward_communication;
      unaccounted_time -= stats->time_linearization_forward_communication;
      unaccounted_time -= stats->time_linearization_adjoint_communication;
    }
#endif

    // most likely spent doing vector operations, norm evalutations, etc. for the inversion methods
    // and some of it for setup (interpolation, mesh generation, ...)
    deallog << "time not accounted for: " << Util::format_duration(unaccounted_time) << " ≈ " << std::fixed
            << std::setprecision(2) << (unaccounted_time / timer_inversion.wall_time() * 100) << "% of total time"
            << std::endl;
    deallog << "total wall time for the inversion : " << Util::format_duration(timer_inversion.wall_time())
            << std::endl;
    deallog << "total cpu time for the inversion : " << Util::format_duration(timer_inversion.cpu_time()) << std::endl;
  }
}

#ifdef WAVEPI_1D
template class WavePI<1, DiscretizedFunction<1>>;
template class WavePI<1, SensorValues<1>>;
#endif

#ifdef WAVEPI_2D
template class WavePI<2, DiscretizedFunction<2>>;
template class WavePI<2, SensorValues<2>>;
#endif

#ifdef WAVEPI_3D
template class WavePI<3, DiscretizedFunction<3>>;
template class WavePI<3, SensorValues<3>>;
#endif

}  // namespace wavepi
/* namespace wavepi */
