/*
 * WavePI.cpp
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#include <deal.II/base/exceptions.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <forward/AdaptiveMesh.h>
#include <forward/L2RightHandSide.h>

#include <inversion/InversionProgress.h>
#include <inversion/NonlinearLandweber.h>
#include <inversion/Regularization.h>
#include <inversion/REGINN.h>

#include <problems/L2AProblem.h>
#include <problems/L2CProblem.h>
#include <problems/L2NuProblem.h>
#include <problems/L2QProblem.h>

#include <util/GridTools.h>

#include <WavePI.h>

#include <tgmath.h>
#include <ccomplex>
#include <cmath>
#include <iostream>
#include <vector>

namespace wavepi {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::problems;
using namespace wavepi::util;

template<int dim, typename Meas> WavePI<dim, Meas>::WavePI(std::shared_ptr<SettingsManager> cfg)
      : cfg(cfg) {

   initialize_measurements();

   for (auto expr : cfg->exprs_rhs)
      pulses.push_back(std::make_shared<MacroFunctionParser<dim>>(expr, cfg->constants_for_exprs));

   AssertThrow(pulses.size() == measures.size(), ExcInternalError());

   initial_guess = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_initial_guess,
         cfg->constants_for_exprs);

   param_a = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_a, cfg->constants_for_exprs);
   param_nu = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_nu, cfg->constants_for_exprs);
   param_c = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_c, cfg->constants_for_exprs);
   param_q = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_q, cfg->constants_for_exprs);
}

template<int dim, typename Meas> Point<dim> WavePI<dim, Meas>::make_point(double x, double y, double z) {
   switch (dim) {
      case 1:
         return Point<dim>(x);
      case 2:
         return Point<dim>(x, y);
      case 3:
         return Point<dim>(x, y, z);
   }
}

#define initialize_measurements_tuple(D) \
template<> void WavePI<D, Tuple<double>>::initialize_measurements() { \
  measures.clear();  \
   for (size_t i = 0; i < cfg->num_configurations; i++) { \
      cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA); \
      cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(i, 1)); \
      if (cfg->measures[i] == SettingsManager::Measure::grid) { \
         auto measure = std::make_shared<GridPointMeasure<D>>(); \
         measure->get_parameters(*cfg->prm); \
         measures.push_back(measure); \
      } else \
         AssertThrow(false, ExcInternalError()); \
      cfg->prm->leave_subsection(); \
      cfg->prm->leave_subsection(); \
   } \
}

initialize_measurements_tuple(1)
initialize_measurements_tuple(2)
initialize_measurements_tuple(3)

#define initialize_measurements_cont(D) \
template<> void WavePI<D, DiscretizedFunction<D>>::initialize_measurements() { \
  measures.clear();  \
   for (size_t i = 0; i < cfg->num_configurations; i++) { \
      cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA); \
      cfg->prm->enter_subsection(SettingsManager::KEY_PROBLEM_DATA_I + Utilities::int_to_string(i, 1)); \
      if (cfg->measures[i] == SettingsManager::Measure::identical) { \
         measures.push_back(std::make_shared<IdenticalMeasure<DiscretizedFunction<D>> >()); \
      } else \
         AssertThrow(false, ExcInternalError()); \
      cfg->prm->leave_subsection(); \
      cfg->prm->leave_subsection(); \
   } \
}

initialize_measurements_cont(1)
initialize_measurements_cont(2)
initialize_measurements_cont(3)


template<int dim, typename Meas> void WavePI<dim, Meas>::initialize_mesh() {
   LogStream::Prefix p("initialize_mesh");

   auto triangulation = std::make_shared<Triangulation<dim>>();

   if (cfg->shape == SettingsManager::MeshShape::hyper_cube)
      GridGenerator::hyper_cube(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
   else if (cfg->shape == SettingsManager::MeshShape::hyper_L)
      GridGenerator::hyper_L(*triangulation, cfg->shape_options["left"], cfg->shape_options["right"]);
   else if (cfg->shape == SettingsManager::MeshShape::hyper_ball) {
      Point<dim> center = make_point(cfg->shape_options["center_x"], cfg->shape_options["center_y"],
            cfg->shape_options["center_z"]);

      GridGenerator::hyper_ball(*triangulation, center, cfg->shape_options["radius"]);
   } else if (cfg->shape == SettingsManager::MeshShape::cheese) {
      std::vector<unsigned int> holes( { 2, 1 });

      if (dim == 3)
         holes.push_back(1);

      GridGenerator::cheese(*triangulation, holes);
      dealii::GridTools::scale(cfg->shape_options["scale"], *triangulation);
   } else
      AssertThrow(false, ExcInternalError())

   wavepi::util::GridTools::set_all_boundary_ids(*triangulation, 0);
   triangulation->refine_global(cfg->initial_refines);

   //   mesh = std::make_shared<AdaptiveMesh<dim>>(times, FE_Q<dim>(fe_degree), QGauss<dim>(quad_order),
   //         triangulation);

   auto a_mesh = std::make_shared<AdaptiveMesh<dim>>(cfg->times, FE_Q<dim>(cfg->fe_degree),
         QGauss<dim>(cfg->quad_order), triangulation);

   mesh = a_mesh;

   // TEST: flag some cells for refinement, and refine them in some step
   {
      LogStream::Prefix pd("TEST");
      for (auto cell : triangulation->active_cell_iterators())
         if (cell->center()[1] > 0)
            cell->set_refine_flag();

      std::vector<bool> ref;
      std::vector<bool> coa;

      triangulation->save_refine_flags(ref);
      triangulation->save_coarsen_flags(coa);

      std::vector<Patch> patches = a_mesh->get_forward_patches();
      patches[cfg->initial_time_steps / 2].emplace_back(ref, coa);

      a_mesh->set_forward_patches(patches);

      mesh->get_dof_handler(0);
   }

   deallog << "Number of active cells: " << triangulation->n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom in the first spatial mesh: " << mesh->get_dof_handler(0)->n_dofs()
         << std::endl;
   deallog << "cell diameters: minimal = " << dealii::GridTools::minimal_cell_diameter(*triangulation)
         << std::endl;
   deallog << "                average = "
         << 10.0 * sqrt((double) dim) / pow(triangulation->n_active_cells(), 1.0 / dim) << std::endl;
   deallog << "                maximal = " << dealii::GridTools::maximal_cell_diameter(*triangulation)
         << std::endl;
   deallog << "dt: " << cfg->dt << std::endl;
}

template<int dim, typename Meas> void WavePI<dim, Meas>::initialize_problem() {
   LogStream::Prefix p("initialize_problem");

   wave_eq = std::make_shared<WaveEquation<dim>>(mesh);

   wave_eq->set_param_a(param_a);
   wave_eq->set_param_c(param_c);
   wave_eq->set_param_q(param_q);
   wave_eq->set_param_nu(param_nu);
   wave_eq->get_parameters(*cfg->prm);

   switch (cfg->problem_type) {
      case SettingsManager::ProblemType::L2Q:
         /* Reconstruct TestQ */
         param_exact = wave_eq->get_param_q();
//         problem = std::make_shared < L2QProblem < dim >> (*wave_eq); // TODO
         break;
      case SettingsManager::ProblemType::L2C:
         /* Reconstruct TestC */
         param_exact = wave_eq->get_param_c();
//         problem = std::make_shared < L2CProblem < dim >> (*wave_eq); // TODO
         break;
      case SettingsManager::ProblemType::L2Nu:
         /* Reconstruct TestNu */
         param_exact = wave_eq->get_param_nu();
//         problem = std::make_shared < L2NuProblem < dim >> (*wave_eq); // TODO
         break;
      case SettingsManager::ProblemType::L2A:
         /* Reconstruct TestA */
         param_exact = wave_eq->get_param_a();
         problem = std::make_shared<L2AProblem<dim, Meas>>(*wave_eq, pulses, measures);
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }
}

template<int dim, typename Meas> void WavePI<dim, Meas>::generate_data() {
   LogStream::Prefix p("generate_data");
   LogStream::Prefix pp(" ");

   DiscretizedFunction<dim> param_exact_disc(mesh, *param_exact);
   Tuple<Meas> data_exact = problem->forward(param_exact_disc);
   double data_exact_norm = data_exact.norm();

   // in itself not wrong, but makes relative errors and noise levels meaningless.
   AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

   data = std::make_shared<Tuple<Meas>>(data_exact);
   data->add(1.0, Tuple<Meas>::noise(data_exact, cfg->epsilon * data_exact_norm));
}

template<int dim, typename Meas> void WavePI<dim, Meas>::run() {
   initialize_mesh();
   initialize_problem();
   generate_data();

   std::shared_ptr<Regularization<Param, Tuple<Meas>, Exact>> regularization;

   deallog.push("Initial Guess");
   auto initial_guess_discretized = std::make_shared<Param>(mesh, *initial_guess);
   deallog.pop();

   cfg->prm->enter_subsection(SettingsManager::KEY_INVERSION);
   switch (cfg->method) {
      case SettingsManager::NonlinearMethod::REGINN:
         regularization = std::make_shared<REGINN<Param, Tuple<Meas>, Exact> >(problem,
               initial_guess_discretized, *cfg->prm);
         break;
      case SettingsManager::NonlinearMethod::NonlinearLandweber:
         regularization = std::make_shared<NonlinearLandweber<Param, Tuple<Meas>, Exact> >(problem,
               initial_guess_discretized, *cfg->prm);
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }
   cfg->prm->leave_subsection();

   regularization->add_listener(
         std::make_shared<GenericInversionProgressListener<Param, Tuple<Meas>, Exact>>("i"));
   regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Tuple<Meas>, Exact>>());
   regularization->add_listener(std::make_shared<OutputProgressListener<dim, Tuple<Meas>>>(*cfg->prm));

   cfg->log_parameters();

   regularization->invert(*data, cfg->tau * cfg->epsilon * data->norm(), param_exact);
}

template class WavePI<1, DiscretizedFunction<1>> ;
template class WavePI<1, Tuple<double>> ;

template class WavePI<2, DiscretizedFunction<2>> ;
template class WavePI<2, Tuple<double>> ;

template class WavePI<3, DiscretizedFunction<3>> ;
template class WavePI<3, Tuple<double>> ;

} /* namespace wavepi */
