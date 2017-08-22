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

template<int dim> WavePI<dim>::WavePI(std::shared_ptr<SettingsManager> cfg)
      : cfg(cfg) {

   rhs = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_rhs, cfg->constants_for_exprs);

   initial_guess = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_initial_guess,
         cfg->constants_for_exprs);

   param_a = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_a, cfg->constants_for_exprs);
   param_nu = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_nu, cfg->constants_for_exprs);
   param_c = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_c, cfg->constants_for_exprs);
   param_q = std::make_shared<MacroFunctionParser<dim>>(cfg->expr_param_q, cfg->constants_for_exprs);
}

template<> Point<1> WavePI<1>::make_point(double x, double y __attribute__((unused)),
      double z __attribute__((unused))) {
   return Point<1>(x);
}

template<> Point<2> WavePI<2>::make_point(double x, double y, double z __attribute__((unused))) {
   return Point<2>(x, y);
}

template<> Point<3> WavePI<3>::make_point(double x, double y, double z) {
   return Point<3>(x, y, z);
}

template<int dim> void WavePI<dim>::initialize_mesh() {
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
         << 10.0 * sqrt((double ) dim) / pow(triangulation->n_active_cells(), 1.0 / dim) << std::endl;
   deallog << "                maximal = " << dealii::GridTools::maximal_cell_diameter(*triangulation)
         << std::endl;
   deallog << "dt: " << cfg->dt << std::endl;
}

template<int dim> void WavePI<dim>::initialize_problem() {
   LogStream::Prefix p("initialize_problem");

   wave_eq = std::make_shared<WaveEquation<dim>>(mesh);

   wave_eq->set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(rhs));
   wave_eq->set_param_a(param_a);
   wave_eq->set_param_c(param_c);
   wave_eq->set_param_q(param_q);
   wave_eq->set_param_nu(param_nu);
   wave_eq->get_parameters(*cfg->prm);

   switch (cfg->problem_type) {
      case SettingsManager::ProblemType::L2Q:
         /* Reconstruct TestQ */
         param_exact = wave_eq->get_param_q();
         problem = std::make_shared<L2QProblem<dim>>(*wave_eq);
         break;
      case SettingsManager::ProblemType::L2C:
         /* Reconstruct TestC */
         param_exact = wave_eq->get_param_c();
         problem = std::make_shared<L2CProblem<dim>>(*wave_eq);
         break;
      case SettingsManager::ProblemType::L2Nu:
         /* Reconstruct TestNu */
         param_exact = wave_eq->get_param_nu();
         problem = std::make_shared<L2NuProblem<dim>>(*wave_eq);
         break;
      case SettingsManager::ProblemType::L2A:
         /* Reconstruct TestA */
         param_exact = wave_eq->get_param_a();
         problem = std::make_shared<L2AProblem<dim>>(*wave_eq);
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }

//   prm->enter_subsection(KEY_MEASUREMENTS);
//   {
// TODO
//      auto measure_title = prm->get(KEY_MEASUREMENTS_TYPE);
//
//      std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>> measure;
//
//      if (measure_title == "Grid") {
//         auto measure1 = std::make_shared<GridPointMeasure<DiscretizedFunction<dim>, Measurement> >(mesh);
//         measure1->get_parameters(*prm);
//         measure = measure1;
//      } else if (measure_title == "Identical")
//         measure = std::make_shared<IdenticalMeasure<DiscretizedFunction<dim>>>();
//     else if (measure_title != "None")
//            AssertThrow(false, ExcMessage("Unknown Measure: " + measure_title));
//
//         if (measure)
//       problem = std::make_shared<
//            MeasurementProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>, Measurement>>(problem,
//            measure);
//   }
//   prm->leave_subsection();
}

template<int dim> void WavePI<dim>::generate_data() {
   LogStream::Prefix p("generate_data");
   LogStream::Prefix pp(" ");

   Sol data_exact = wave_eq->run();
   data_exact.throw_away_derivative();
   data_exact.set_norm(DiscretizedFunction<dim>::L2L2_Trapezoidal_Mass);
   double data_exact_norm = data_exact.norm();

   // in itself not wrong, but makes relative errors and noise levels meaningless.
   AssertThrow(data_exact_norm > 0, ExcMessage("Exact Data is zero"));

   data = std::make_shared<Sol>(DiscretizedFunction<dim>::noise(data_exact, cfg->epsilon * data_exact_norm));
   data->add(1.0, data_exact);
}

template<int dim> void WavePI<dim>::run() {
   initialize_mesh();
   initialize_problem();
   generate_data();

   std::shared_ptr<Regularization<Param, Sol, Exact>> regularization;

   deallog.push("Initial Guess");
   auto initial_guess_discretized = std::make_shared<Param>(mesh, *initial_guess);
   deallog.pop();

   cfg->prm->enter_subsection(SettingsManager::KEY_INVERSION);
   switch (cfg->method) {
      case SettingsManager::NonlinearMethod::REGINN:
         regularization = std::make_shared<REGINN<Param, Sol, Exact> >(problem, initial_guess_discretized,
               *cfg->prm);
         break;
      case SettingsManager::NonlinearMethod::NonlinearLandweber:
         regularization = std::make_shared<NonlinearLandweber<Param, Sol, Exact> >(problem,
               initial_guess_discretized, *cfg->prm);
         break;
      default:
         AssertThrow(false, ExcInternalError())
   }
   cfg->prm->leave_subsection();

   regularization->add_listener(std::make_shared<GenericInversionProgressListener<Param, Sol, Exact>>("i"));
   regularization->add_listener(std::make_shared<CtrlCProgressListener<Param, Sol, Exact>>());
   regularization->add_listener(std::make_shared<OutputProgressListener<dim>>(*cfg->prm));

   cfg->log();

   regularization->invert(*data, cfg->tau * cfg->epsilon * data->norm(), param_exact);
}

template class WavePI<1> ;
template class WavePI<2> ;
template class WavePI<3> ;

} /* namespace wavepi */
