/*
 * WaveProblem.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_WAVEPROBLEM_H_
#define INCLUDE_PROBLEMS_WAVEPROBLEM_H_

#include <deal.II/base/function.h>
#include <deal.II/base/timer.h>

#include <forward/DiscretizedFunction.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>

#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>

#include <util/Tuple.h>
#include <measurements/Measure.h>

#include <deal.II/base/mpi.h>

#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {

/**
 * Collection of inverse problems that we want to look at.
 */
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::measurements;
using namespace wavepi::util;

template<int dim, typename Measurement>
class WaveProblem: public NonlinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
   public:
      virtual ~WaveProblem() = default;

      WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
            : stats(std::make_shared<NonlinearProblemStats>()), wave_equation(weq), adjoint_solver(
                  adjoint_solver), right_hand_sides(right_hand_sides), measures(measures) {
         AssertThrow(right_hand_sides.size() == measures.size() && measures.size(), ExcInternalError());
      }

      WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
            : WaveProblem<dim, Measurement>(weq, right_hand_sides, measures,
                  WaveEquationBase<dim>::WaveEquationAdjoint) {
      }

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>>> derivative(
            const DiscretizedFunction<dim>& param) {
         Assert(this->current_param->relative_error(param) < 1e-10, ExcInternalError());

         std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> derivs;

         for (size_t i = 0; i < right_hand_sides.size(); i++)
            derivs.push_back(derivative(i));

         return std::make_unique<WaveProblem<dim, Measurement>::Linearization>(derivs, measures, stats,
               norm_domain, norm_codomain);
      }

      virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& param) {
         LogStream::Prefix p("forward");
         AssertThrow(param.get_norm() == norm_domain,
               ExcMessage("Argument of Forward Operator has unexpected norm"))

         Timer fw_timer;
         Timer meas_timer;

         // save a copy of the parameter
         this->current_param = std::make_shared<DiscretizedFunction<dim>>(param);

         /*
          Tuple<Measurement> result;

          for (size_t i = 0; i < right_hand_sides.size(); i++) {
          fw_timer.start();
          auto fwd = forward(i);
          fw_timer.stop();

          meas_timer.start();
          result.push_back(measures[i]->evaluate(fwd));
          meas_timer.stop();
          }
          */

         size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
         size_t rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

         Tuple<Measurement> result;
         result.reserve(right_hand_sides.size());

         bool zero_available = true;
         for (size_t i = 0; i < right_hand_sides.size(); i++)
            zero_available &= measures[i]->zero_available();

         if (zero_available) {
            std::vector<std::vector<MPI_Win>> result_windows;
            result_windows.reserve(right_hand_sides.size());

            for (size_t i = 0; i < right_hand_sides.size(); i++) {
               result.push_back(measures[i]->zero());
               result_windows.push_back(result[i].make_windows());
            }

            for (size_t i = 0; i < right_hand_sides.size(); i++) {
               if (i % n_procs != rank)
                  continue;

               std::cout << rank << " working on field+meas " << i << std::endl;

               fw_timer.start();
               auto fwd = forward(i);
               fw_timer.stop();

               meas_timer.start();
               result[i] = measures[i]->evaluate(fwd);
               meas_timer.stop();

               for (size_t k = 0; k < n_procs; k++)
                  if (k != rank)
                     result[i].copy_to(result_windows[i], k);
            }

            for (size_t i = 0; i < right_hand_sides.size(); i++)
               for (size_t j = 0; j < result_windows[i].size(); j++) {
                  MPI_Win_wait(result_windows[i][j]);
                  MPI_Win_complete(result_windows[i][j]);
                  MPI_Win_free(&result_windows[i][j]);
               }
         } else {
            Tuple<DiscretizedFunction<dim>> result_fields;
            std::vector<std::vector<MPI_Win>> result_windows;

            result_fields.reserve(right_hand_sides.size());
            result_windows.reserve(right_hand_sides.size());

            for (size_t i = 0; i < right_hand_sides.size(); i++) {
               result_fields.push_back(DiscretizedFunction<dim>(param.get_mesh()));
               result_windows.push_back(result_fields[i].make_windows());

               for (size_t j = 0; j < result_windows[i].size(); j++)
                  MPI_Win_fence(0, result_windows[i][j]);
            }

            for (size_t i = 0; i < right_hand_sides.size(); i++) {
               if (i % n_procs != rank)
                  continue;

               std::cout << rank << " working on field " << i << std::endl;

               fw_timer.start();
               result_fields[i] = forward(i);
               fw_timer.stop();

               for (size_t k = 0; k < n_procs; k++)
                  if (k != rank) {
                     std::cout << rank << " sending " << i << " to " << k << std::endl;
                     result_fields[i].copy_to(result_windows[i], k);
                  }
            }

            for (size_t i = 0; i < right_hand_sides.size(); i++)
               for (size_t j = 0; j < result_windows[i].size(); j++) {
//                  MPI_Win_wait(result_windows[i][j]);
//                  MPI_Win_complete(result_windows[i][j]);

                  //std::cout << rank << " flushing " << i << "." << j << std::endl;
                  //MPI_Win_flush_all(result_windows[i][j]);
               }

            std::cout << rank << " barrier" << std::endl;
                       MPI_Barrier(MPI_COMM_WORLD);

            for (size_t i = 0; i < right_hand_sides.size(); i++)
               for (size_t j = 0; j < result_windows[i].size(); j++) {
                  //                  MPI_Win_wait(result_windows[i][j]);
                  //                  MPI_Win_complete(result_windows[i][j]);

                  std::cout << rank << " fencing " << i << "." << j << std::endl;
                  MPI_Win_fence(0, result_windows[i][j]);
                  MPI_Win_free(&result_windows[i][j]);
               }

            // everyone does the measurements
            meas_timer.start();

            for (size_t i = 0; i < right_hand_sides.size(); i++)
               result.push_back(measures[i]->evaluate(result_fields[i]));

            meas_timer.stop();
         }

         stats->calls_forward++;
         stats->time_forward += fw_timer.wall_time();

         stats->calls_measure_forward++;
         stats->time_measure_forward += meas_timer.wall_time();

         return result;
      }

      virtual std::shared_ptr<NonlinearProblemStats> get_statistics() {
         return stats;
      }

      const Norm& get_norm_domain() const {
         return norm_domain;
      }

      /**
       * Set the norm to use for parameters. Defaults to `Norm::L2L2`.
       */
      void set_norm_domain(const Norm& norm_domain) {
         this->norm_domain = norm_domain;
      }

      /**
       * Set the norm to use for fields. Defaults to `Norm::L2L2`.
       * Be aware, that this has to match the norm that the measurements expect its inputs to have.
       * Only this way adjoints are calculated correctly.
       *
       */
      const Norm& get_norm_codomain() const {
         return norm_codomain;
      }

      void set_norm_codomain(const Norm& norm_codomain) {
         this->norm_codomain = norm_codomain;
      }

   protected:
      std::shared_ptr<NonlinearProblemStats> stats;

      Norm norm_domain = Norm::L2L2;
      Norm norm_codomain = Norm::L2L2;

      WaveEquation<dim> wave_equation;
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;
      std::vector<std::shared_ptr<Function<dim>>> right_hand_sides;

      std::shared_ptr<DiscretizedFunction<dim>> current_param;

      /**
       * Get derivative for param `current_param`. It is guaranteed to be the parameter for which `forward` was last called.
       *
       * @param rhs_index index of right hand side to use
       */
      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            size_t rhs_index) = 0;

      /**
       * Evaluate forward operator for param `current_param`.
       *
       * @param rhs_index index of right hand side to use
       */
      virtual DiscretizedFunction<dim> forward(size_t rhs_index) = 0;

   private:
      std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;

      class Linearization: public LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
         public:
            virtual ~Linearization() = default;

            Linearization(
                  std::vector<
                        std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> sub_problems,
                  std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
                  std::shared_ptr<NonlinearProblemStats> parent_stats, Norm norm_domain, Norm norm_codomain)
                  : stats(std::make_shared<LinearProblemStats>()), parent_stats(parent_stats), sub_problems(
                        sub_problems), measures(measures), norm_domain(norm_domain), norm_codomain(
                        norm_codomain) {
            }

            virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& h) {
               LogStream::Prefix p("lin_forward");
               AssertThrow(h.get_norm() == norm_domain,
                     ExcMessage("Argument of Linearization has unexpected norm"))

               Timer fw_timer;
               Timer meas_timer;

               Tuple<Measurement> result;

               for (size_t i = 0; i < measures.size(); i++) {
                  fw_timer.start();
                  auto fw = sub_problems[i]->forward(h);

                  AssertThrow(fw.get_norm() == norm_codomain,
                        ExcMessage("Output of Linearization has unexpected norm"))

                  fw_timer.stop();

                  meas_timer.start();
                  result.push_back(measures[i]->evaluate(fw));
                  meas_timer.stop();
               }

               stats->calls_forward++;
               stats->time_forward += fw_timer.wall_time();

               stats->calls_measure_forward++;
               stats->time_measure_forward += meas_timer.wall_time();

               if (parent_stats) {
                  parent_stats->calls_linearization_forward++;
                  parent_stats->time_linearization_forward += fw_timer.wall_time();

                  parent_stats->calls_measure_forward++;
                  parent_stats->time_measure_forward += meas_timer.wall_time();
               }

               return result;
            }

            virtual DiscretizedFunction<dim> adjoint(const Tuple<Measurement>& g) {
               LogStream::Prefix p("lin_adjoint");
               Timer adj_timer;
               Timer adj_meas_timer;

               DiscretizedFunction<dim> result(zero());

               for (size_t i = 0; i < measures.size(); i++) {
                  adj_meas_timer.start();
                  auto am = measures[i]->adjoint(g[i]);
                  adj_meas_timer.stop();

                  AssertThrow(am.get_norm() == norm_codomain,
                        ExcMessage("Output of Measure adjoint has unexpected norm"))

                  adj_timer.start();
                  result += sub_problems[i]->adjoint(am);
                  adj_timer.stop();

                  // Norm checking for sub_problems[i]->adjoint(am)
                  // not necessary, `+=` would fail.
               }

               stats->calls_adjoint++;
               stats->time_adjoint += adj_timer.wall_time();

               stats->calls_measure_adjoint++;
               stats->time_measure_adjoint += adj_meas_timer.wall_time();

               if (parent_stats) {
                  parent_stats->calls_linearization_adjoint++;
                  parent_stats->time_linearization_adjoint += adj_timer.wall_time();

                  parent_stats->calls_measure_adjoint++;
                  parent_stats->time_measure_adjoint += adj_meas_timer.wall_time();
               }

               return result;
            }

            virtual DiscretizedFunction<dim> zero() {
               auto res = sub_problems[0]->zero();
               AssertThrow(res.get_norm() == norm_domain,
                     ExcMessage("sub_problems[0}->zero() has unexpected norm"))

               return res;
            }

            virtual std::shared_ptr<LinearProblemStats> get_statistics() {
               return stats;
            }

         private:
            std::shared_ptr<LinearProblemStats> stats;
            std::shared_ptr<NonlinearProblemStats> parent_stats;

            std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> sub_problems;
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;

            Norm norm_domain;
            Norm norm_codomain;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_WAVEPROBLEM_H_ */
