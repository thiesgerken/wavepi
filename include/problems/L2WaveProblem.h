/*
 * L2WaveProblem.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_
#define INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_

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

#include <stddef.h>
#include <memory>
#include <vector>

namespace wavepi {
namespace problems {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;
using namespace wavepi::measurements;
using namespace wavepi::util;

template<int dim, typename Measurement>
class L2WaveProblem: public NonlinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
   public:
      virtual ~L2WaveProblem() = default;

      L2WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
            typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
            : stats(std::make_shared<NonlinearProblemStats>()), wave_equation(weq), adjoint_solver(
                  adjoint_solver), right_hand_sides(right_hand_sides), measures(measures) {
         AssertThrow(right_hand_sides.size() == measures.size() && measures.size(), ExcInternalError());
      }

      L2WaveProblem(WaveEquation<dim>& weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
            : L2WaveProblem<dim, Measurement>(weq, right_hand_sides, measures,
                  WaveEquationBase<dim>::WaveEquationAdjoint) {
      }

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>>> derivative(
            const DiscretizedFunction<dim>& param) {
         Assert(this->current_param->relative_error(param) < 1e-10, ExcInternalError());

         std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> derivs;

         for (size_t i = 0; i < right_hand_sides.size(); i++)
            derivs.push_back(derivative(i));

         return std::make_unique<L2WaveProblem<dim, Measurement>::Linearization>(derivs, measures, stats);
      }

      virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& param) {
         LogStream::Prefix p("forward");
         Timer fw_timer;
         Timer meas_timer;

         // save a copy of the parameter
         this->current_param = std::make_shared<DiscretizedFunction<dim>>(param);

         Tuple<Measurement> result;

         for (size_t i = 0; i < right_hand_sides.size(); i++) {
            fw_timer.start();
            auto fwd = forward(i);
            fw_timer.stop();

            meas_timer.start();
            result.push_back(measures[i]->evaluate(fwd));
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

   protected:
      std::shared_ptr<NonlinearProblemStats> stats;

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
                  std::shared_ptr<NonlinearProblemStats> parent_stats)
                  : stats(std::make_shared<LinearProblemStats>()), parent_stats(parent_stats), sub_problems(
                        sub_problems), measures(measures) {
            }

            virtual Tuple<Measurement> forward(const DiscretizedFunction<dim>& h) {
               LogStream::Prefix p("lin_forward");
               Timer fw_timer;
               Timer meas_timer;

               Tuple<Measurement> result;

               for (size_t i = 0; i < measures.size(); i++) {
                  fw_timer.start();
                  auto fw = sub_problems[i]->forward(h);
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

                  adj_timer.start();
                  result += sub_problems[i]->adjoint(am);
                  adj_timer.stop();
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
               return sub_problems[0]->zero();
            }

            virtual std::shared_ptr<LinearProblemStats> get_statistics() {
               return stats;
            }

         private:
            std::shared_ptr<LinearProblemStats> stats;
            std::shared_ptr<NonlinearProblemStats> parent_stats;

            std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim> >>> sub_problems;
            std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;
      };

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_L2WAVEPROBLEM_H_ */
