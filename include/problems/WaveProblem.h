/*
 * WaveProblem.h
 *
 *  Created on: 23.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_PROBLEMS_WAVEPROBLEM_H_
#define INCLUDE_PROBLEMS_WAVEPROBLEM_H_

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>

#include <base/DiscretizedFunction.h>
#include <base/Tuple.h>
#include <forward/WaveEquation.h>
#include <forward/WaveEquationAdjoint.h>
#include <forward/WaveEquationBase.h>
#include <inversion/LinearProblem.h>
#include <inversion/NonlinearProblem.h>
#include <measurements/Measure.h>

#include <stddef.h>
#include <memory>
#include <utility>
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

template <int dim, typename Measurement>
class WaveProblem : public NonlinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
 public:
  virtual ~WaveProblem() = default;

  WaveProblem(const WaveEquation<dim> &weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
              std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
              typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver)
      : stats(std::make_shared<NonlinearProblemStats>()),
        wave_equation(weq),
        adjoint_solver(adjoint_solver),
        right_hand_sides(right_hand_sides),
        measures(measures) {
    AssertThrow(right_hand_sides.size() == measures.size() && measures.size(), ExcInternalError());
  }

  WaveProblem(const WaveEquation<dim> &weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
              std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures)
      : WaveProblem<dim, Measurement>(weq, right_hand_sides, measures, WaveEquationBase<dim>::WaveEquationAdjoint) {}

  virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>>> derivative(
      const DiscretizedFunction<dim> &param) {
    Assert(this->current_param->relative_error(param) < 1e-10, ExcInternalError());

    std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> derivs;

    for (size_t i = 0; i < right_hand_sides.size(); i++)
      derivs.push_back(derivative(i));

    return std::make_unique<WaveProblem<dim, Measurement>::Linearization>(derivs, measures, stats, norm_domain,
                                                                          norm_codomain);
  }

  virtual Tuple<Measurement> forward(const DiscretizedFunction<dim> &param) {
    LogStream::Prefix p("forward");
    AssertThrow(param.get_norm() == norm_domain, ExcMessage("Argument of Forward Operator has unexpected norm"));

    Timer fw_timer;
    Timer meas_timer;
    Timer comm_timer;

    // save a copy of the parameter
    this->current_param = std::make_shared<DiscretizedFunction<dim>>(param);

#ifdef WAVEPI_MPI
    size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    Tuple<Measurement> result;
    result.reserve(right_hand_sides.size());

    deallog << "rank " << rank << " entering parallel section" << std::endl;
    std::vector<std::vector<MPI_Request>> recv_requests(right_hand_sides.size());

    std::vector<DiscretizedFunction<dim>> result_fields(right_hand_sides.size(),
                                                        DiscretizedFunction<dim>(param.get_mesh(), true));

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      result_fields[i].set_norm(norm_codomain);

      if (i % n_procs != rank) result_fields[i].mpi_irecv(i % n_procs, recv_requests[i]);
    }

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      if (i % n_procs != rank) continue;

      deallog << "rank " << rank << " working on field " << i << std::endl;

      fw_timer.start();
      result_fields[i] = forward(i);
      fw_timer.stop();

      // TODO: Isend or Ibcast (and wait for those requests as well) -> better
      // performance for more than one pde / node
      comm_timer.start();
      for (size_t k = 0; k < n_procs; k++)
        if (k != rank) {
          deallog << "rank " << rank << " sending field " << i << " to rank " << k << std::endl;
          result_fields[i].mpi_send(k);
        }
      comm_timer.stop();
    }

    deallog << "rank " << rank << " waiting on Irecvs " << std::endl;

    comm_timer.start();
    for (size_t i = 0; i < right_hand_sides.size(); i++)
      for (size_t j = 0; j < recv_requests[i].size(); j++) {
        MPI_Wait(&recv_requests[i][j], MPI_STATUS_IGNORE);
      }
    comm_timer.stop();

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      if (i % n_procs == rank) continue;

      deallog << "rank " << rank << " looking at field " << i << std::endl;

      fw_timer.start();
      forward(i, result_fields[i]);
      fw_timer.stop();
    }

    deallog << "rank " << rank << " exiting parallel section" << std::endl;

    // everyone does the measurements
    // TODO: could be done in parallel as well
    // would need some kind of allocating function that uses the field
    meas_timer.start();
    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      result_fields[i].throw_away_derivative();
      result.push_back(measures[i]->evaluate(result_fields[i]));
    }
    meas_timer.stop();
#else
    Tuple<Measurement> result;

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      fw_timer.start();
      auto fwd = forward(i);
      fw_timer.stop();

      meas_timer.start();
      result.push_back(measures[i]->evaluate(fwd));
      meas_timer.stop();
    }
#endif

    stats->calls_forward++;
    stats->time_forward += fw_timer.wall_time();

    stats->calls_measure_forward++;
    stats->time_measure_forward += meas_timer.wall_time();

    stats->time_communication += comm_timer.wall_time();

    return result;
  }

  virtual std::shared_ptr<NonlinearProblemStats> get_statistics() { return stats; }

  const Norm &get_norm_domain() const { return norm_domain; }

  /**
   * Set the norm to use for parameters. Defaults to `Norm::L2L2`.
   */
  void set_norm_domain(const Norm &norm_domain) { this->norm_domain = norm_domain; }

  /**
   * Set the norm to use for fields. Defaults to `Norm::L2L2`.
   * Be aware, that this has to match the norm that the measurements expect its
   * inputs to have. Only this way adjoints are calculated correctly.
   *
   */
  const Norm &get_norm_codomain() const { return norm_codomain; }

  void set_norm_codomain(const Norm &norm_codomain) { this->norm_codomain = norm_codomain; }

 protected:
  std::shared_ptr<NonlinearProblemStats> stats;

  Norm norm_domain   = Norm::L2L2;
  Norm norm_codomain = Norm::L2L2;

  WaveEquation<dim> wave_equation;
  typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver;
  std::vector<std::shared_ptr<Function<dim>>> right_hand_sides;

  std::shared_ptr<DiscretizedFunction<dim>> current_param;

  /**
   * Get derivative for param `current_param`. It is guaranteed to be the
   * parameter for which `forward` was last called.
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

  /**
   * `forward(size_t)` could want to save information about the solution,
   * but when using MPI, this is not called. This function gets the function
   * that another process computed and can save it if needed.
   */
  virtual void forward(size_t rhs_index __attribute__((unused)),
                       const DiscretizedFunction<dim> &u __attribute__((unused))) {}

 private:
  std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;

  class Linearization : public LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
   public:
    virtual ~Linearization() = default;

    Linearization(
        std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> sub_problems,
        std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
        std::shared_ptr<NonlinearProblemStats> parent_stats, Norm norm_domain, Norm norm_codomain)
        : stats(std::make_shared<LinearProblemStats>()),
          parent_stats(parent_stats),
          sub_problems(sub_problems),
          measures(measures),
          norm_domain(norm_domain),
          norm_codomain(norm_codomain) {}

    virtual Tuple<Measurement> forward(const DiscretizedFunction<dim> &h) {
      LogStream::Prefix p("lin_forward");
      AssertThrow(h.get_norm() == norm_domain, ExcMessage("Argument of Linearization has unexpected norm"));

      Timer fw_timer;
      Timer meas_timer;
      Timer comm_timer;

      Tuple<Measurement> result;
      result.reserve(measures.size());

#ifdef WAVEPI_MPI
      size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      deallog << "rank " << rank << " entering parallel section" << std::endl;
      std::vector<std::vector<MPI_Request>> recv_requests(measures.size());
      std::vector<DiscretizedFunction<dim>> result_fields(measures.size(), DiscretizedFunction<dim>(h.get_mesh()));

      for (size_t i = 0; i < measures.size(); i++) {
        result_fields[i].set_norm(norm_codomain);

        if (i % n_procs != rank) result_fields[i].mpi_irecv(i % n_procs, recv_requests[i]);
      }

      for (size_t i = 0; i < measures.size(); i++) {
        if (i % n_procs != rank) continue;

        deallog << "rank " << rank << " working on field " << i << std::endl;

        fw_timer.start();
        result_fields[i] = sub_problems[i]->forward(h);
        fw_timer.stop();

        AssertThrow(result_fields[i].get_norm() == norm_codomain,
                    ExcMessage("Output of Linearization has unexpected norm"));

        // TODO: Isend or Ibcast (and wait for those requests as well) ->
        // better performance for more than one pde / node
        comm_timer.start();
        for (size_t k = 0; k < n_procs; k++)
          if (k != rank) {
            deallog << "rank " << rank << " sending field " << i << " to rank " << k << std::endl;
            result_fields[i].mpi_send(k);
          }
        comm_timer.stop();
      }

      deallog << "rank " << rank << " waiting on Irecvs " << std::endl;

      comm_timer.start();
      for (size_t i = 0; i < measures.size(); i++)
        for (size_t j = 0; j < recv_requests[i].size(); j++) {
          MPI_Wait(&recv_requests[i][j], MPI_STATUS_IGNORE);
        }
      comm_timer.stop();

      deallog << "rank " << rank << " exiting parallel section" << std::endl;

      // everyone does the measurements
      // TODO: could be done in parallel as well
      // would need some kind of allocating function that uses the field
      meas_timer.start();
      for (size_t i = 0; i < measures.size(); i++)
        result.push_back(measures[i]->evaluate(result_fields[i]));
      meas_timer.stop();
#else
      for (size_t i = 0; i < measures.size(); i++) {
        fw_timer.start();
        auto fw = sub_problems[i]->forward(h);

        AssertThrow(fw.get_norm() == norm_codomain, ExcMessage("Output of Linearization has unexpected norm"));

        fw_timer.stop();

        meas_timer.start();
        result.push_back(measures[i]->evaluate(fw));
        meas_timer.stop();
      }
#endif

      stats->calls_forward++;
      stats->time_forward += fw_timer.wall_time();

      stats->calls_measure_forward++;
      stats->time_measure_forward += meas_timer.wall_time();

      stats->time_communication += comm_timer.wall_time();

      if (parent_stats) {
        parent_stats->calls_linearization_forward++;
        parent_stats->time_linearization_forward += fw_timer.wall_time();

        parent_stats->calls_measure_forward++;
        parent_stats->time_measure_forward += meas_timer.wall_time();

        parent_stats->time_communication += comm_timer.wall_time();
      }

      return result;
    }

    virtual DiscretizedFunction<dim> adjoint(const Tuple<Measurement> &g) {
      LogStream::Prefix p("lin_adjoint");
      Timer adj_timer;
      Timer adj_meas_timer;
      Timer comm_timer;

      DiscretizedFunction<dim> result(zero());

#ifdef WAVEPI_MPI
      size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      deallog << "rank " << rank << " entering parallel section" << std::endl;

      // sum of local jobs
      DiscretizedFunction<dim> private_result(zero());

      for (size_t i = 0; i < measures.size(); i++) {
        if (i % n_procs != rank) continue;

        deallog << "rank " << rank << " working on task " << i << std::endl;

        adj_meas_timer.start();
        auto am = measures[i]->adjoint(g[i]);
        adj_meas_timer.stop();

        AssertThrow(am.get_norm() == norm_codomain, ExcMessage("Output of Measure adjoint has unexpected norm"));

        adj_timer.start();
        private_result += sub_problems[i]->adjoint(am);
        adj_timer.stop();
      }

      deallog << "rank " << rank << " performing all_reduce" << std::endl;

      comm_timer.start();
      result.mpi_all_reduce(private_result, MPI_SUM);
      comm_timer.stop();

      deallog << "rank " << rank << " exiting parallel section" << std::endl;
#else
      for (size_t i = 0; i < measures.size(); i++) {
        adj_meas_timer.start();
        auto am = measures[i]->adjoint(g[i]);
        adj_meas_timer.stop();

        AssertThrow(am.get_norm() == norm_codomain, ExcMessage("Output of Measure adjoint has unexpected norm"));

        adj_timer.start();
        result += sub_problems[i]->adjoint(am);
        adj_timer.stop();

        // Norm checking for sub_problems[i]->adjoint(am)
        // not necessary, `+=` would fail.
      }
#endif

      stats->calls_adjoint++;
      stats->time_adjoint += adj_timer.wall_time();

      stats->calls_measure_adjoint++;
      stats->time_measure_adjoint += adj_meas_timer.wall_time();

      stats->time_communication += comm_timer.wall_time();

      if (parent_stats) {
        parent_stats->calls_linearization_adjoint++;
        parent_stats->time_linearization_adjoint += adj_timer.wall_time();

        parent_stats->calls_measure_adjoint++;
        parent_stats->time_measure_adjoint += adj_meas_timer.wall_time();

        parent_stats->time_communication += comm_timer.wall_time();
      }

      return result;
    }

    virtual DiscretizedFunction<dim> zero() {
      auto res = sub_problems[0]->zero();
      AssertThrow(res.get_norm() == norm_domain, ExcMessage("sub_problems[0]->zero() has unexpected norm"));

      return res;
    }

    virtual std::shared_ptr<LinearProblemStats> get_statistics() { return stats; }

   private:
    std::shared_ptr<LinearProblemStats> stats;
    std::shared_ptr<NonlinearProblemStats> parent_stats;

    std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> sub_problems;
    std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures;

    Norm norm_domain;
    Norm norm_codomain;
  };
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_WAVEPROBLEM_H_ */
