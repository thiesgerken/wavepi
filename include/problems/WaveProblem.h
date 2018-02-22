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
#include <base/Transformation.h>
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
using namespace wavepi::base;
using namespace wavepi::inversion;
using namespace wavepi::measurements;

template <int dim, typename Measurement>
class WaveProblem : public NonlinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
 public:
  virtual ~WaveProblem() = default;

  WaveProblem(
      const WaveEquation<dim> &weq, std::vector<std::shared_ptr<Function<dim>>> right_hand_sides,
      std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
      std::shared_ptr<Transformation<dim>> transform,
      typename WaveEquationBase<dim>::L2AdjointSolver adjoint_solver = WaveEquationBase<dim>::WaveEquationAdjoint)
      : stats(std::make_shared<NonlinearProblemStats>()),
        wave_equation(weq),
        adjoint_solver(adjoint_solver),
        right_hand_sides(right_hand_sides),
        measures(measures),
        transform(transform) {
    AssertThrow(transform && right_hand_sides.size() == measures.size() && measures.size(), ExcInternalError());
  }

  virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>>> derivative(
      const DiscretizedFunction<dim> &param) {
    Assert(this->current_param_transformed->relative_error(param) < 1e-10, ExcInternalError());

    std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> derivs;

#ifdef WAVEPI_MPI
    size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    // only add those derivatives that I am responsible for
    // (the others did not receive the field, so they cannot do anything anyway)
    // This loop would work without MPI, but it is more readable to have different implementations here.
    for (size_t i = 0; i < right_hand_sides.size(); i++)
      if (i % n_procs == rank)
        derivs.push_back(derivative(i));
      else
        derivs.push_back(0);
#else
    for (size_t i = 0; i < right_hand_sides.size(); i++)
      derivs.push_back(derivative(i));
#endif

    return std::make_unique<WaveProblem<dim, Measurement>::Linearization>(derivs, measures, current_param_transformed,
                                                                          transform, stats, norm_domain, norm_codomain);
  }

  virtual Tuple<Measurement> forward(const DiscretizedFunction<dim> &param) {
    LogStream::Prefix p("forward");
    AssertThrow(param.get_norm() == norm_domain, ExcMessage("Argument of Forward Operator has unexpected norm"));

    Timer fw_timer;
    Timer meas_timer;
    Timer comm_timer;

    // save a copy of the (inverse transformed) parameter
    this->current_param             = std::make_shared<DiscretizedFunction<dim>>(transform->transform_inverse(param));
    this->current_param_transformed = std::make_shared<DiscretizedFunction<dim>>(param);

#ifdef WAVEPI_MPI
    size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
    size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    Tuple<Measurement> result;
    result.reserve(right_hand_sides.size());

    deallog << "rank " << rank << " entering parallel section" << std::endl;
    std::vector<std::vector<MPI_Request>> recv_requests(right_hand_sides.size());

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      result.push_back(measures[i]->zero());

      if (i % n_procs != rank) result[i].mpi_irecv(i % n_procs, recv_requests[i]);
    }

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      if (i % n_procs != rank) continue;

      deallog << "rank " << rank << " working on field " << i << std::endl;

      fw_timer.start();
      auto field = forward(i);
      fw_timer.stop();

      // could be moved into forward again ...
      field.throw_away_derivative();

      meas_timer.start();
      result[i] = measures[i]->evaluate(field);
      meas_timer.stop();

      // Possible improvement: use Isend or Ibcast (and wait for those requests as well) ->
      // better performance for more than one pde / node?
      comm_timer.start();
      for (size_t k = 0; k < n_procs; k++)
        if (k != rank) {
          deallog << "rank " << rank << " sending measurement " << i << " to rank " << k << std::endl;
          result[i].mpi_send(k);
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

    deallog << "rank " << rank << " exiting parallel section" << std::endl;
#else
    Tuple<Measurement> result;

    for (size_t i = 0; i < right_hand_sides.size(); i++) {
      fw_timer.start();
      auto fwd = forward(i);
      fw_timer.stop();

      // could be moved into `forward` again ...
      field.throw_away_derivative();

      meas_timer.start();
      result.push_back(measures[i]->evaluate(fwd));
      meas_timer.stop();
    }
#endif

    stats->calls_forward++;
    stats->time_forward += fw_timer.wall_time();

    stats->calls_measure_forward++;
    stats->time_measure_forward += meas_timer.wall_time();

    stats->time_forward_communication += comm_timer.wall_time();

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
  std::shared_ptr<DiscretizedFunction<dim>> current_param_transformed;
  std::shared_ptr<Transformation<dim>> transform;

  class Linearization : public LinearProblem<DiscretizedFunction<dim>, Tuple<Measurement>> {
   public:
    virtual ~Linearization() = default;

    Linearization(
        std::vector<std::shared_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>>> sub_problems,
        std::vector<std::shared_ptr<Measure<DiscretizedFunction<dim>, Measurement>>> measures,
        std::shared_ptr<DiscretizedFunction<dim>> current_param_transformed,
        std::shared_ptr<Transformation<dim>> transform, std::shared_ptr<NonlinearProblemStats> parent_stats,
        Norm norm_domain, Norm norm_codomain)
        : stats(std::make_shared<LinearProblemStats>()),
          parent_stats(parent_stats),
          sub_problems(sub_problems),
          measures(measures),
          current_param_transformed(current_param_transformed),
          transform(transform),
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

      auto h_transformed = transform->inverse_derivative(*current_param_transformed, h);

#ifdef WAVEPI_MPI
      size_t n_procs = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);
      size_t rank    = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

      deallog << "rank " << rank << " entering parallel section" << std::endl;
      std::vector<std::vector<MPI_Request>> recv_requests(measures.size());

      for (size_t i = 0; i < measures.size(); i++) {
        result.push_back(measures[i]->zero());

        if (i % n_procs != rank) result[i].mpi_irecv(i % n_procs, recv_requests[i]);
      }

      for (size_t i = 0; i < measures.size(); i++) {
        if (i % n_procs != rank) continue;

        deallog << "rank " << rank << " working on field " << i << std::endl;

        fw_timer.start();
        auto field = sub_problems[i]->forward(h_transformed);
        fw_timer.stop();

        AssertThrow(field.get_norm() == norm_codomain, ExcMessage("Output of Linearization has unexpected norm"));

        meas_timer.start();
        result[i] = measures[i]->evaluate(field);
        meas_timer.stop();

        // Possible improvement: use Isend or Ibcast (and wait for those requests as well) ->
        // better performance for more than one pde / node?
        comm_timer.start();
        for (size_t k = 0; k < n_procs; k++)
          if (k != rank) {
            deallog << "rank " << rank << " sending measurement " << i << " to rank " << k << std::endl;
            result[i].mpi_send(k);
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
#else
      for (size_t i = 0; i < measures.size(); i++) {
        fw_timer.start();
        auto fw = sub_problems[i]->forward(h_transformed);

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

      stats->time_forward_communication += comm_timer.wall_time();

      if (parent_stats) {
        parent_stats->calls_linearization_forward++;
        parent_stats->time_linearization_forward += fw_timer.wall_time();

        parent_stats->calls_measure_forward++;
        parent_stats->time_measure_forward += meas_timer.wall_time();

        parent_stats->time_linearization_forward_communication += comm_timer.wall_time();
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

      // one application of dot_transform / dot_transform inverse could be omitted because the above adjoint-calls
      // also need it. Problem: then the subproblems would need to be concerned about the transformation, right now they
      // can just ignore this. Solution: instead of adjoint use transpose?
      if (!std::dynamic_pointer_cast<IdentityTransform<dim>, Transformation<dim>>(transform)) {
        result.dot_transform();
        result = transform->inverse_derivative_transpose(*current_param_transformed, result);
        result.dot_transform_inverse();
      }

      stats->calls_adjoint++;
      stats->time_adjoint += adj_timer.wall_time();

      stats->calls_measure_adjoint++;
      stats->time_measure_adjoint += adj_meas_timer.wall_time();

      stats->time_adjoint_communication += comm_timer.wall_time();

      if (parent_stats) {
        parent_stats->calls_linearization_adjoint++;
        parent_stats->time_linearization_adjoint += adj_timer.wall_time();

        parent_stats->calls_measure_adjoint++;
        parent_stats->time_measure_adjoint += adj_meas_timer.wall_time();

        parent_stats->time_linearization_adjoint_communication += comm_timer.wall_time();
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
    std::shared_ptr<DiscretizedFunction<dim>> current_param_transformed;
    std::shared_ptr<Transformation<dim>> transform;

    Norm norm_domain;
    Norm norm_codomain;
  };
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_PROBLEMS_WAVEPROBLEM_H_ */
