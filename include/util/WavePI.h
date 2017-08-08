/*
 * WavePI.h
 *
 *  Created on: 08.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_UTIL_WAVEPI_H_
#define INCLUDE_UTIL_WAVEPI_H_

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/tria.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>
#include <forward/WaveEquation.h>

#include <inversion/NonlinearProblem.h>

#include <memory>
#include <string>

namespace wavepi {
namespace util {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
class TestNu: public Function<dim> {
   public:
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p[0] * this->get_time();
      }
};

template<int dim>
class TestF: public Function<dim> {
   public:
      TestF()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         if ((this->get_time() <= 0.5) && (p.distance(actor_position) < 0.4))
            return std::sin(this->get_time() * 2 * numbers::PI);
         else
            return 0.0;
      }
   private:
      static const Point<dim> actor_position;
};

template<> const Point<1> TestF<1>::actor_position = Point<1>(1.0);
template<> const Point<2> TestF<2>::actor_position = Point<2>(1.0, 0.5);
template<> const Point<3> TestF<3>::actor_position = Point<3>(1.0, 0.5, 0.0);

template<int dim>
double rho(const Point<dim> &p, double t);

template<>
double rho(const Point<1> &p, double t) {
   return p.distance(Point<1>(t - 3.0)) < 1.0 ? 1.5 : 1.0;
}

template<>
double rho(const Point<2> &p, double t) {
   return p.distance(Point<2>(t - 3.0, t - 2.0)) < 1.2 ? 1.5 : 1.0;
}

template<>
double rho(const Point<3> &p, double t) {
   return p.distance(Point<3>(t - 3.0, t - 2.0, 0.0)) < 1.2 ? 1.5 : 1.0;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / (rho(p, this->get_time()) * 1.0);
      }
};

template<int dim>
class TestA: public Function<dim> {
   public:
      TestA()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return 1.0 / rho(p, this->get_time());
      }
};

template<int dim>
class TestQ: public Function<dim> {
   public:
      TestQ()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const {
         Assert(component == 0, ExcIndexRange(component, 0, 1));

         return p.distance(q_position) < 1.0 ? 10 * std::sin(this->get_time() / 2 * 2 * numbers::PI) : 0.0;
      }

      static const Point<dim> q_position;
};

template<> const Point<1> TestQ<1>::q_position = Point<1>(-1.0);
template<> const Point<2> TestQ<2>::q_position = Point<2>(-1.0, 0.5);
template<> const Point<3> TestQ<3>::q_position = Point<3>(-1.0, 0.5, 0.0);

template<int dim>
class WavePI {
   public:
      static void declare_parameters(ParameterHandler &prm);

      WavePI(ParameterHandler &prm);

      void run();

      void initialize_mesh();
      void initialize_problem();
      void generate_data();

   private:
      static const std::string KEY_FE_DEGREE;
      static const std::string KEY_QUAD_ORDER;
      static const std::string KEY_PROBLEM_TYPE;
      static const std::string KEY_END_TIME;
      static const std::string KEY_EPSILON;
      static const std::string KEY_TAU;
      static const std::string KEY_INITIAL_REFINES;
      static const std::string KEY_INITIAL_TIME_STEPS;

      // possible problems
      enum class ProblemType {
         L2Q = 1, L2A = 2, L2C = 3, L2Nu = 4
      };

      FE_Q<dim> fe;
      QGauss<dim> quad;
      ProblemType problem_type;

      double end_time;
      double epsilon;
      double tau;
      int initial_refines;
      int initial_time_steps;

      Triangulation<dim> triangulation;
      std::shared_ptr<DoFHandler<dim>> dof_handler;
      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<WaveEquation<dim>> wave_eq;

      using Param = DiscretizedFunction<dim>;
      using Sol = DiscretizedFunction<dim>;

      std::shared_ptr<NonlinearProblem<Param, Sol>> problem;
      std::shared_ptr<Function<dim>> param_exact_cont;
      std::shared_ptr<Param> param_exact;
      std::shared_ptr<Param> initialGuess;
      std::shared_ptr<Sol> data; // noisy data

};

} /* namespace util */
} /* namespace wavepi */

#endif /* INCLUDE_UTIL_WAVEPI_H_ */
