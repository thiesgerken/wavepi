#include <iostream>

#include <forward/WaveEquation.h>
#include <inversion/WaveProblem.h>
#include <inversion/Landweber.h>

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::inversion;

template<int dim>
class TestF: public Function<dim> {
   public:
      TestF()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<>
double TestF<1>::value(const Point<1> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5) && (p.distance(Point<1>(1.0)) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<2>::value(const Point<2> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5) && (p.distance(Point<2>(1.0, 0.5)) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<>
double TestF<3>::value(const Point<3> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));
   if ((this->get_time() <= 0.5) && (p.distance(Point<3>(1.0, 0.5, 0.0)) < 0.4))
      return std::sin(this->get_time() * 2 * numbers::PI);
   else
      return 0.0;
}

template<int dim>
class TestC: public Function<dim> {
   public:
      TestC()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<int dim>
double rho(const Point<dim> &p, double t);

template<>
double rho(const Point<1> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<1>(t - 3.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<>
double rho(const Point<2> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<2>(t - 3.0, t - 2.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<>
double rho(const Point<3> &p, double t) {
// return  p.distance(Point<2>(1.0*std::cos(2*numbers::PI * t / 8.0), 1.0*std::sin(2*numbers::PI * t / 8.0))) < 0.65 ? 20.0 : 1.0;
   return p.distance(Point<3>(t - 3.0, t - 2.0, 0.0)) < 1.2 ? 1.0 / 3.0 : 1.0;
}

template<int dim>
double TestC<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   return 1.0 / (rho(p, this->get_time()) * 4.0);
}

template<int dim>
class TestA: public Function<dim> {
   public:
      TestA()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<int dim>
double TestA<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   return 1.0 / rho(p, this->get_time());
}

template<int dim>
class TestQ: public Function<dim> {
   public:
      TestQ()
            : Function<dim>() {
      }
      double value(const Point<dim> &p, const unsigned int component = 0) const;
};

template<int dim>
double TestQ<dim>::value(const Point<dim> &p, const unsigned int component) const {
   Assert(component == 0, ExcIndexRange(component, 0, 1));

   return p[0]+this->get_time();
}

template<int dim>
class QProblem: public WaveProblem<dim> {
   public:
      virtual ~QProblem() {
      }

      QProblem(WaveEquation<dim> weq)
            : WaveProblem<dim>(weq) {
      }

      virtual std::unique_ptr<LinearProblem<DiscretizedFunction<dim>, DiscretizedFunction<dim>>> derivative(
            DiscretizedFunction<dim>& f) const {
         // TODO
      }

      virtual DiscretizedFunction<dim> forward(DiscretizedFunction<dim>& q) {
         this->wave_equation.set_param_q(&q);

         return this->wave_equation.run();
      }
};

template<int dim>
void test() {
   std::ofstream logout("wave_test.log");
   deallog.attach(logout);
   deallog.depth_console(2);
   deallog.depth_file(100);
   deallog.precision(3);
   deallog.pop();
   // deallog.log_execution_time(true);

   Triangulation<dim> triangulation;

   // GridGenerator::cheese(triangulation, std::vector<unsigned int>( { 1, 1 }));
   GridGenerator::hyper_cube(triangulation, -5, 5);
   triangulation.refine_global(5);

   FE_Q<dim> fe(1);
   DoFHandler<dim> dof_handler;
   dof_handler.initialize(triangulation, fe);

   deallog << "Number of active cells: " << triangulation.n_active_cells() << std::endl;
   deallog << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

   double t_start = 0.0, t_end = 6.0, dt = 1.0 / 64.0;
   std::vector<double> times;

   for (size_t i = 0; t_start + i * dt <= t_end; i++)
      times.push_back(t_start + i * dt);

   WaveEquation<dim> wave_eq(&dof_handler, times);

   TestF<dim> rhs;
   L2RightHandSide<dim> l2rhs(&rhs);
   wave_eq.set_right_hand_side(&l2rhs);

   TestA<dim> a;
   wave_eq.set_param_a(&a);

   TestC<dim> c;
   wave_eq.set_param_c(&c);

   TestQ<dim> q;
   DiscretizedFunction<dim> q_exact = DiscretizedFunction<dim>::discretize(&q, times, &dof_handler);

   wave_eq.set_param_q(&q);

   QProblem<dim> my_problem(wave_eq);
   double epsilon = 1e-2;

   DiscretizedFunction<dim> data_exact = wave_eq.run();

   DiscretizedFunction<dim> data = DiscretizedFunction<dim>::noise(data_exact, epsilon * data_exact.norm());
   data.add(1.0, data_exact);

   DiscretizedFunction<dim> initialGuess(data_exact); // same grids!
   initialGuess = 0.0;

   Landweber<DiscretizedFunction<dim>, DiscretizedFunction<dim>> lw(&my_problem, initialGuess, 1e-2);

   lw.invert(data, 1.5 * epsilon, &q_exact);

   deallog.timestamp();
}

int main() {
   try {
      test<2>();
   } catch (std::exception &exc) {
      std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what() << std::endl << "Aborting!" << std::endl
            << "----------------------------------------------------" << std::endl;

      return 1;
   } catch (...) {
      std::cerr << std::endl << std::endl << "----------------------------------------------------" << std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!" << std::endl
            << "----------------------------------------------------" << std::endl;
      return 1;
   }

   return 0;
}
