/*
 * discretized_params.cpp
 *
 *  Created on: 11.07.2017
 *      Author: thies
 */

#include "gtest/gtest.h"

#include <forward/WaveEquation.h>

namespace {

using namespace dealii;
using namespace wavepi::forward;

template<int dim>
class TestF: public Function<dim> {
public:
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		Assert(component == 0, ExcIndexRange(component, 0, 1));
		if ((this->get_time() <= 1) && (p.norm() < 0.5))
			return std::sin(this->get_time() * 2 * numbers::PI);
		else
			return 0.0;
	}
};

template<int dim>
double rho(const Point<dim> &p, double t) {
	return p.norm() + t + 1.0;
}

template<int dim>
double c_squared(const Point<dim> &p, double t) {
	double tmp = p.norm() * t + 1.0;

	return tmp * tmp;
}

template<int dim>
class TestC: public Function<dim> {
public:
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		Assert(component == 0, ExcIndexRange(component, 0, 1));

		return 1.0 / (rho(p, this->get_time()) * c_squared(p, this->get_time()));
	}
};

template<int dim>
class TestA: public Function<dim> {
public:
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		Assert(component == 0, ExcIndexRange(component, 0, 1));

		return 1.0 / rho(p, this->get_time());
	}
};

template<int dim>
class TestNu: public Function<dim> {
public:
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		Assert(component == 0, ExcIndexRange(component, 0, 1));

		return p[0] * this->get_time();
	}
};

template<int dim>
class TestQ: public Function<dim> {
public:
	double value(const Point<dim> &p, const unsigned int component = 0) const {
		Assert(component == 0, ExcIndexRange(component, 0, 1));

		return p.norm() < 0.3 ? this->get_time() : 0.0;
	}
};

template<int dim>
class DiscretizedFunctionDisguise: public Function<dim> {
public:
	DiscretizedFunctionDisguise(std::shared_ptr<DiscretizedFunction<dim>> base) :
			base(base) {
	}

	double value(const Point<dim> &p, const unsigned int component = 0) const {
		return base->value(p, component);
	}

	void set_time(const double new_time) {
		Function<dim>::set_time(new_time);
		base->set_time(new_time);
	}
private:
	std::shared_ptr<DiscretizedFunction<dim>> base;
};

// checks, whether the matrix assembly of discretized parameters works correct
// (by supplying DiscretizedFunctions and DiscretizedFunctionDisguises)
template<int dim>
void run_discretized_test(int fe_order, int quad_order, int refines) {
	std::ofstream logout("wavepi_test.log");
	deallog.attach(logout);
	deallog.depth_console(0);
	deallog.depth_file(100);
	deallog.precision(3);
	deallog.pop();

	Timer timer;

	Triangulation<dim> triangulation;
	GridGenerator::hyper_cube(triangulation, -1, 1);
	triangulation.refine_global(refines);

	FE_Q<dim> fe(fe_order);
	Quadrature<dim> quad = QGauss<dim>(quad_order); // exact in poly degree 2n-1 (needed: fe_dim^3)

	DoFHandler<dim> dof_handler;
	dof_handler.initialize(triangulation, fe);

	double t_start = 0.0, t_end = 2.0, dt = 1.0 / 32.0;
	std::vector<double> times;

	for (size_t i = 0; t_start + i * dt <= t_end; i++)
		times.push_back(t_start + i * dt);

	deallog << std::endl << "Number of active cells: "
			<< triangulation.n_active_cells() << std::endl;
	deallog << "Number of degrees of freedom: " << dof_handler.n_dofs()
			<< std::endl;
	deallog << "Number of time steps: " << times.size() << std::endl
			<< std::endl;

	WaveEquation<dim> wave_eq(&dof_handler, times, quad);

	/* continuous */

	wave_eq.set_param_a(std::make_shared<TestA<dim>>());
	wave_eq.set_param_c(std::make_shared<TestC<dim>>());
	wave_eq.set_param_q(std::make_shared<TestQ<dim>>());
	wave_eq.set_param_nu(std::make_shared<TestNu<dim>>());

	wave_eq.set_right_hand_side(
			std::make_shared<L2RightHandSide<dim>>(
					std::make_shared<TestF<dim>>()));

	timer.restart();
	DiscretizedFunction<dim> sol_cont = wave_eq.run();
	timer.stop();
	deallog << "continuous params: " << std::fixed << timer.wall_time()
			<< " s of wall time" << std::endl;
	EXPECT_GT(sol_cont.norm(), 0.0);

	/* discretized */

	TestC<dim> c;
	auto c_disc = std::make_shared<DiscretizedFunction<dim>>(c, times,
			&dof_handler);
	wave_eq.set_param_c(c_disc);

	TestA<dim> a;
	auto a_disc = std::make_shared<DiscretizedFunction<dim>>(a, times,
			&dof_handler);
	wave_eq.set_param_a(a_disc);

	TestQ<dim> q;
	auto q_disc = std::make_shared<DiscretizedFunction<dim>>(q, times,
			&dof_handler);
	wave_eq.set_param_q(q_disc);

	TestNu<dim> nu;
	auto nu_disc = std::make_shared<DiscretizedFunction<dim>>(nu, times,
			&dof_handler);
	wave_eq.set_param_nu(nu_disc);

	TestF<dim> f;
	auto f_disc = std::make_shared<DiscretizedFunction<dim>>(f, times,
			&dof_handler);
	wave_eq.set_right_hand_side(std::make_shared<L2RightHandSide<dim>>(f_disc));

	timer.restart();
	DiscretizedFunction<dim> sol_disc = wave_eq.run();
	timer.stop();
	deallog << "all discretized: " << std::fixed << timer.wall_time()
			<< " s of wall time" << std::endl;
	EXPECT_GT(sol_disc.norm(), 0.0);

	/* discretized, q disguised */

	auto c_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(
			c_disc);
	auto a_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(
			a_disc);
	auto q_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(
			q_disc);
	auto nu_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(
			nu_disc);
	auto f_disguised = std::make_shared<DiscretizedFunctionDisguise<dim>>(
			f_disc);

	wave_eq.set_param_q(q_disguised);

	timer.restart();
	DiscretizedFunction<dim> sol_disc_except_q = wave_eq.run();
	timer.stop();
	deallog << "all discretized, q disguised: " << std::fixed
			<< timer.wall_time() << " s of wall time" << std::endl;
	EXPECT_GT(sol_disc_except_q.norm(), 0.0);

	/* discretized, a disguised */

	wave_eq.set_param_a(a_disguised);
	wave_eq.set_param_q(q_disc);

	timer.restart();
	DiscretizedFunction<dim> sol_disc_except_a = wave_eq.run();
	timer.stop();
	deallog << "all discretized, a disguised: " << std::fixed
			<< timer.wall_time() << " s of wall time" << std::endl;
	EXPECT_GT(sol_disc_except_a.norm(), 0.0);

	/* disguised */

	wave_eq.set_param_nu(nu_disguised);
	wave_eq.set_param_q(q_disguised);
	wave_eq.set_param_a(a_disguised);
	wave_eq.set_param_c(c_disguised);
	wave_eq.set_right_hand_side(
			std::make_shared<L2RightHandSide<dim>>(f_disguised));

	timer.restart();
	DiscretizedFunction<dim> sol_disguised = wave_eq.run();
	timer.stop();
	deallog << "all discretized and disguised as continuous: " << std::fixed
			<< timer.wall_time() << " s of wall time" << std::endl << std::endl;
	EXPECT_GT(sol_disguised.norm(), 0.0);

	/* results */

	DiscretizedFunction<dim> tmp(sol_cont);
	tmp -= sol_disc;
	double err_cont_vs_disc = tmp.norm() / sol_cont.norm();

	deallog << "rel. error between continuous and full discrete: "
			<< std::scientific << err_cont_vs_disc << std::endl;
	EXPECT_LT(err_cont_vs_disc, 1.0);

	tmp.sadd(0.0, 1.0, sol_disc_except_q);
	tmp -= sol_disguised;
	double err_disguised_vs_disc_except_q = tmp.norm() / sol_disguised.norm();

	deallog
			<< "rel. error between disguised discrete and discrete (q disguised): "
			<< std::scientific << err_disguised_vs_disc_except_q << std::endl;
	EXPECT_LT(err_disguised_vs_disc_except_q, 1e-7);

	tmp.sadd(0.0, 1.0, sol_disc_except_a);
	tmp -= sol_disguised;
	double err_disguised_vs_disc_except_a = tmp.norm() / sol_disguised.norm();

	deallog
			<< "rel. error between disguised discrete and discrete (a disguised): "
			<< std::scientific << err_disguised_vs_disc_except_a << std::endl;
	EXPECT_LT(err_disguised_vs_disc_except_a, 1e-7);

	tmp.sadd(0.0, 1.0, sol_disc);
	tmp -= sol_disguised;
	double err_disguised_vs_disc = tmp.norm() / sol_disguised.norm();

	deallog << "rel. error between disguised discrete and full discrete: "
			<< std::scientific << err_disguised_vs_disc << std::endl
			<< std::endl;
	EXPECT_LT(err_disguised_vs_disc, 1e-7);
}
}

TEST(WaveEquationTest, DiscretizedParameters1DFE1) {
	run_discretized_test<1>(1, 3, 8);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE1) {
	run_discretized_test<2>(1, 3, 4);
}

TEST(WaveEquationTest, DiscretizedParameters3DFE1) {
	run_discretized_test<3>(1, 3, 2);
}

TEST(WaveEquationTest, DiscretizedParameters1DFE2) {
	run_discretized_test<1>(2, 4, 8);
}

TEST(WaveEquationTest, DiscretizedParameters2DFE2) {
	run_discretized_test<2>(2, 4, 4);
}

TEST(WaveEquationTest, DiscretizedParameters3DFE2) {
	run_discretized_test<3>(2, 4, 1);
}

TEST(WaveEquationTest, L2Adjointness) {
	EXPECT_EQ(0, 0); // TODO
}
