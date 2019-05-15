/*
 * Transformation.cpp
 *
 *  Created on: 15.02.2018
 *      Author: thies
 */

#include <base/ComposedFunction.h>
#include <base/Transformation.h>
#include <deal.II/base/function_parser.h>

namespace wavepi {
namespace base {

template <int dim>
DiscretizedFunction<dim> IdentityTransform<dim>::transform(const DiscretizedFunction<dim> &param) {
  return param;
}

template <int dim>
std::shared_ptr<LightFunction<dim>> IdentityTransform<dim>::transform(const std::shared_ptr<LightFunction<dim>> param) {
  return param;
}

template <int dim>
DiscretizedFunction<dim> IdentityTransform<dim>::transform_inverse(const DiscretizedFunction<dim> &param) {
  return param;
}

template <int dim>
DiscretizedFunction<dim> IdentityTransform<dim>::inverse_derivative(const DiscretizedFunction<dim> &param
                                                                    __attribute((unused)),
                                                                    const DiscretizedFunction<dim> &h) {
  return h;
}

template <int dim>
DiscretizedFunction<dim> IdentityTransform<dim>::inverse_derivative_transpose(const DiscretizedFunction<dim> &param
                                                                              __attribute((unused)),
                                                                              const DiscretizedFunction<dim> &g) {
  return g;
}

template <int dim>
void LogTransform<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("LogTransform");
  prm.declare_entry("lower bound", "0.0", Patterns::Double(),
                    "transformation function for log transform is φ(x) = log(x-x₀), where x₀ is the lower bound you "
                    "want to enforce.");
  prm.leave_subsection();
}

template <int dim>
void LogTransform<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("LogTransform");
  lower_bound = prm.get_double("lower bound");
  prm.leave_subsection();
}

template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::transform(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++) {
      Assert(param[i][j] > lower_bound,
             ExcMessage("LogTransform::transform called on param with entries <= lower bound"));
      tmp[i][j] = std::log(param[i][j] - lower_bound);
    }

  return tmp;
}

template <int dim>
std::shared_ptr<LightFunction<dim>> LogTransform<dim>::transform(const std::shared_ptr<LightFunction<dim>> param) {
  return std::make_shared<ComposedFunction<dim>>(param,
                                                 std::make_shared<LogTransform<dim>::TransformFunction>(lower_bound));
}

template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::transform_inverse(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = std::exp(param[i][j]) + lower_bound;

  return tmp;
}

template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::inverse_derivative(const DiscretizedFunction<dim> &param,
                                                               const DiscretizedFunction<dim> &h) {
  AssertThrow(!param.has_derivative() && !h.has_derivative(), ExcMessage("Not transforming derivatives!"));
  AssertThrow(param.get_mesh() == h.get_mesh(), ExcMessage("LogTransform: meshes must match"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = std::exp(param[i][j]) * h[i][j];

  return tmp;
}
template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                         const DiscretizedFunction<dim> &g) {
  return inverse_derivative(param, g);
}

template <int dim>
void ArtanhTransform<dim>::declare_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ArtanhTransform");
  prm.declare_entry("lower bound", "0.0", Patterns::Double(),
                    "transformation function for log transform is φ: (a,b) → ℝ,  φ(x) = tanh⁻¹((2x-(a+b))/(b-a)) (pointwise), where a and b are the bounds you would like to enforce");  
  prm.declare_entry("upper bound", "1.0", Patterns::Double(),
                    "transformation function for log transform is φ: (a,b) → ℝ,  φ(x) = tanh⁻¹(((2x-(a+b))/(b-a)) (pointwise), where a and b are the bounds you would like to enforce");
  prm.leave_subsection();
}

template <int dim>
void ArtanhTransform<dim>::get_parameters(ParameterHandler &prm) {
  prm.enter_subsection("ArtanhTransform");
  lower_bound = prm.get_double("lower bound");
  upper_bound = prm.get_double("upper bound");
  prm.leave_subsection();
}

template <int dim>
DiscretizedFunction<dim> ArtanhTransform<dim>::transform(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++) {
      Assert(param[i][j] > lower_bound,
             ExcMessage("ArtanhTransform::transform called on param with entries <= lower bound"));
      Assert(param[i][j] < upper_bound,
             ExcMessage("ArtanhTransform::transform called on param with entries >= upper bound"));

      tmp[i][j] = std::atanh((2 * param[i][j] - (lower_bound + upper_bound)) / (upper_bound - lower_bound));
    }

  return tmp;
}

template <int dim>
std::shared_ptr<LightFunction<dim>> ArtanhTransform<dim>::transform(const std::shared_ptr<LightFunction<dim>> param) {
  return std::make_shared<ComposedFunction<dim>>(param,
                                                 std::make_shared<ArtanhTransform<dim>::TransformFunction>(lower_bound, upper_bound));
}

template <int dim>
DiscretizedFunction<dim> ArtanhTransform<dim>::transform_inverse(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = (upper_bound-lower_bound)/2 * std::tanh(param[i][j]) + (lower_bound+upper_bound)/2;
      
  return tmp;
}

template <int dim>
DiscretizedFunction<dim> ArtanhTransform<dim>::inverse_derivative(const DiscretizedFunction<dim> &param,
                                                               const DiscretizedFunction<dim> &h) {
  AssertThrow(!param.has_derivative() && !h.has_derivative(), ExcMessage("Not transforming derivatives!"));
  AssertThrow(param.get_mesh() == h.get_mesh(), ExcMessage("ArtanhTransform: meshes must match"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++) {
      double u = std::cosh(param[i][j]);
      tmp[i][j] = (upper_bound-lower_bound)/2 * 1/(u*u) * h[i][j];
    }

  return tmp;
}

template <int dim>
DiscretizedFunction<dim> ArtanhTransform<dim>::inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                         const DiscretizedFunction<dim> &g) {
  return inverse_derivative(param, g);
}

template class Transformation<1>;
template class Transformation<2>;
template class Transformation<3>;

template class IdentityTransform<1>;
template class IdentityTransform<2>;
template class IdentityTransform<3>;

template class LogTransform<1>;
template class LogTransform<2>;
template class LogTransform<3>;

template class ArtanhTransform<1>;
template class ArtanhTransform<2>;
template class ArtanhTransform<3>;

}  // namespace base
}  // namespace wavepi
