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
std::shared_ptr<Function<dim>> IdentityTransform<dim>::transform(const std::shared_ptr<Function<dim>> param) {
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
DiscretizedFunction<dim> LogTransform<dim>::transform(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), false, param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++) {
      Assert(param[i][j] > 0, ExcMessage("LogTransform::transform called on param with negative entries"));
      tmp[i][j] = std::log(param[i][j] - 0.1);
    }

  return tmp;
}

template <int dim>
std::shared_ptr<Function<dim>> LogTransform<dim>::transform(const std::shared_ptr<Function<dim>> param) {
  return std::make_shared<ComposedFunction<dim>>(param, std::make_shared<LogTransform<dim>::TransformFunction>());
}

template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::transform_inverse(const DiscretizedFunction<dim> &param) {
  AssertThrow(!param.has_derivative(), ExcMessage("Not transforming derivatives!"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), false, param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = std::exp(param[i][j]) + 0.1;

  return tmp;
}

template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::inverse_derivative(const DiscretizedFunction<dim> &param,
                                                               const DiscretizedFunction<dim> &h) {
  AssertThrow(!param.has_derivative() && !h.has_derivative(), ExcMessage("Not transforming derivatives!"));
  AssertThrow(param.get_mesh() == h.get_mesh(), ExcMessage("LogTransform: meshes must match"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), false, param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = std::exp(param[i][j]) * h[i][j];

  return tmp;
}
template <int dim>
DiscretizedFunction<dim> LogTransform<dim>::inverse_derivative_transpose(const DiscretizedFunction<dim> &param,
                                                                         const DiscretizedFunction<dim> &g) {
  AssertThrow(!param.has_derivative() && !g.has_derivative(), ExcMessage("Not transforming derivatives!"));
  AssertThrow(param.get_mesh() == g.get_mesh(), ExcMessage("LogTransform: meshes must match"));

  DiscretizedFunction<dim> tmp(param.get_mesh(), false, param.get_norm());

  for (size_t i = 0; i < param.length(); i++)
    for (size_t j = 0; j < param[i].size(); j++)
      tmp[i][j] = std::exp(param[i][j]) * g[i][j];

  return tmp;
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

}  // namespace base
}  // namespace wavepi
