/*
 * MacroFunctionParser.cpp
 *
 *  Created on: 18.08.2017
 *      Author: thies
 */

#include <base/MacroFunctionParser.h>
#include <iostream>
#include <memory>
#include <regex>

namespace wavepi {
namespace base {

template <int dim>
MacroFunctionParser<dim>::MacroFunctionParser(const std::vector<std::string>& expressions,
                                              const std::map<std::string, double>& constants) {
  std::vector<std::string> exprs;

  for (auto expr : expressions)
    exprs.push_back(replace(expr));

  this->initialize(FunctionParser<dim>::default_variable_names() + ",t", exprs, constants, true);
}

template <int dim>
MacroFunctionParser<dim>::MacroFunctionParser(const std::string& expression,
                                              const std::map<std::string, double>& constants) {
  auto expr = replace(expression);

  this->initialize(FunctionParser<dim>::default_variable_names() + ",t", expr, constants, true);
}

template <int dim>
std::string MacroFunctionParser<dim>::replace(const std::string& expr) {
  std::regex norm_pattern("norm\\{([^\\{\\}\\|]+)\\|([^\\{\\}\\|]+)\\|([^\\{\\}\\|]+)\\}");
  return std::regex_replace(expr, norm_pattern, norm_replacement);
}

template <>
const std::string MacroFunctionParser<1>::norm_replacement = "sqrt(pow($1,2))";

template <>
const std::string MacroFunctionParser<2>::norm_replacement = "sqrt(pow($1,2)+pow($2,2))";

template <>
const std::string MacroFunctionParser<3>::norm_replacement = "sqrt(pow($1,2)+pow($2,2)+pow($3,2))";

template <int dim>
std::shared_ptr<LightFunction<dim>> MacroFunctionParser<dim>::parse(const std::string& expression,
                                                                    const std::map<std::string, double>& constants,
                                                                    double shape_scale) {
  if (expression == "MovingRingSegment") {
    return std::make_shared<RingShapeFunction<dim>>(shape_scale);
  } else if (expression == "PulsingLDot") {
    return std::make_shared<LShapeDotFunction<dim>>(shape_scale);
  } else
    return std::make_shared<MacroFunctionParser<dim>>(expression, constants);
}

template <>
double RingShapeFunction<1>::evaluate(const Point<1>& p __attribute__((unused)),
                                      const double time __attribute__((unused))) const {
  throw ExcNotImplemented("RingShapeFunction<1>");
}

template <>
double RingShapeFunction<2>::evaluate(const Point<2>& p, const double time) const {
  double r = p.norm();
  if (r < radius1 || r > radius2) return 0.0;

  double phi = atan2(p[1], p[0]);
  if (phi < 0) phi += 2 * M_PI;

  double tm  = fmod(time, 2.0 * M_PI);
  double tm2 = fmod(time + M_PI / 2.0, 2.0 * M_PI);

  if (phi >= tm && phi <= tm + M_PI / 2.0)
    return shape_scale;
  else if (tm2 < tm && phi <= tm2)
    return shape_scale;
  else
    return 0.0;
}

template <>
double RingShapeFunction<3>::evaluate(const Point<3>& p, const double time) const {
  double r = p.norm();
  if (r < radius1 || r > radius2) return 0.0;

  double phi = atan2(p[1], p[0]);
  if (phi < 0) phi += 2 * M_PI;

  double tm  = fmod(time, 2.0 * M_PI);
  double tm2 = fmod(time + M_PI / 2.0, 2.0 * M_PI);

  if (phi >= tm && phi <= tm + M_PI / 2.0)
    return shape_scale;
  else if (tm2 < tm && phi <= tm2)
    return shape_scale;
  else
    return 0.0;
}

template <>
double LShapeDotFunction<1>::evaluate(const Point<1>& p __attribute__((unused)),
                                      const double time __attribute__((unused))) const {
  throw ExcNotImplemented("LShapeDotFunction<1>");
}

template <>
double LShapeDotFunction<2>::evaluate(const Point<2>& p, const double time) const {
  double value = 0.0;

  double dist_dot_sq = (p[0] - (1 - boundary_dist - dot_radius)) * (p[0] - (1 - boundary_dist - dot_radius)) +
                       (p[1] - (1 - boundary_dist - dot_radius)) * (p[1] - (1 - boundary_dist - dot_radius));

  if (dist_dot_sq <= dot_radius * dot_radius) value -= 1.0 - dist_dot_sq / (dot_radius * dot_radius);

  if (p[0] >= -1.0 + boundary_dist && p[0] <= 1.0 - boundary_dist && p[1] >= -1.0 + boundary_dist &&
      p[1] <= -1.0 + boundary_dist + l_width)
    value += 1.0;
  else if (p[1] >= -1.0 + boundary_dist && p[1] <= 1.0 - boundary_dist && p[0] >= -1.0 + boundary_dist &&
           p[0] <= -1.0 + boundary_dist + l_width)
    value += 1.0;

  if (value == 0.0) return 0.0;
  value *= 0.2 + 0.8 * sin(time) * sin(time);
  return shape_scale * value;
}

template <>
double LShapeDotFunction<3>::evaluate(const Point<3>& p, const double time) const {
  double value = 0.0;

  double dist_dot_sq = (p[0] - (1 - boundary_dist - dot_radius)) * (p[0] - (1 - boundary_dist - dot_radius)) +
                       (p[1] - (1 - boundary_dist - dot_radius)) * (p[1] - (1 - boundary_dist - dot_radius)) +
                       (p[2] - (1 - boundary_dist - dot_radius)) * (p[2] - (1 - boundary_dist - dot_radius));

  if (dist_dot_sq <= dot_radius * dot_radius) value -= 1.0 - dist_dot_sq / (dot_radius * dot_radius);

  if (p[0] >= -1.0 + boundary_dist && p[0] <= 1.0 - boundary_dist && p[1] >= -1.0 + boundary_dist &&
      p[1] <= 1.0 - boundary_dist && p[2] >= -1.0 + boundary_dist && p[2] <= -1.0 + boundary_dist + l_width)
    value += 1.0;
  else if (p[1] >= -1.0 + boundary_dist && p[1] <= 1.0 - boundary_dist && p[2] >= -1.0 + boundary_dist &&
           p[2] <= 1.0 - boundary_dist && p[0] >= -1.0 + boundary_dist && p[0] <= -1.0 + boundary_dist + l_width)
    value += 1.0;
  else if (p[2] >= -1.0 + boundary_dist && p[2] <= 1.0 - boundary_dist && p[0] >= -1.0 + boundary_dist &&
           p[0] <= 1.0 - boundary_dist && p[1] >= -1.0 + boundary_dist && p[1] <= -1.0 + boundary_dist + l_width)
    value += 1.0;

  if (value == 0.0) return 0.0;
  value *= 0.2 + 0.8 * sin(time) * sin(time);

  return shape_scale * value;
}

template class RingShapeFunction<1>;
template class RingShapeFunction<2>;
template class RingShapeFunction<3>;

template class LShapeDotFunction<1>;
template class LShapeDotFunction<2>;
template class LShapeDotFunction<3>;

template class MacroFunctionParser<1>;
template class MacroFunctionParser<2>;
template class MacroFunctionParser<3>;

}  // namespace base
} /* namespace wavepi */
