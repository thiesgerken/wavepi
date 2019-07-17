/*
 * MacroFunctionParser.h
 *
 *  Created on: 18.08.2017
 *      Author: thies
 */

#ifndef LIB_UTIL_MACROFUNCTIONPARSER_H_
#define LIB_UTIL_MACROFUNCTIONPARSER_H_

#include <base/FunctionParser.h>
#include <base/LightFunction.h>

#include <math.h>
#include <map>
#include <string>
#include <vector>

namespace wavepi {
namespace base {
using namespace dealii;

/**
 * `FunctionParser` with additional commands.
 * In contrast to its base class, you do not need to call `initialize` (not virtual ...).
 *
 * Additional replacements:
 * `norm{x|y|z}` becomes `sqrt(pow(x,2))`, `sqrt(pow(x,2)+pow(y,2))` or `sqrt(pow(x,2)+pow(y,2)+pow(z,2))`, depending on
 * dimension. Currently this is implemented using naive string replacement, so the characters '}', '{' and '|' must not
 * be used in expressions x,y or z.
 */
template <int dim>
class MacroFunctionParser : public FunctionParser<dim> {
 public:
  virtual ~MacroFunctionParser() = default;

  MacroFunctionParser(const std::string& expression, const std::map<std::string, double>& constants);
  MacroFunctionParser(const std::vector<std::string>& expressions, const std::map<std::string, double>& constants);

  /**
   * if expression contains a known function name then return a instance of that, otherwise use expression and constants
   * to create a MacroFunctionParser.
   */
  static std::shared_ptr<LightFunction<dim>> parse(const std::string& expression,
                                                   const std::map<std::string, double>& constants, double shape_scale);

 private:
  static const std::string norm_replacement;

  static std::string replace(const std::string& expr);
};

template <int dim>
class RingShapeFunction : public LightFunction<dim> {
 public:
  virtual ~RingShapeFunction() = default;
  RingShapeFunction(double shape_scale) : shape_scale(shape_scale) {}

  virtual double evaluate(const Point<dim>& p, const double time) const;

 private:
  double shape_scale;

  const double radius1 = 0.3;
  const double radius2 = 0.8;
};

template <int dim>
class LShapeDotFunction : public LightFunction<dim> {
 public:
  virtual ~LShapeDotFunction() = default;
  LShapeDotFunction(double shape_scale) : shape_scale(shape_scale) {}

  virtual double evaluate(const Point<dim>& p, const double time) const;

 private:
  double shape_scale;

  const double dot_radius    = 0.45;
  const double l_width       = 0.3;
  const double boundary_dist = 0.2;
};

template <int dim>
class LShapeDotConstantFunction : public LightFunction<dim> {
 public:
  virtual ~LShapeDotConstantFunction() = default;
  LShapeDotConstantFunction(double shape_scale) : shape_scale(shape_scale) {}

  virtual double evaluate(const Point<dim>& p, const double time) const;

 private:
  double shape_scale;

  const double dot_radius    = 0.45;
  const double l_width       = 0.3;
  const double boundary_dist = 0.2;
};

}  // namespace base
} /* namespace wavepi */

#endif /* LIB_UTIL_MACROFUNCTIONPARSER_H_ */
