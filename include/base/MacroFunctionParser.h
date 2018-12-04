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

 private:
  static const std::string norm_replacement;

  static std::string replace(const std::string& expr);
};

}  // namespace base
} /* namespace wavepi */

#endif /* LIB_UTIL_MACROFUNCTIONPARSER_H_ */
