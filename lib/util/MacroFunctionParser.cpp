/*
 * MacroFunctionParser.cpp
 *
 *  Created on: 18.08.2017
 *      Author: thies
 */

#include <util/MacroFunctionParser.h>
#include <iostream>
#include <regex>

namespace wavepi {
namespace util {

template<int dim>
MacroFunctionParser<dim>::MacroFunctionParser(const std::vector<std::string> & expressions,
      const std::map<std::string, double> & constants) {
   std::vector<std::string> exprs;

   for (auto expr : expressions)
      exprs.push_back(replace(expr));

   this->initialize(FunctionParser<dim>::default_variable_names() + ",t", exprs, constants, true);
}

template<int dim>
MacroFunctionParser<dim>::MacroFunctionParser(const std::string & expression,
      const std::map<std::string, double> & constants) {
   auto expr = replace(expression);

   this->initialize(FunctionParser<dim>::default_variable_names() + ",t", expr, constants, true);
}

template<int dim>
std::string MacroFunctionParser<dim>::replace(const std::string & expr) {
   std::regex norm_pattern("norm\\{([^\\{\\}\\|]+)\\|([^\\{\\}\\|]+)\\|([^\\{\\}\\|]+)\\}");
   return std::regex_replace(expr, norm_pattern, norm_replacement);
}

template<>
const std::string MacroFunctionParser<1>::norm_replacement = "sqrt(pow($1,2))";

template<>
const std::string MacroFunctionParser<2>::norm_replacement = "sqrt(pow($1,2)+pow($2,2))";

template<>
const std::string MacroFunctionParser<3>::norm_replacement = "sqrt(pow($1,2)+pow($2,2)+pow($3,2))";

template class MacroFunctionParser<1> ;
template class MacroFunctionParser<2> ;
template class MacroFunctionParser<3> ;

} /* namespace util */
} /* namespace wavepi */