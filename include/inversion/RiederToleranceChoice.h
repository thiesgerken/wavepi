/*
 * RiederToleranceChoice.h
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_
#define INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_

#include <deal.II/base/parameter_handler.h>

#include <inversion/ToleranceChoice.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

class RiederToleranceChoice : public ToleranceChoice {
 public:
  RiederToleranceChoice(double tol_start, double tol_max, double zeta, double beta, bool safeguarding);
  RiederToleranceChoice(ParameterHandler &prm);

  static void declare_parameters(ParameterHandler &prm);
  void get_parameters(ParameterHandler &prm);

 protected:
  using ToleranceChoice::discrepancies;
  using ToleranceChoice::previous_tolerances;
  using ToleranceChoice::required_steps;
  using ToleranceChoice::target_discrepancy;

  virtual double calculate_tolerance() const;

 private:
  double tol_start;
  double tol_max;
  double zeta;
  double beta;
  bool safeguarding;
};

}  // namespace inversion
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_ */
