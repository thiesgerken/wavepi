/*
 * WinklerToleranceChoice.h
 *
 *  Created on: 14.06.2019
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_WINKLERTOLERANCECHOICE_H_
#define INCLUDE_INVERSION_WINKLERTOLERANCECHOICE_H_

#include <deal.II/base/parameter_handler.h>

#include <inversion/ToleranceChoice.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

class WinklerToleranceChoice : public ToleranceChoice {
 public:
  WinklerToleranceChoice(double tol_max, double zeta, double beta, bool safeguarding);
  WinklerToleranceChoice(ParameterHandler &prm);

  static void declare_parameters(ParameterHandler &prm);
  void get_parameters(ParameterHandler &prm);

 protected:
  using ToleranceChoice::discrepancies;
  using ToleranceChoice::previous_tolerances;
  using ToleranceChoice::required_steps;
  using ToleranceChoice::target_discrepancy;

  virtual double calculate_tolerance() const;

 private:
  double tol_max;
  double zeta;
  double beta;
  bool safeguarding;
};

}  // namespace inversion
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_WinklerTOLERANCECHOICE_H_ */
