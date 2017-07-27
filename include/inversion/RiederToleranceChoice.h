/*
 * RiederToleranceChoice.h
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_
#define INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_

#include <inversion/ToleranceChoice.h>

namespace wavepi {
namespace inversion {

class RiederToleranceChoice: public ToleranceChoice {
   public:
      RiederToleranceChoice(double tol_start, double tol_max, double zeta, double beta);
   protected:
      using ToleranceChoice::previous_tolerances;
      using ToleranceChoice::discrepancies;
      using ToleranceChoice::required_steps;
      using ToleranceChoice::target_discrepancy;

      virtual double calculate_tolerance() const;

   private:
      double tol_start;
      double tol_max;
      double zeta;
      double beta;
};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_RIEDERTOLERANCECHOICE_H_ */
