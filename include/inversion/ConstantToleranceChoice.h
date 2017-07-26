/*
 * ConstantToleranceChoice.h
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_
#define INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_

#include <inversion/ToleranceChoice.h>

namespace wavepi {
namespace inversion {

class ConstantToleranceChoice: public ToleranceChoice {
   public:
      ConstantToleranceChoice(double tol);

      double get_tol() const;
      void set_tol(double tol);

   protected:
      virtual double calculate_tolerance() const;

   private:
      double tol;

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_ */
