/*
 * ConstantToleranceChoice.h
 *
 *  Created on: 26.07.2017
 *      Author: thies
 */

#ifndef INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_
#define INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_

#include <deal.II/base/parameter_handler.h>
#include <inversion/ToleranceChoice.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

class ConstantToleranceChoice: public ToleranceChoice {
   public:
      ConstantToleranceChoice(double tol);
      ConstantToleranceChoice(ParameterHandler &prm);

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

   protected:
      virtual double calculate_tolerance() const;

   private:
      double tol;

};

} /* namespace problems */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSION_CONSTANTTOLERANCECHOICE_H_ */
