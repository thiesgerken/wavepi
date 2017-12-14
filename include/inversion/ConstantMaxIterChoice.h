/*
 * ConstantMaxIterChoice.h
 *
 *  Created on: 14.12.2017
 *      Author: thies
 */

#ifndef LIB_INVERSION_CONSTANTMAXITERCHOICE_H_
#define LIB_INVERSION_CONSTANTMAXITERCHOICE_H_

#include <deal.II/base/parameter_handler.h>
#include <inversion/MaxIterChoice.h>

namespace wavepi {
namespace inversion {
using namespace dealii;

class ConstantMaxIterChoice: public MaxIterChoice {
   public:
      ConstantMaxIterChoice(int max_iter);
      ConstantMaxIterChoice(ParameterHandler &prm);

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

   protected:
      virtual int calculate_max_iter() const;

   private:
      int max_iter;
};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* LIB_INVERSION_CONSTANTMAXITERCHOICE_H_ */
