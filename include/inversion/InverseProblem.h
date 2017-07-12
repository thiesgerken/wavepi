/*
 * InverseProblem.h
 *
 *  Created on: 03.07.2017
 *      Author: thies
 */

#ifndef INVERSION_INVERSEPROBLEM_H_
#define INVERSION_INVERSEPROBLEM_H_

#include <memory>

namespace wavepi {
namespace inversion {

template<typename Param, typename Sol>
class InverseProblem {
   public:
      typedef Sol& (*ParamToSol)(const Param&);
      typedef Param& (*SolToParam)(const Sol&);

      virtual ~InverseProblem() {
      }

      virtual Sol forward(Param& f) = 0;

      // progress indicator that iterative methods can call
      // exact_param might be equal to null_ptr
      // default implementation does nothing
      // virtual void progress(const Param& current_estimate, const Sol& current_residual, const Sol& data,
      //  int iteration_number, const Param* exact_param);
      virtual void progress(const Param& current_estimate __attribute__((unused)),
            const Sol& current_residual __attribute__((unused)), const Sol& data __attribute__((unused)),
            int iteration_number __attribute__((unused)),std::shared_ptr<const Param> exact_param __attribute__((unused))) {
      }

};

} /* namespace inversion */
} /* namespace wavepi */

#endif /* INCLUDE_INVERSEPROBLEM_H_ */
