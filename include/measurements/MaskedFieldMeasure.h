/*
 * FieldMeasure.h
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_MASKEDFIELDMEASURE_H_
#define INCLUDE_MEASUREMENTS_MASKEDFIELDMEASURE_H_

#include <base/DiscretizedFunction.h>
#include <base/Norm.h>
#include <base/SpaceTimeMesh.h>
#include <deal.II/base/exceptions.h>
#include <measurements/Measure.h>
#include <memory>

namespace wavepi {
namespace measurements {
using namespace dealii;
using namespace wavepi::base;

/**
 * No real measurement, just returns the argument.
 * This way one can still reconstruct from the whole field even when the rest of the code expects measurement operators.
 */
template <int dim>
class MaskedFieldMeasure : public Measure<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
 public:
  virtual ~MaskedFieldMeasure() = default;

  MaskedFieldMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh, std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm, std::shared_ptr<DiscretizedFunction<dim>> mask)
      : mesh(mesh), norm(norm) , mask(mask) {
    AssertThrow(mesh && norm, ExcNotInitialized());
  }

  virtual DiscretizedFunction<dim> evaluate(const DiscretizedFunction<dim>& field) override {
    AssertThrow(mesh == field.get_mesh(), ExcMessage("MaskedFieldMeasure called with different meshes"));
    AssertThrow(*norm == *field.get_norm(), ExcMessage("MaskedFieldMeasure called with different norms"));

    auto tmp = field;
    tmp.pointwise_multiplication(*mask);

    return tmp;
  }

  virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& measurements) override {
    AssertThrow(mesh == measurements.get_mesh(), ExcMessage("MaskedFieldMeasure called with different meshes"));
    AssertThrow(*norm == *measurements.get_norm(), ExcMessage("MaskedFieldMeasure called with different norms"));

    auto tmp = measurements;
    tmp.pointwise_multiplication(*mask);
    return tmp;
  }

  virtual DiscretizedFunction<dim> zero() override { return DiscretizedFunction<dim>(mesh, norm); }

 private:
  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  std::shared_ptr<Norm<DiscretizedFunction<dim>>> norm;
  std::shared_ptr<DiscretizedFunction<dim>> mask;
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_FIELDMEASURE_H_ */
