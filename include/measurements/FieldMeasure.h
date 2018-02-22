/*
 * FieldMeasure.h
 *
 *  Created on: 22.02.2018
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_FIELDMEASURE_H_
#define INCLUDE_MEASUREMENTS_FIELDMEASURE_H_

#include <base/DiscretizedFunction.h>
#include <base/SpaceTimeMesh.h>
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
class FieldMeasure : public Measure<DiscretizedFunction<dim>, DiscretizedFunction<dim>> {
 public:
  virtual ~FieldMeasure() = default;

  FieldMeasure(std::shared_ptr<SpaceTimeMesh<dim>> mesh, Norm norm) : mesh(mesh), norm(norm) {
    AssertThrow(mesh, ExcNotInitialized());
  }

  virtual DiscretizedFunction<dim> evaluate(const DiscretizedFunction<dim>& field) override {
    AssertThrow(mesh == field.get_mesh(), ExcMessage("FieldMeasure called with different meshes"));
    AssertThrow(norm == field.get_norm(), ExcMessage("FieldMeasure called with different norms"));

    return field;
  }

  virtual DiscretizedFunction<dim> adjoint(const DiscretizedFunction<dim>& measurements) override {
    AssertThrow(mesh == measurements.get_mesh(), ExcMessage("Field measure called with different meshes"));
    AssertThrow(norm == measurements.get_norm(), ExcMessage("FieldMeasure called with different norms"));

    return measurements;
  }

  virtual DiscretizedFunction<dim> zero() override { return DiscretizedFunction<dim>(mesh, false, norm); }

 private:
  std::shared_ptr<SpaceTimeMesh<dim>> mesh;
  Norm norm;
};

}  // namespace measurements
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_FIELDMEASURE_H_ */
