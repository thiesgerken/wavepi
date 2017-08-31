/*
 * PointMeasure.h
 *
 *  Created on: 31.08.2017
 *      Author: thies
 */

#ifndef INCLUDE_MEASUREMENTS_POINTMEASURE_H_
#define INCLUDE_MEASUREMENTS_POINTMEASURE_H_

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/function.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>

#include <forward/DiscretizedFunction.h>
#include <forward/SpaceTimeMesh.h>

#include <measurements/Measure.h>
#include <measurements/MeasuredValues.h>

#include <stddef.h>

#include <util/SpaceTimeGrid.h>

#include <list>
#include <memory>
#include <utility>

namespace wavepi {
namespace measurements {

using namespace dealii;
using namespace wavepi::forward;
using namespace wavepi::util;

/**
 * Point measurements, implemented as scalar product between the given field and a delta-approximating function.
 */
template<int dim>
class PointMeasure: public Measure<DiscretizedFunction<dim>, MeasuredValues<dim>> {
   public:

      virtual ~PointMeasure() = default;

      /**
       * @param points Points in space and time (last dimension is time) where you want those point measurements.
       * @param delta_shape Shape of the delta-approximating function. Should be supported in [-1,1]^{dim+1}.
       * @param delta_scale_space Desired support radius in space
       * @param delta_scale_time Desired support radius in time
       */
      PointMeasure(std::shared_ptr<SpaceTimeGrid<dim>> points, std::shared_ptr<Function<dim + 1>> delta_shape,
            double delta_scale_space, double delta_scale_time);

      /**
       * Does not initialize most of the values, you have to use get_parameters afterwards and use `set_measurement_points`.       *
       */
      PointMeasure();

      static void declare_parameters(ParameterHandler &prm);
      void get_parameters(ParameterHandler &prm);

      virtual MeasuredValues<dim> evaluate(const DiscretizedFunction<dim>& field);

      /**
       * Adjoint, discretized on the mesh last used for evaluate
       */
      virtual DiscretizedFunction<dim> adjoint(const MeasuredValues<dim>& measurements);

      const std::shared_ptr<SpaceTimeGrid<dim>>& get_measurement_points() const {
         return measurement_points;
      }

      void set_measurement_points(const std::shared_ptr<SpaceTimeGrid<dim>>& measurement_points) {
         this->measurement_points = measurement_points;
      }

   protected:
      std::shared_ptr<SpaceTimeMesh<dim>> mesh;
      std::shared_ptr<SpaceTimeGrid<dim>> measurement_points;

      std::shared_ptr<Function<dim + 1>> delta_shape;
      double delta_scale_space;
      double delta_scale_time;

      struct AssemblyScratchData {
            AssemblyScratchData(const FiniteElement<dim> &fe, const Quadrature<dim> &quad);
            AssemblyScratchData(const AssemblyScratchData &scratch_data);
            FEValues<dim> fe_values;
      };

      struct AssemblyCopyData {
            std::vector<double> cell_values;
      };

      void copy_local_to_global(const std::vector<std::pair<size_t, double>> &jobs, MeasuredValues<dim> &dest,
            const AssemblyCopyData &copy_data) const;

      void local_add_contributions(const std::vector<std::pair<size_t, double>> &jobs, const Vector<double> &u,
            double time, const typename DoFHandler<dim>::active_cell_iterator &cell,
            AssemblyScratchData &scratch_data, AssemblyCopyData &copy_data) const;

};

} /* namespace forward */
} /* namespace wavepi */

#endif /* INCLUDE_MEASUREMENTS_POINTMEASURE_H_ */
