#ifndef PARAMETERS_ELECPOT_H
#define PARAMETERS_ELECPOT_H

#include <deal.II/base/parameter_handler.h>

namespace Parameters
  {
    // @sect4{Geometry}

    // Make adjustments to the problem geometry and the applied load. 
    struct Geometry
    {
      unsigned int global_refinement;
      double       scale;
      std::string  mesh_file;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

    // @sect4{Time}

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct ElectroChem
    {
      double rho;
      double kappa;
      double current;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

  
// @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single container
    // that holds all of our run-time selections.
    struct AllParameters : 
      public Geometry,
      public ElectroChem
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };
}
#endif
