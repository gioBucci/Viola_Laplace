#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <deal.II/base/parameter_handler.h>

namespace Parameters
  {
    // @sect4{Finite Element system}

    // Here we specify the polynomial order used to approximate the solution.
    // The quadrature order should be adjusted accordingly.
    struct FESystem
    {
      unsigned int poly_degree;
      unsigned int quad_order;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };


   // @sect4{Geometry}

    // Make adjustments to the problem geometry and the applied load. 
    struct Geometry
    {
      unsigned int global_refinement;
      double       scale;
      int          grid_flag;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };


  // @sect4{Material_electrode}

    // We also need the Young's modulus $ E_0 $ and Poisson ration $ \nu $ for the
    // neo-Hookean material. We also need the rate of change of E with Li concentration
    // and the volumetric expansion rate dbetadc, plus other material constants.
    struct Material_electrode
    {
      double nu;
      double E0;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

 // @sect4{Material_electrolyte}

    // We also need the Young's modulus $ E_0 $ and Poisson ration $ \nu $ for the
    // neo-Hookean material. We also need the rate of change of E with Li concentration
    // and the volumetric expansion rate dbetadc, plus other material constants.
    struct Material_electrolyte
    {
      double nu_el;
      double E0_el;
     
      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

// @sect4{Linear solver}

    // Next, we choose both the linear solver settings within a Newton increment in a nonlinear problem.
    struct LinearSolver
    {
      std::string type_lin;
      double      tol_lin;
      double      max_iterations_lin;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };


 // @sect4{Nonlinear solver}

    // A Newton-Raphson scheme is used to solve the nonlinear system of governing
    // equations.  We now define the tolerances and the maximum number of
    // iterations for the Newton-Raphson nonlinear solver.
    struct NonlinearSolver
    {
      unsigned int max_iterations_NR;
      double       tol_f;
      double       tol_u;
    
      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };


    // @sect4{Time}

    // Set the timestep size $ \varDelta t $ and the simulation end-time.
    struct Time
    {
      double delta_t;
      double end_time;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

  
// @sect4{All parameters}

    // Finally we consolidate all of the above structures into a single container
    // that holds all of our run-time selections.
    struct AllParameters : 
      public FESystem,
      public Geometry,
      public Material_electrode,
      public Material_electrolyte,
      public LinearSolver,
      public NonlinearSolver,
      public Time
    {
      AllParameters(const std::string &input_file);

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };
}
#endif
