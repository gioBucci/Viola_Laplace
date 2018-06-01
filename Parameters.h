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
      double dEdc;
      double dbetadc;
      double rho_electrode;
      double D;
      double c0;
      double cmax;

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
      double dEdc_el;
      double dbetadc_el;
      double rho_electrolyte;
      double D_el;
      double c0_el;
      double cmax_el;

      static void
      declare_parameters(dealii::ParameterHandler &prm);

      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

  // @sect4{ElectroChem}
    struct ElectroChem
    {
      double RT;
      double Farad; 
      double mu0;
      double Io; 
      double alpha;
      double current;

      static void
      declare_parameters(dealii::ParameterHandler &prm);
      
      void
      parse_parameters(dealii::ParameterHandler &prm);
    };

// @sect4{Fracture}

    // Fracture parameters: maximum opening displacement and maximum traction
    // Material intrinsic properties are the fracture energy and the maximum opening displacement,
    // which it needs to be scaled acoording to the mesh sixe.
    // The max traction is computed from the fracture energy and the scaled opening displacement
    struct Fracture
    {
      double coh_ELstiff;
      double Elast_open_disp; 
      double elasticToCrit_disp;
      double fracture_En; 

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
      double      Bbar;
      // std::string preconditioner_type;
      // double      preconditioner_relaxation;

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
      double       tol_fc;
      double       tol_c;
      double       tol_fmu;
      double       tol_mu;

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
      public ElectroChem,
      public Fracture,
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
