  // @sect3{Run-time parameters}
  //
  // There are several parameters that can be set in the code so we set up a
  // ParameterHandler object to read in the choices at run-time.  

#include "Parameters.h"

void Parameters::FESystem::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Finite element system");
  {
    prm.declare_entry("Polynomial degree", "1",
		      dealii::Patterns::Integer(0),
		      "Displacement system polynomial order");

    prm.declare_entry("Quadrature order", "2",
		      dealii::Patterns::Integer(0),
		      "Gauss quadrature order");
  }
  prm.leave_subsection();
}

void Parameters::FESystem::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Finite element system");
  {
    poly_degree = prm.get_integer("Polynomial degree");
    quad_order = prm.get_integer("Quadrature order");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::Geometry::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Geometry");
  {
    prm.declare_entry("Global refinement", "1",
		      dealii::Patterns::Integer(0),
		      "Global refinement level");

    prm.declare_entry("Grid scale", "1.0e-5",
		      dealii::Patterns::Double(0.0),
		      "Global grid scaling factor");

    prm.declare_entry("Grid generation choice", "1",
		      dealii::Patterns::Selection("0|1|2|3|4"),
		      "Flag for generating or reading the mesh");
  }
  prm.leave_subsection();
}

void Parameters::Geometry::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Geometry");
  {
    global_refinement = prm.get_integer("Global refinement");
    scale = prm.get_double("Grid scale");
    grid_flag = prm.get_integer("Grid generation choice");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::Material_electrode::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrode material properties");
  {
    prm.declare_entry("Poisson's ratio", "0.22",
		      dealii::Patterns::Double(-1.0,0.5),
		      "Poisson's ratio");
    prm.declare_entry("Young's modulus", "5.0e9",  // 50.0e9
		      dealii::Patterns::Double(),
		      "Young's modulus, initial value");
    prm.declare_entry("Young's modulus rate", "0.0",
		      dealii::Patterns::Double(),
		      "Young's modulus rate");
    prm.declare_entry("Volumetric expansion rate", "-0.01",
		      dealii::Patterns::Double(-1.5,1.5),
		      "Volumetric expansion rate");
    prm.declare_entry("Electrode molar density", "7.874e4",  // 7.874e04  2.6e-14
		      dealii::Patterns::Double(0.0),
		      "Electrode molar density");
    prm.declare_entry("Diffusivity", "1.0e-13",  // 1.0e-12
		      dealii::Patterns::Double(0.0),
		      "Diffusivity");
    prm.declare_entry("Initial concentration", "0.05",
		      dealii::Patterns::Double(0.0, 5.0),
		      "Initial concentration");
    prm.declare_entry("Maximum concentration", "3.75",
		      dealii::Patterns::Double(0.0,5.0),
		      "Electrode Maximum concentration");
  }
  prm.leave_subsection();
}

void Parameters::Material_electrode::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrode material properties");
  {
    nu = prm.get_double("Poisson's ratio");
    E0 = prm.get_double("Young's modulus");
    dEdc = prm.get_double("Young's modulus rate");
    dbetadc = prm.get_double("Volumetric expansion rate");
    rho_electrode = prm.get_double("Electrode molar density");
    D = prm.get_double("Diffusivity");
    c0 = prm.get_double("Initial concentration");
    cmax = prm.get_double("Maximum concentration");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::Material_electrolyte::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrolyte material properties");
  {
    prm.declare_entry("Poisson's ratio", "0.22",
		      dealii::Patterns::Double(-1.0,0.5),
		      "Electrolyte Poisson's ratio");
    prm.declare_entry("Young's modulus", "100.0e9",   // 20.0e9
		      dealii::Patterns::Double(),
		      "Electrolyte Young's modulus, initial value");
    prm.declare_entry("Young's modulus rate", "0.0",
		      dealii::Patterns::Double(),
		      "Electrolyte Young's modulus rate");
    prm.declare_entry("Volumetric expansion rate", "0.0",
		      dealii::Patterns::Double(-1.0,1.0),
		      "Electrolyte Volumetric expansion rate");
    prm.declare_entry("Electrode molar density", "7.874e4",   //7.874e04
		      dealii::Patterns::Double(),
		      "Electrolyte molar density");
    prm.declare_entry("Diffusivity", "1.0e-11",   // 1.0e-12
		      dealii::Patterns::Double(0.0),
		      "Electrolyte Diffusivity");
    prm.declare_entry("Initial concentration", "0.05",
		      dealii::Patterns::Double(0.0, 5.0),
		      "Electrolyte Initial concentration");
    prm.declare_entry("Maximum concentration", "0.5",
		      dealii::Patterns::Double(0.0,5.0),
		      "Electrolyte Maximum concentration");
  }
  prm.leave_subsection();
}

void Parameters::Material_electrolyte::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrolyte material properties");
  {
    nu_el = prm.get_double("Poisson's ratio");
    E0_el = prm.get_double("Young's modulus");
    dEdc_el = prm.get_double("Young's modulus rate");
    dbetadc_el = prm.get_double("Volumetric expansion rate");
    rho_electrolyte = prm.get_double("Electrode molar density");
    D_el = prm.get_double("Diffusivity");
    c0_el = prm.get_double("Initial concentration");
    cmax_el = prm.get_double("Maximum concentration");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::ElectroChem::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrochemical properties");
  {
    prm.declare_entry("Gas constant x Temperature", "2477.572",    // 2477.572
		      dealii::Patterns::Double(0.0),
		      "Gas constant x Temperature");
    prm.declare_entry("Farady constant", "96485.3365",
		      dealii::Patterns::Double(0.0),
		      "Farady constant");
    prm.declare_entry("Reference chemical potential", "2.05241",   // 2.05241e12
		      dealii::Patterns::Double(0.0),
		      "Reference chemical potential");
    prm.declare_entry("Exchange current density", "0.01",
		      dealii::Patterns::Double(0.0),
		      "Exchange current density");
    prm.declare_entry("Reaction symmetry constant", "0.5",
		      dealii::Patterns::Double(0.0, 1.0),
		      "Reaction symmetry constant");
    prm.declare_entry("Applied current density", "10.0",
		      dealii::Patterns::Double(0.0),
		      "Applied current density");
  }
  prm.leave_subsection();
}

void Parameters::ElectroChem::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrochemical properties");
  {
    RT = prm.get_double("Gas constant x Temperature");
    Farad = prm.get_double("Farady constant");
    mu0 = prm.get_double("Reference chemical potential");
    Io = prm.get_double("Exchange current density");
    alpha = prm.get_double("Reaction symmetry constant");
    current =  prm.get_double("Applied current density");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::Fracture::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Fracture");
  {
    prm.declare_entry("Initial elastic cohesive stiffness", "1.e15",   // 10.e14 for scale 1.e6
		      dealii::Patterns::Double(0.0),
		      "Initial cohesive stiffness in the elastic range");
    
    prm.declare_entry("Opening displacement elastic limit", "7.0711e-9",
		      dealii::Patterns::Double(0.0),
		      "Opening displacement at the elastic limit of the traction-separation law");

    prm.declare_entry("Factor critical opening displacement", "20",
		      dealii::Patterns::Double(0.0),
		      "Factor critical opening displacement");

    prm.declare_entry("Fracture energy", "1.0",
		      dealii::Patterns::Double(0.0),
		      "Cohesive fracture energy (J/m^2)");   // the same for kg/s^2

    // prm.declare_entry("Ratio tangential/normal energy", "1.0",
    // 		      dealii::Patterns::Double(0.0),
    // 		      "Ratio between the works of tangential and normal separation (J/m^2)");

  }
  prm.leave_subsection();
}

void Parameters::Fracture::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Fracture");
  {
    coh_ELstiff = prm.get_double("Initial elastic cohesive stiffness");
    Elast_open_disp = prm.get_double("Opening displacement elastic limit");
    elasticToCrit_disp = prm.get_double("Factor critical opening displacement");
    fracture_En = prm.get_double("Fracture energy");
 
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::LinearSolver::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Linear solver");
  {
    prm.declare_entry("Solver type", "UMFPACK",
		      dealii::Patterns::Selection("UMFPACK|PETSc_MUMPS"),
		      "Type of solver used to solve the linear system");

    prm.declare_entry("Residual", "1e-6",
		      dealii::Patterns::Double(0.0),
		      "Linear solver residual (scaled by residual norm)");

    prm.declare_entry("Max iteration multiplier", "1",
		      dealii::Patterns::Double(0.0),
		      "Linear solver iterations (multiples of the system matrix size)");

    prm.declare_entry("B bar method factor", "0.02",
    		      dealii::Patterns::Double(0.0, 1.0),
                      "Factor in B bar method for concentration DOF");

    // prm.declare_entry("Preconditioner type", "ssor",
    //                   Patterns::Selection("jacobi|ssor"),
    //                   "Type of preconditioner");

    // prm.declare_entry("Preconditioner relaxation", "0.65",
    //                   Patterns::Double(0.0),
    //                   "Preconditioner relaxation value");
  }
  prm.leave_subsection();
}

void Parameters::LinearSolver::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Linear solver");
  {
    type_lin = prm.get("Solver type");
    tol_lin = prm.get_double("Residual");
    max_iterations_lin = prm.get_double("Max iteration multiplier");
    Bbar = prm.get_double("B bar method factor");
    // preconditioner_type = prm.get("Preconditioner type");
    // preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::NonlinearSolver::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver");
  {
    prm.declare_entry("Max iterations Newton-Raphson", "15",
		      dealii::Patterns::Integer(0),
		      "Number of Newton-Raphson iterations allowed");

    prm.declare_entry("Tolerance force", "1.0e-6",
		      dealii::Patterns::Double(0.0),
		      "Force residual tolerance");

    prm.declare_entry("Tolerance displacement", "1.0e-6",
		      dealii::Patterns::Double(0.0),
		      "Displacement error tolerance");

    prm.declare_entry("Tolerance concentration residual", "1.0e-8",
		      dealii::Patterns::Double(0.0),
		      "Concentration residual tolerance");

    prm.declare_entry("Tolerance concentration", "1.0e-6",
		      dealii::Patterns::Double(0.0),
		      "Concentrationt error tolerance");

    prm.declare_entry("Tolerance chemical potential residual", "5.0e-5",
		      dealii::Patterns::Double(0.0),
		      "Chemical potential residual tolerance");

    prm.declare_entry("Tolerance chemical potential", "5.0e-5",
		      dealii::Patterns::Double(0.0),
		      "Chemical potential error tolerance");
  }
  prm.leave_subsection();
}

void Parameters::NonlinearSolver::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Nonlinear solver");
  {
    max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");
    tol_f = prm.get_double("Tolerance force");
    tol_u = prm.get_double("Tolerance displacement");
    tol_fc = prm.get_double("Tolerance concentration residual");
    tol_c = prm.get_double("Tolerance concentration");
    tol_fmu = prm.get_double("Tolerance chemical potential residual");
    tol_mu = prm.get_double("Tolerance chemical potential");
  }
  prm.leave_subsection();
}

// -----------------------------------

void Parameters::Time::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Time");
  {
    prm.declare_entry("End time", "5000",
		      dealii::Patterns::Double(),
		      "End time");

    prm.declare_entry("Time step size", "1.0",
		      dealii::Patterns::Double(),
		      "Time step size");
  }
  prm.leave_subsection();
}

void Parameters::Time::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Time");
  {
    end_time = prm.get_double("End time");
    delta_t = prm.get_double("Time step size");
  }
  prm.leave_subsection();
}


Parameters::AllParameters::AllParameters(const std::string &input_file)
{
  dealii::ParameterHandler prm;
  declare_parameters(prm); 
  prm.read_input(input_file);
  parse_parameters(prm);
}

void Parameters::AllParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  FESystem::declare_parameters(prm);
  Geometry::declare_parameters(prm);
  Material_electrode::declare_parameters(prm);
  Material_electrolyte::declare_parameters(prm);
  ElectroChem::declare_parameters(prm);
  Fracture::declare_parameters(prm);
  LinearSolver::declare_parameters(prm);
  NonlinearSolver::declare_parameters(prm);
  Time::declare_parameters(prm);
}

void Parameters::AllParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  FESystem::parse_parameters(prm);
  Geometry::parse_parameters(prm);
  Material_electrode::parse_parameters(prm);
  Material_electrolyte::parse_parameters(prm);
  ElectroChem::parse_parameters(prm);
  Fracture::parse_parameters(prm);
  LinearSolver::parse_parameters(prm);
  NonlinearSolver::parse_parameters(prm);
  Time::parse_parameters(prm);
}
