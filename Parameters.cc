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
  }
  prm.leave_subsection();
}

void Parameters::Material_electrode::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrode material properties");
  {
    nu = prm.get_double("Poisson's ratio");
    E0 = prm.get_double("Young's modulus");
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
  }
  prm.leave_subsection();
}

void Parameters::Material_electrolyte::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrolyte material properties");
  {
    nu_el = prm.get_double("Poisson's ratio");
    E0_el = prm.get_double("Young's modulus");
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
  prm.parse_input(input_file);
  parse_parameters(prm);
}

void Parameters::AllParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  FESystem::declare_parameters(prm);
  Geometry::declare_parameters(prm);
  Material_electrode::declare_parameters(prm);
  Material_electrolyte::declare_parameters(prm);
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
  LinearSolver::parse_parameters(prm);
  NonlinearSolver::parse_parameters(prm);
  Time::parse_parameters(prm);
}
