#include "Parameters_ElecPot.h"

void Parameters::ElectroChem::declare_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrochemical properties");
  {
    prm.declare_entry("Interfacial resistance", "0.01",
		      dealii::Patterns::Double(0.0),
		      "Resistance at the anode interface [Ohm m2]");
    prm.declare_entry("Electrolyte conductivity", "0.1",
		      dealii::Patterns::Double(0.0, 1.0),
		      "Solid Electrolyte conductivity [S/m]");
    prm.declare_entry("Applied current density", "10.0",
		      dealii::Patterns::Double(0.0),
		      "Applied current density [A/m2]");
  }
  prm.leave_subsection();
}

void Parameters::ElectroChem::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Electrochemical properties");
  {
    rho = prm.get_double("Interfacial resistance");
    kappa = prm.get_double("Electrolyte conductivity");
    current =  prm.get_double("Applied current density");
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

    prm.declare_entry("Mesh file name", "640round_short",
		      dealii::Patterns::FileName(),
		      "Mesh file name (without extension)");
  }
  prm.leave_subsection();
}

void Parameters::Geometry::parse_parameters(dealii::ParameterHandler &prm)
{
  prm.enter_subsection("Geometry");
  {
    global_refinement = prm.get_integer("Global refinement");
    scale = prm.get_double("Grid scale");
    mesh_file = prm.get("Mesh file name");
  }
  prm.leave_subsection();
}

// -----------------------------------

Parameters::AllParameters::AllParameters(const std::string &input_file)
{
  dealii::ParameterHandler prm;
  declare_parameters(prm); 
  prm.parse_input(input_file);
  parse_parameters(prm);
}

void Parameters::AllParameters::declare_parameters(dealii::ParameterHandler &prm)
{
  ElectroChem::declare_parameters(prm);
  Geometry::declare_parameters(prm);
}

void Parameters::AllParameters::parse_parameters(dealii::ParameterHandler &prm)
{
  ElectroChem::parse_parameters(prm);
  Geometry::parse_parameters(prm);
}

