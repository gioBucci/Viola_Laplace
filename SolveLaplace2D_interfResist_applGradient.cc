/* ---------------------------------------------------------------------
 *
 * Copyright (C) by Robert Bosch LLC
 *
 * This program is based on the Viola code developed by Giovanna Bucci at MIT
 * and it makes extensive use of the deal.II library.
 *
 *
 * ---------------------------------------------------------------------
 */

#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

#include <deal.II/base/logstream.h>

using namespace dealii;

// @sect3{The <code>EPotential</code> class template}

// The  <code>EPotential</code> class is declared as a class
// with a template parameter, and the template parameter is the
// spatial dimension in which we would like to solve the Laplace equation. Of
// course, several of the member variables depend on this dimension as well,
// in particular the Triangulation class, which has to represent
// quadrilaterals or hexahedra, respectively.
template <int dim>
class EPotential
{
public:
  EPotential ();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;
  void PointValueEvaluation ();
  void PointXDerivativeEvaluation();

  unsigned int         numVariables;
  unsigned int         grid_choice;

  Triangulation<dim>   triangulation;
  FESystem<dim>        fe;
  DoFHandler<dim>      dof_handler;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  ConstraintMatrix     constraints;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  unsigned int         Xnormal_positive_plane;
  unsigned int         Xnormal_negative_plane;
  unsigned int         Xnormal_negative_plane_noCurr;
  unsigned int         Ynormal_positive_plane;
  unsigned int         Ynormal_negative_plane;

  bool                 applied_current;
  double               current_density;
  double               bulk_conductivity;
  double               surface_resistance;
  double               convective_coeff;

  std::vector<Point<dim> >  evaluation_points;
  std::vector<double>       normal_current_density;
  std::map<double,double>   evaluationPt_list;
  std::string mesh;
 
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


template <int dim>
class ComputeIntensity : public DataPostprocessorVector<dim>
{
public:
  ComputeIntensity ();

  virtual
  void
  compute_derived_quantities_scalar (const std::vector< double >               &uh,
                                     const std::vector<Tensor<1, dim> >        &duh,
                                     const std::vector<Tensor<2, dim> >        &dduh,
                                     const std::vector<Point<dim> >            &normals,
                                     const std::vector<Point<dim> >            &evaluation_points,
                                     std::vector<Vector<double> >              &computed_quantities) const;
};

// In the constructor, we need two arguments. The first denotes the name 
// by which the vector quantity computed by this class should be represented in output files. 
// The second argument is a set of flags that indicate which data 
// is needed by the postprocessor in order to compute the output quantities. 
template <int dim>
ComputeIntensity<dim>::ComputeIntensity ()
  :
  DataPostprocessorVector<dim> ("Electric_field",
                                update_gradients)
{}

// The actual postprocessing happens in the following function. 
// The derivative of the solution is used to calculate the output. 
// Make sure you have updated the derivative in the constructor.
// The derived quantities are returned in the computed_quantities vector. 

template <int dim>
void
ComputeIntensity<dim>::compute_derived_quantities_scalar (const std::vector< double >               & /*uh*/,
							  const std::vector<Tensor<1, dim> >        &duh,
							  const std::vector<Tensor<2, dim> >        & /*dduh*/,
							  const std::vector<Point<dim> >            & /*normals*/,
							  const std::vector<Point<dim> >            & /*evaluation_points*/,
							  std::vector<Vector<double> >              &computed_quantities) const
{
  Assert(computed_quantities.size() == duh.size(),
         ExcDimensionMismatch (computed_quantities.size(), duh.size()));
  // The computation itself is straightforward: We iterate over each entry in the output vector and compute the gradient:
  
  for (unsigned int i=0; i<computed_quantities.size(); i++)
    {
      Assert(computed_quantities[i].size() == dim,
             ExcDimensionMismatch (computed_quantities[i].size(), dim));
      
      for (unsigned int j=0; j<dim; j++)
	computed_quantities[i](j) = duh[i][j];
    }
}




// @sect4{EPotential::EPotential}

// Here is the constructor of the <code>EPotential</code>
// class. It specifies the desired polynomial degree of the finite elements
// and associates the DoFHandler to the triangulation
template <int dim>
EPotential<dim>::EPotential ()
  :
  numVariables(1),
  grid_choice(0),
  fe(FE_Q<dim>(1), numVariables),
  dof_handler (triangulation),
  Xnormal_positive_plane(0),
  Xnormal_negative_plane(1),
  Xnormal_negative_plane_noCurr(2),
  Ynormal_positive_plane(3),
  Ynormal_negative_plane(4),
  applied_current(1),
  // The properties below are hard-coded. They will be passed as input in future versions
  current_density(1000.0),          // 1 A/m2 -> 0.1 mA/cm2
  // the current density here is to be inteded as the actual applied current density divided by the conductivity
  bulk_conductivity(0.1),        // 1 mS/cm -> 0.1 S/m
  // The surface resistance is being varied in the simulations in the range 0.1-1000 Ohm cm2
  surface_resistance(0.001),    // 100 Ohm cm2  -> 0.01 Ohm m2
  convective_coeff(1.0 / (bulk_conductivity * surface_resistance)),
  mesh ("640round_long")

{}

// @sect4{EPotential::make_grid}

// Here we read the mesh from a gmsh file generated with external software. 
// We also loop over the cells and the boundary faces to assign boundary IDs. 
// The boundary IDs will be used later to impose boundary conditions
template <int dim>
void EPotential<dim>::make_grid ()
{
  // This tringulation represents a solid electrolyte with a surface crack
  if (grid_choice == 0) {
    dealii::GridIn<dim> gridin;
    gridin.attach_triangulation(triangulation);

    std::string input;
    input = "SEdomain_" + mesh + ".msh";

    std::ifstream f(input);
    gridin.read_msh(f);

    const double SEW = 1.0;
    const double SEH = 1.0;
    const double CrL = 0.2;
    const double CrW = CrL/16.0;	
    const double tol = 1.e-8;
  
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell!= triangulation.end(); ++cell)
      {
	if ( cell->at_boundary() == true )
	  for (unsigned int face = 0;
	       face < GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if ( cell->face(face)->at_boundary() == true ) 
		{ 
		  if ( (cell->face(face)->center()[0] <= CrL + tol && std::fabs(cell->face(face)->center()[1]) <= 1.25*CrW)
		       || (cell->face(face)->center()[0] <= tol
			   // && std::fabs( cell->face(face)->center()[1]) < 0.8*SEH
			   ) )
		    cell->face(face)->set_boundary_id(Xnormal_negative_plane);	
		  
		  else if (std::fabs( cell->face(face)->center()[0] - SEW) < tol)
		    cell->face(face)->set_boundary_id(Xnormal_positive_plane);		  
		  else if (std::fabs( cell->face(face)->center()[1] + SEH) < tol)
		    cell->face(face)->set_boundary_id(Ynormal_negative_plane);
		  else if (std::fabs( cell->face(face)->center()[1] - SEH) < tol)
		    cell->face(face)->set_boundary_id(Ynormal_positive_plane);
		}
	    }
      }
    
  }

  // Alternatively the mesh can be generated with dealii
  // when simple geometries are used
  else {
    const double edge = 0.50;
    const double tol = 1.e-6;
    // GridGenerator::hyper_cube (triangulation, -edge, edge);
    Point<dim> p1(0, -2*edge);
    Point<dim> p2(2*edge, 2*edge);
    
    GridGenerator::hyper_rectangle(triangulation, p1, p2);
    
    triangulation.refine_global (6);
    
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell!= triangulation.end(); ++cell)
      {
	if ( cell->at_boundary() == true )
	  for (unsigned int face = 0;
	       face < GeometryInfo<dim>::faces_per_cell; ++face)
	    {	
	      // faces on the external surface  
	      if (std::fabs( cell->face(face)->center()[0] + 0) < tol)
		cell->face(face)->set_boundary_id(Xnormal_negative_plane);
	      else if (std::fabs( cell->face(face)->center()[0] - 2*edge) < tol)
		cell->face(face)->set_boundary_id(Xnormal_positive_plane);
	      else if (std::fabs( cell->face(face)->center()[1] + 2*edge) < tol)
		cell->face(face)->set_boundary_id(Ynormal_negative_plane);
	      else if (std::fabs( cell->face(face)->center()[1] - 2*edge) < tol)
		cell->face(face)->set_boundary_id(Ynormal_positive_plane);
	    }
      }
  }
  GridTools::scale(1.e-5, triangulation);
  
  std::cout << "Mesh info:" << std::endl
	    << " dimension: " << dim << std::endl
	    << " no. of cells: " << triangulation.n_active_cells() << std::endl
	    << " minimum cell size" << GridTools::minimal_cell_diameter(triangulation) << std::endl;
}

// @sect4{EPotential::setup_system}

// In the system set-up the solution and residual vectors are initialized 
// and sized according to the total numeber of DOFs (problem's unknowns)
// The stiffness matrix is also initialized based on the sparsity pattern.
// The sparsity pattern takes into account the constrained nodes (this is why
// we generate the constrains first) and its purpouse is to optimize storage 
// and computation of the stiffness matrix
template <int dim>
void EPotential<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());


  constraints.clear ();
  
  if (!applied_current) {
    VectorTools::interpolate_boundary_values (dof_handler,
					      Xnormal_positive_plane,
					      ConstantFunction<dim>(numVariables,1.0),
					      constraints);
  }
  
  constraints.close ();
  
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);

  // FIND list of nodes along the interface where to record the values of potential and electric field
  unsigned int selected_boundary_id = Xnormal_negative_plane;
  
  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    if ( cell->at_boundary() == true )
      for (unsigned int face = 0; 
	   face < GeometryInfo<dim>::faces_per_cell; ++face)
	{
	  if ( cell->face(face)->at_boundary() == true && 
	       cell->face(face)->boundary_id() == selected_boundary_id)
	    {
	      for (unsigned int vertex=0;
		   vertex<GeometryInfo<dim>::vertices_per_face;
		   ++vertex)
		{
		  Point<dim>  evaluation_point = cell->face(face)->vertex(vertex);
		  // We list the Y coordinate first because that should be unique
		  evaluationPt_list.insert(std::make_pair(evaluation_point[1],evaluation_point[0]));
		}
	    }
	}
  
  std::cout << "  Number of unique points pound along the interface   " << evaluationPt_list.size()
	    << std::endl;

}

// @sect4{EPotential::assemble_system}

// For each cell (or finite element) we assemble the local stiffness matrix and residual.
// Then we copy them into the global tangent matrix and residual by calling the dealii
// function constraints.distribute_local_to_global (see dealii documentation for details)
template <int dim>
void EPotential<dim>::assemble_system ()
{
  QGauss<dim>  qf_cell(2); 
  QGauss<dim-1>  qf_face(2);

  FEValues<dim> fe_values (fe, qf_cell,
                           update_values   | 
			   update_gradients |
			   update_quadrature_points |
                           update_JxW_values);

  FEFaceValues<dim>    fe_face_values (fe, qf_face,
				       update_JxW_values |
				       update_normal_vectors |
				       update_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = qf_cell.size();
  const unsigned int   n_q_points_f  = qf_face.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
    cell = dof_handler.begin_active(),
    endc = dof_handler.end();

  for (; cell!=endc; ++cell)
    {
      fe_values.reinit (cell);
      cell_matrix = 0;
      cell_rhs = 0;

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        for (unsigned int i=0; i<dofs_per_cell; ++i)
          {
	    const unsigned int component_i = fe.system_to_component_index(i).first;	
	  
            for (unsigned int j=0; j<dofs_per_cell; ++j)
	      {
		const unsigned int component_j = fe.system_to_component_index(j).first;	
		
		if (component_i == component_j)
		  cell_matrix(i,j) += ( fe_values.shape_grad (i, q_index) *
					fe_values.shape_grad (j, q_index) *
					fe_values.JxW (q_index));
		
	      }
	    // if (applied_current) 
	    //   cell_rhs(i) += (fe_values.shape_value (i, q_index) *
	    //                   current_density *
	    //                   fe_values.JxW (q_index));
	  }

      if (cell->at_boundary())
      	{
      	  for (unsigned int face_number=0; face_number<GeometryInfo<dim>::faces_per_cell; ++face_number)
      	    if (cell->face(face_number)->at_boundary())
	      if (cell->face(face_number)->boundary_id() == Xnormal_negative_plane)
		{
		  fe_face_values.reinit (cell, face_number);
		  for (unsigned int fq_point=0; fq_point<n_q_points_f; ++fq_point)
		    { 
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const unsigned int component_i = fe.system_to_component_index(i).first;
			  const double Ni = fe_face_values.shape_value(i, fq_point);
			  const double JxW = fe_face_values.JxW(fq_point);
			  
			  for (unsigned int j=0; j<dofs_per_cell; ++j)
			    {
			      const unsigned int component_j = fe.system_to_component_index(j).first;
			      const double Nj = fe_face_values.shape_value(j, fq_point);
		    
			      if (component_i == component_j)
				cell_matrix(i,j) += convective_coeff * Ni * Nj * JxW;
			    }
      		      }
      		  }  
      	      }
	      else if  (cell->face(face_number)->boundary_id() == Xnormal_positive_plane
			&& applied_current) 
		{
		  fe_face_values.reinit (cell, face_number);
		  for (unsigned int fq_point=0; fq_point<n_q_points_f; ++fq_point)
		    { 
		      for (unsigned int i=0; i<dofs_per_cell; ++i)
			{
			  const double Ni = fe_face_values.shape_value(i, fq_point);
			  const double JxW = fe_face_values.JxW(fq_point);
			  cell_rhs(i) += Ni * current_density * JxW;
			}
		    }
		}
      	}

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
					     local_dof_indices,
					     system_matrix, system_rhs);
    }
}

// @sect4{EPotential::solve}

// Solve the linear system by Conjugate Gradient method
template <int dim>
void EPotential<dim>::solve ()
{
  SolverControl           solver_control (5000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;

  constraints.distribute(solution);
}

// @sect4{EPotential::output_results}

// Here we write the solution (electric potential) and the solution gradient 
// (electric field) in a format readable by Paraview
template <int dim>
void EPotential<dim>::output_results () const
{
  //  ComputeSolutionGradient<dim> electric_field;
  ComputeIntensity<dim> intensities;
  DataOut<dim> data_out;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(DataComponentInterpretation::component_is_scalar);
  
  std::vector<std::string> solution_name(numVariables, "potential");
  // solution_name.push_back("electric_field");
  
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, 
			   solution_name,
			   DataOut<dim>::type_dof_data,
			   data_component_interpretation);

  data_out.add_data_vector (solution, intensities);

  // data_out.add_data_vector(solution, electric_field);
			  
  data_out.build_patches ();

  std::ostringstream filename;
  filename << "potential-" << mesh << "-" << current_density/100.0 << "mA_cm2-"
	   << surface_resistance*10000 << "ohmCm2_interfResist-" << bulk_conductivity*10 <<  "mS_cm_bulkCond.vtk";

  std::ofstream output(filename.str().c_str());
  
  data_out.write_vtk (output);
}


// @sect4{EPotential::run}

// This is the function which has the top-level control over everything. Apart
// from one line of additional output, it is the same as for the previous
// example.
template <int dim>
void EPotential<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  std::cout << "Solving problem for " << numVariables << " number of variables." << std::endl;

  make_grid();
  setup_system ();
  assemble_system ();
  solve (); 
  output_results ();
  PointValueEvaluation();
  PointXDerivativeEvaluation();
}


int main ()
{
  // deallog.depth_console (0);
  {
    EPotential<2> laplace_problem_2d;
    laplace_problem_2d.run ();
  }
  return 0;
}

// Write the solution values (electric potential) at the nodes
// along the interface with the Li-metal anode. The data is written
// in a .csv file and it is later imported in a Mathematica notebook 
// for plotting and port-processing

template <int dim>
void EPotential<dim>::PointValueEvaluation ()
{
  // Set up file where to save the output
  std::ostringstream filename2;
      
  filename2 << "PotentialField-" << mesh << "-" << current_density/100.0 << "mA_cm2-"
	    << surface_resistance*10000 << "ohmCm2_interfResist-" << bulk_conductivity*10 <<  "mS_cm_bulkCond.csv";
  
  std::ofstream SurfacePotential_file (filename2.str().c_str());
  
  SurfacePotential_file.open (filename2.str().c_str(), std::ios::out | std::ios::trunc);
  SurfacePotential_file.close();
  
  SurfacePotential_file.open (filename2.str().c_str(), std::ios::app);
  SurfacePotential_file.precision(9); 
  SurfacePotential_file.setf(std::ios::fixed);
  SurfacePotential_file.setf(std::ios::showpoint);
  
  for (std::map<double,double>::iterator it = evaluationPt_list.begin(); 
       it != evaluationPt_list.end(); ++it)
    {
      Point<dim> evaluation_point (it->second,it->first);
      double point_value = 1e20;
      
      typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
      bool evaluation_point_found = false;
      for (; (cell!=endc) && !evaluation_point_found; ++cell)
	for (unsigned int vertex=0;
	     vertex<GeometryInfo<dim>::vertices_per_cell;
	     ++vertex)
	  if (cell->vertex(vertex).distance (evaluation_point)
	      <
	      cell->diameter() * 1e-8)
	    {
	      point_value = solution(cell->vertex_dof_index(vertex,0));
	      
	      evaluation_point_found = true;
	      break;
	    }
      
      AssertThrow (evaluation_point_found,
		   ExcInternalError());
      
      std::cout << "   Point value=" << point_value
		<< std::endl;
      
      SurfacePotential_file << evaluation_point[0] << ',' << evaluation_point[1] <<  ',' <<  point_value <<  '\n';
    }
  SurfacePotential_file.close();
}


// Write the solution normal gradient (electric field orthogonal 
// to the interface) at the nodes along the interface with the Li-metal anode. 
// The data is written in a .csv file and it is later imported in 
// a Mathematica notebook for plotting and port-processing

template <int dim>
void EPotential<dim>:: PointXDerivativeEvaluation()
{
  // Setup file where to store output data
  
  std::ostringstream filename;

  filename << "ElectricField-" << mesh << "-" << current_density/100.0 << "mA_cm2-"
	   << surface_resistance*10000 << "ohmCm2_interfResist-" << bulk_conductivity*10 <<  "mS_cm_bulkCond.csv";
  
  std::ofstream ElectricField_file (filename.str().c_str());
  
  ElectricField_file.open (filename.str().c_str(), std::ios::out | std::ios::trunc);
  ElectricField_file.close();
  
  ElectricField_file.open (filename.str().c_str(), std::ios::app);
  ElectricField_file.precision(9); 
  ElectricField_file.setf(std::ios::fixed);
  ElectricField_file.setf(std::ios::showpoint);

  // We use a special quadrature rule with points at the vertices, since these are
  // what we are interested in. The appropriate rule is the trapezoidal rule
  
  QTrapez<dim-1>  vertex_quadrature;
  
  FEFaceValues<dim> fe_face_values (dof_handler.get_fe(),
				    vertex_quadrature,
				    update_values | 
				    update_gradients |
				    update_normal_vectors |
				    update_quadrature_points);
  
  std::vector<Tensor<1,dim> >
    solution_gradients (vertex_quadrature.size());
  
  for (std::map<double,double>::iterator it = evaluationPt_list.begin(); 
       it != evaluationPt_list.end(); ++it)
    {
      Point<dim> evaluation_point (it->second,it->first);
      double point_derivative = 0;
      Tensor<1, dim> point_tensor_derivative;
      unsigned int evaluation_point_hits = 0;
      
      typename DoFHandler<dim>::active_cell_iterator
	cell = dof_handler.begin_active(),
	endc = dof_handler.end();
      for (; cell!=endc; ++cell)	
	if ( cell->at_boundary() == true )
	  for (unsigned int face = 0; 
	       face < GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      for (unsigned int vertex=0;
		   vertex<GeometryInfo<dim>::vertices_per_face;
		   ++vertex)
		if (cell->face(face)->vertex(vertex).distance (evaluation_point)
		    <
		    cell->diameter() * 1e-8)
		  {
		    // Initialize the <code>FEValues</code> object on this cell
		    //  fe_values.reinit (cell);
		    // Initialize the FEFaceValues object on this cell and face
		    fe_face_values.reinit(cell, face);
		    // and extract the gradients of the solution vector at the
		    // vertices:
		    fe_face_values.get_function_gradients (solution,
							   solution_gradients);
		    
		    // Now we have the gradients at all vertices, so pick out that
		    // one which belongs to the evaluation point (note that the
		    // order of vertices is not necessarily the same as that of the
		    // quadrature points):
		    unsigned int q_point = 0;
		    for (; q_point<solution_gradients.size(); ++q_point)
		      if (fe_face_values.quadrature_point(q_point).distance (evaluation_point)
			  <
			  cell->diameter() * 1e-8)
			break;
		    
		    // Check that the evaluation point was indeed found,
		    Assert (q_point < solution_gradients.size(),
			    ExcInternalError());
		    // and if so take the derivative of the gradient
		    // and sum the contributions form contigous cells
		    // nb: make sure to sum the as vectors, not as scalars (e.g. normal component or norm)
		    // and increase the counter
		 
		    point_tensor_derivative  += solution_gradients[q_point];
		    ++evaluation_point_hits;
		    
		    // Finally break out of the innermost loop iterating over the
		    // vertices of the present cell, since if we have found the
		    // evaluation point at one vertex it cannot be at a following
		    // vertex as well
		    break;
		  }
	    }
 
      AssertThrow (evaluation_point_hits > 0,
		   ExcInternalError());

      // Now we have looped over all faces we average the values between neighbouring faces
      point_derivative =  point_tensor_derivative.norm() / evaluation_point_hits;
   
      std::cout << "   Point normal-derivative=" << point_derivative
		<< std::endl;
      
      ElectricField_file << evaluation_point[0] << ',' << evaluation_point[1] <<  ',' <<  point_derivative <<  '\n';
    }
  ElectricField_file.close();
}
