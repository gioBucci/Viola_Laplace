/* ---------------------------------------------------------------------
 * This code is self-contained. It solves the laplace equation to smooth
 * the mesh after moving the nodes along one of the bondaries along the  
 * direction normal to the boundary itself
 * ---------------------------------------------------------------------
 *
 * Author: Giovanna Bucci, Robert Bosch LLC, 2018
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

template <int dim>
class Step4
{
public:
  Step4 ();
  void run ();

private:
  void make_grid ();
  void setup_system();
  void assemble_system ();
  void solve ();
  void output_results () const;

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
  unsigned int         Ynormal_positive_plane;
  unsigned int         Ynormal_negative_plane;
};


template <int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};



template <int dim>
class BoundaryValues : public Function<dim>
{
public:
  BoundaryValues () : Function<dim>() {}

  virtual double value (const Point<dim>   &p,
                        const unsigned int  component = 0) const;
};

template <int dim>
class BoundaryVectorValues : public Function<dim>
{
public:
  BoundaryVectorValues () : Function<dim>() {}

  virtual Tensor<1,dim> value (const Point<dim>   &p,
			       const unsigned int  component = 0) const;
};

      

template <int dim>
double RightHandSide<dim>::value (const Point<dim> &p,
                                  const unsigned int /*component*/) const
{
  double return_value = 0.0;

  return return_value;
}


// As boundary values, we choose $x^2+y^2$ in 2D, and $x^2+y^2+z^2$ in 3D. This
// happens to be equal to the square of the vector from the origin to the
// point at which we would like to evaluate the function, irrespective of the
// dimension. So that is what we return:
template <int dim>
double BoundaryValues<dim>::value (const Point<dim> &p,
                                   const unsigned int /*component*/) const
{
  return 0.0; 
}

// - - - - 

template <int dim>
Tensor<1,dim> BoundaryVectorValues<dim>::value (const Point<dim> &p,
						const unsigned int /*component*/) const
{
  Tensor<1,dim> velocity;
  velocity[0] = 0.1;
  if (dim>1)
    for (unsigned int i=1; i<dim; ++i)
      velocity[i] = 0.0;
  
  return velocity;
}

template <int dim>
Step4<dim>::Step4 ()
  :
  numVariables(2),
  grid_choice(0),
  fe(FE_Q<dim>(1), numVariables),
  dof_handler (triangulation),
  Xnormal_positive_plane(0),
  Xnormal_negative_plane(1),
  Ynormal_positive_plane(2),
  Ynormal_negative_plane(3)
{}


template <int dim>
void Step4<dim>::make_grid ()
{
  if (grid_choice == 0) {
      dealii::GridIn<dim> gridin;
      gridin.attach_triangulation(triangulation);

      std::string input;
      input = "LiMetal.msh";
      std::ifstream f(input);
      gridin.read_msh(f);

      const double LiW = 0.1;
      const double LiH = 1.0;
      const double tol = 1.e-5;
  
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell!= triangulation.end(); ++cell)
      {
	if ( cell->at_boundary() == true )
	  for (unsigned int face = 0;
	       face < GeometryInfo<dim>::faces_per_cell; ++face)
	    {
	      if ( cell->face(face)->at_boundary() == true ) 
		{ 
		  if ( cell->face(face)->center()[0] + LiW < tol)
		    cell->face(face)->set_boundary_id(Xnormal_negative_plane);
		  
		  else if ( cell->face(face)->center()[0] >= 0 && cell->face(face)->center()[1] > 0 )
		    cell->face(face)->set_boundary_id(Xnormal_positive_plane);
		  
		  else if (std::fabs( cell->face(face)->center()[1] ) < tol)
		    cell->face(face)->set_boundary_id(Ynormal_negative_plane);
		  else if (std::fabs( cell->face(face)->center()[1] - LiH) < tol)
		    cell->face(face)->set_boundary_id(Ynormal_positive_plane);
		}
	    }
      }
    
  }

  else {
    const double edge = 1.0;
    GridGenerator::hyper_rectangle(triangulation,
				   (dim==3 ? Point<dim>(-edge, -edge, -edge) : Point<dim>(-edge, -edge)),
				   (dim==3 ? Point<dim>(edge, edge, edge) : Point<dim>(edge, edge)),
				   true);

    triangulation.refine_global(6); 
    const double tol = 1.e-4 * GridTools::minimal_cell_diameter(triangulation);
    
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell!= triangulation.end(); ++cell)
      {
	if ( cell->at_boundary() == true )
	  for (unsigned int face = 0;
	       face < GeometryInfo<dim>::faces_per_cell; ++face)
	    {	
	      // faces on the external surface  
	      if (std::fabs( cell->face(face)->center()[0] + edge) < tol)
		cell->face(face)->set_boundary_id(Xnormal_negative_plane);
	      else if (std::fabs( cell->face(face)->center()[0] - edge) < tol)
		cell->face(face)->set_boundary_id(Xnormal_positive_plane);
	      else if (std::fabs( cell->face(face)->center()[1] + edge) < tol)
		cell->face(face)->set_boundary_id(Ynormal_negative_plane);
	      else if (std::fabs( cell->face(face)->center()[1] - edge) < tol)
		cell->face(face)->set_boundary_id(Ynormal_positive_plane);
	    }
      }
  }
  std::cout << "Mesh info:" << std::endl
	    << " dimension: " << dim << std::endl
	    << " no. of cells: " << triangulation.n_active_cells() << std::endl;
}


template <int dim>
void Step4<dim>::setup_system ()
{
 dof_handler.distribute_dofs (fe);
  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());


  constraints.clear ();

  std::set<types::boundary_id> normal_flux_boundaries;
  double velocity = 0.005;
  normal_flux_boundaries.insert (Xnormal_positive_plane);

  typename FunctionMap<dim>::type      boundary_functions;
  ConstantFunction<dim>                constant_velocity_bc (velocity,numVariables);
  boundary_functions[0] = &constant_velocity_bc; 
  
  VectorTools::compute_nonzero_normal_flux_constraints	( dof_handler,
  							  0,                       //  first_vector_component,
  							  normal_flux_boundaries,
  							  boundary_functions,
  							  constraints,
  							  StaticMappingQ1< dim >::mapping 
  							  );
  
  const FEValuesExtractors::Scalar x_displacement(0);
  const FEValuesExtractors::Scalar y_displacement(1);

  {
    VectorTools::interpolate_boundary_values (dof_handler,
					      Xnormal_negative_plane,
					      ZeroFunction<dim>(numVariables),
					      constraints,
					      fe.component_mask(x_displacement));
    
    VectorTools::interpolate_boundary_values (dof_handler,
					      Ynormal_negative_plane,
					      ZeroFunction<dim>(numVariables),
					      constraints,
					      fe.component_mask(y_displacement));
    
    VectorTools::interpolate_boundary_values (dof_handler,
					      Ynormal_positive_plane,
					      ZeroFunction<dim>(numVariables),
					      constraints,
					      fe.component_mask(y_displacement));
  }

  constraints.close ();
  
  
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ true);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit (sparsity_pattern);

}


template <int dim>
void Step4<dim>::assemble_system ()
{
  QGauss<dim>  quadrature_formula(2);


  const RightHandSide<dim> right_hand_side;

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values   | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();

  const double coefficient = 0.01;

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
		// const unsigned int j_group = fe.system_to_base_index(j).first.first;
		const unsigned int component_j = fe.system_to_component_index(j).first;	
		
		if (component_i == component_j)
		  cell_matrix(i,j) += (coefficient *
				       fe_values.shape_grad (i, q_index) *
				       fe_values.shape_grad (j, q_index) *
				       fe_values.JxW (q_index));
	      }
            cell_rhs(i) += (fe_values.shape_value (i, q_index) *
                            right_hand_side.value (fe_values.quadrature_point (q_index)) *
                            fe_values.JxW (q_index));
	    
          }
     

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs,
					     local_dof_indices,
					     system_matrix, system_rhs);
    }

}


template <int dim>
void Step4<dim>::solve ()
{
  SolverControl           solver_control (5000, 1e-12);
  SolverCG<>              solver (solver_control);
  solver.solve (system_matrix, solution, system_rhs,
                PreconditionIdentity());

  // We have made one addition, though: since we suppress output from the
  // linear solvers, we have to print the number of iterations by hand.
  std::cout << "   " << solver_control.last_step()
            << " CG iterations needed to obtain convergence."
            << std::endl;

  constraints.distribute(solution);
}



template <int dim>
void Step4<dim>::output_results () const
{
  DataOut<dim> data_out;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(numVariables,
				  DataComponentInterpretation::component_is_part_of_vector);

  
  std::vector<std::string> solution_name(numVariables, "displacement");

  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
			   solution_name,
			   DataOut<dim>::type_dof_data,
			   data_component_interpretation);
  
  data_out.build_patches ();
  
  std::ofstream output (dim == 2 ?
                        "solution-2d.vtk" :
                        "solution-1d.vtk");
  data_out.write_vtk (output);
}


template <int dim>
void Step4<dim>::run ()
{
  std::cout << "Solving problem in " << dim << " space dimensions." << std::endl;
  std::cout << "Solving problem for " << numVariables << " number of variables." << std::endl;

  make_grid();
  setup_system ();
  assemble_system ();
  solve ();
  output_results ();
}


int main ()
{
  // deallog.depth_console (0);
  {
    Step4<2> laplace_problem_2d;
    laplace_problem_2d.run ();
  }

  return 0;
}
