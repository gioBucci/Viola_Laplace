// ---------------------------------------------------------------------
/*
 * Author: Giovanna Bucci, Department of Materials Science and Engineering
 *         Massachussets Institute of Technology, March 2014.
 */
// ---------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>

#include <deal.II/fe/fe_dgp_monomial.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>

#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/compressed_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <iostream>
#include <fstream>
#include <math.h>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>

#include "standard_tensors.h"
#include "time.h"
#include "Parameters.h"
#include "PointHistory2D_mech.h"


#define NON_UNIFORM_REFINEMENT 0
#define NONHOM_DIRICHLET 0       
#define NEUMANN_BOUNDARY_COND 1
#define TRACTION_LOAD 0
#define GROWTH 0


namespace Step44
{
  using namespace dealii;

  class Time
  {
  public:
    Time (const double time_end,
	  const double delta_t)
      :
      timestep(0),
      time_current(0.0),
      time_end(time_end),
      delta_t(delta_t)
    {}

    virtual ~Time()
    {}
  
    double current() const
    {
      return time_current;
    }
  
    double end() const
    {
      return time_end;
    }
  
    double get_delta_t() const
    {
      return delta_t;
    }
  
    unsigned int get_timestep() const
    {
      return timestep;
    }
  
    void increment()
    {
      time_current += delta_t;
      ++timestep;
    }
  
  private:
    unsigned int timestep;
    double time_current;
    const double time_end;
    const double delta_t;
  };

  template <int dim>
  class Solid
  {
  public:
    Solid(const std::string &input_file);

    virtual
    ~Solid();

    void
    run();

  private:

    struct PerTaskData_K;
    struct ScratchData_K;

    struct PerTaskData_M;
    struct ScratchData_M;

    struct PerTaskData_RHS;
    struct ScratchData_RHS;

    struct PerTaskData_UQPH;
    struct ScratchData_UQPH;

    // To handle two different materials for the electrode and electrolyte domain
    enum
      {
        electrode_domain_id,
	electrolyte_domain_id
      };
    
    static bool
    cell_is_in_electrode_domain (const typename Triangulation<dim>::active_cell_iterator &cell);
    
    static bool
    cell_is_in_electrolyte_domain (const typename Triangulation<dim>::active_cell_iterator &cell);

    void
    make_grid(const int &grid_f);

    // Set up the finite element system to be solved:
    void
    system_setup();

    void
    make_nonhomDirichlet_constraint();

    void
    impose_nonhomog_DirichletBC();

    // void
    // determine_component_extractors();

    void
    assemble_system_tangent();

    void
    assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
				     ScratchData_K &scratch,
				     PerTaskData_K &data);
    void
    copy_local_to_global_K(const PerTaskData_K &data);

    void
    copy_local_to_global_M(const PerTaskData_M &data);

    void
    assemble_system_rhs();

    void
    assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
				 ScratchData_RHS &scratch,
				 PerTaskData_RHS &data);

    void
    copy_local_to_global_rhs(const PerTaskData_RHS &data);

    // Apply Dirichlet boundary conditions
    void
    make_constraints(const int &it_nr);

    // Create and update the quadrature points. Here, no data needs to be
    // copied into a global object, so the copy_local_to_global function is
    // empty:
    void
    setup_qph();

    void
    update_qph_incremental(const Vector<double> &solution_delta);

    void
    update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
				    ScratchData_UQPH &scratch,
				    PerTaskData_UQPH &data);

    void
    copy_local_to_global_UQPH(const PerTaskData_UQPH &/*data*/)
    {}

    // Solve for the displacement using a Newton-Raphson method. We break this
    // function into the nonlinear loop and the function that solves the
    // linearized Newton-Raphson step:
    
    void
    solve_nonlinear_timestep(Vector<double> &solution_delta); 
    std::pair<unsigned int, double>
    solve_linear_system(Vector<double> &newton_update);

    // Solution retrieval as well as post-processing and writing data to file:
    Vector<double>
    get_total_solution(const Vector<double> &solution_delta) const;

    void
    output_results() const;

    void 
    output_stress() const;

    Parameters::AllParameters        parameters;
    Parameters::Fracture             parameters_Fracture;

    double                           vol_reference;
    double                           vol_current;
    double                           max_cell_diameter;
    double                           min_cell_diameter;

    // ...and description of the geometry on which the problem is solved:
    Triangulation<dim>               triangulation;

    Time                             time;
    TimerOutput                      timer;

    // A storage object for quadrature point information.  
    std::vector<PointHistory<dim> >  quadrature_point_history;

    const unsigned int               degree;
    const FESystem<dim>              fe;
    DoFHandler<dim>                  dof_handler_ref;
    const unsigned int               dofs_per_cell;
    const unsigned int               dofs_per_face;
    const FEValuesExtractors::Vector u_fe;

    static const unsigned int        n_components = dim;
    static const unsigned int        first_u_component = 0;
   
    const QGauss<dim>                qf_cell;
    const QGauss<dim - 1>            qf_face;
    const unsigned int               n_q_points;
    const unsigned int               n_q_points_f;

    ConstraintMatrix                 constraints;
    SparsityPattern                  sparsity_pattern;
    SparseMatrix<double>             tangent_matrix;
    // SparseMatrix<double>             mass_matrix;

    Vector<double>                   system_rhs;
    Vector<double>                   solution_n;
    Vector<double>                   solution_current;
    Vector<double>                   tmp_vector;
    Vector<double>                   growth_vector;
  
    unsigned int                     Xnormal_positive_plane;
    unsigned int                     Xnormal_negative_plane;
    unsigned int                     Ynormal_positive_plane;
    unsigned int                     Ynormal_negative_plane;

    //   . . . . . . . . GROWTH MODEL . . . . . . . . . . .
#if GROWTH
    void setup_system_gr();
    void assemble_system_gr ();
    void solve_gr ();
    void output_results_gr () const;
    
    unsigned int         numVariables;    
    FESystem<dim>        fe_gr;
    DoFHandler<dim>      dof_handler_gr;
    
    SparsityPattern      sparsity_pattern_gr;
    SparseMatrix<double> system_matrix_gr;
    ConstraintMatrix     constraints_gr;
    
    Vector<double>       system_rhs_gr;
#endif

    // Then define a number of variables to store norms and update norms and
    // normalisation factors.
    struct Errors
    {
      Errors()
	:
	norm(1.0) //, u(1.0), c(1.0), mu(1.0)
      {}
      
      void reset()
      {
	norm = 1.0;
      }
      void normalise(const Errors &rhs)
      {
	if (rhs.norm != 0.0)
	  norm /= rhs.norm;
      }

      double norm, u, c, mu;
    };

    Errors error_residual, error_residual_0, error_residual_norm, error_update,
      error_update_0, error_update_norm;

    // Methods to calculate error measures
    void
    get_error_residual(Errors &error_residual);

    void
    get_error_update(const Vector<double> &newton_update,
		     Errors &error_update);

    // Print information to screen in a pleasing way...
    static
    void
    print_conv_header();

    void
    print_conv_footer();
  };

  // @sect3{Implementation of the <code>Solid</code> class}

  // @sect4{Public interface}

  // We initialise the Solid class using data extracted from the parameter file.
  template <int dim>
  Solid<dim>::Solid(const std::string &input_file)
    :
    parameters(input_file),
    triangulation(Triangulation<dim>::maximum_smoothing),
    time(parameters.end_time, parameters.delta_t),
    timer(std::cout,
	  TimerOutput::summary,
	  TimerOutput::wall_times),
    degree(parameters.poly_degree),
    fe(FE_Q<dim>(parameters.poly_degree), dim),       
    dof_handler_ref(triangulation),
    dofs_per_cell (fe.dofs_per_cell),
    dofs_per_face (fe.dofs_per_face),
    u_fe(first_u_component),
    qf_cell(parameters.quad_order),  // I may need a higher number of quadatrure points to integrate the mass matrix
    qf_face(parameters.quad_order),
    n_q_points (qf_cell.size()),
    n_q_points_f (qf_face.size()),
    Xnormal_positive_plane(100),
    Xnormal_negative_plane(101),
    Ynormal_positive_plane(102),
    Ynormal_negative_plane(103)
    //   . . . . . . . . GROWTH MODEL . . . . . . . . . . .
#if GROWTH
     numVariables(dim),
    fe_gr(FE_Q<dim>(parameters.poly_degree), numVariables),
    dof_handler_gr (triangulation)
#endif
  {}

  template <int dim>
  Solid<dim>::~Solid()
  {
    dof_handler_ref.clear();
  }

  template <int dim>
  void Solid<dim>::run()
  {
    make_grid(parameters.grid_flag);
    system_setup();
    {
      ConstraintMatrix constraints;         
      constraints.close();
    }
    output_results();
    time.increment();

    Vector<double> solution_delta(dof_handler_ref.n_dofs());

    while (time.current() <= time.end())
      {
	solution_delta = 0.0;
	solution_current = solution_n;

#if GROWTH

	std::cout << "Solving growth problem in " << dim << " space dimensions." << std::endl;
	
	setup_system_gr ();
	assemble_system_gr ();
	solve_gr ();
	output_results_gr ();
	
#endif

	solve_nonlinear_timestep(solution_delta); 
	solution_n += solution_delta; 

	output_results();

	time.increment();
      }
  }


  template <int dim>
  struct Solid<dim>::PerTaskData_K
  {
    FullMatrix<double>        cell_matrix;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_K(const unsigned int dofs_per_cell)
      :
      cell_matrix(dofs_per_cell, dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_matrix = 0.0;
    }
  };


  template <int dim>
  struct Solid<dim>::ScratchData_K
  {
    FEValues<dim> fe_values_ref;

    std::vector<std::vector<double> >                   Nx;
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;

    ScratchData_K(const FiniteElement<dim> &fe_cell,
		  const QGauss<dim> &qf_cell,
		  const UpdateFlags uf_cell)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      Nx(qf_cell.size(),
	 std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
	      std::vector<Tensor<2, dim> >(fe_cell.dofs_per_cell))
    {}

    ScratchData_K(const ScratchData_K &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
		    rhs.fe_values_ref.get_quadrature(),
		    rhs.fe_values_ref.get_update_flags()),
      Nx(rhs.Nx),
      grad_Nx(rhs.grad_Nx)
    {}

    void reset()
    {
      const unsigned int n_q_points = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	{
	  Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
	  Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
		  ExcInternalError());
	  for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
	    {
	      Nx[q_point][k] = 0.0;
	      grad_Nx[q_point][k] = 0.0;
	    }
	}
    }
    
  };

  template <int dim>
  struct Solid<dim>::PerTaskData_RHS
  {
    Vector<double>            cell_rhs;
    std::vector<types::global_dof_index> local_dof_indices;

    PerTaskData_RHS(const unsigned int dofs_per_cell)
      :
      cell_rhs(dofs_per_cell),
      local_dof_indices(dofs_per_cell)
    {}

    void reset()
    {
      cell_rhs = 0.0;
    }
  };


  template <int dim>
  struct Solid<dim>::ScratchData_RHS
  {
    FEValues<dim>     fe_values_ref;
    FEFaceValues<dim> fe_face_values_ref;

    std::vector<std::vector<double> >                   Nx;
    std::vector<std::vector<Tensor<2, dim> > >          grad_Nx;

    ScratchData_RHS(const FiniteElement<dim> &fe_cell,
		    const QGauss<dim> &qf_cell, const UpdateFlags uf_cell,
		    const QGauss<dim - 1> &qf_face, const UpdateFlags uf_face)
      :
      fe_values_ref(fe_cell, qf_cell, uf_cell),
      fe_face_values_ref(fe_cell, qf_face, uf_face),
      Nx(qf_cell.size(),
	 std::vector<double>(fe_cell.dofs_per_cell)),
      grad_Nx(qf_cell.size(),
	      std::vector<Tensor<2, dim> >
	      (fe_cell.dofs_per_cell))
    {}

    ScratchData_RHS(const ScratchData_RHS &rhs)
      :
      fe_values_ref(rhs.fe_values_ref.get_fe(),
		    rhs.fe_values_ref.get_quadrature(),
		    rhs.fe_values_ref.get_update_flags()),
      fe_face_values_ref(rhs.fe_face_values_ref.get_fe(),
			 rhs.fe_face_values_ref.get_quadrature(),
			 rhs.fe_face_values_ref.get_update_flags()),
      Nx(rhs.Nx),
      grad_Nx(rhs.grad_Nx)
    {}

    void reset()
    {
      const unsigned int n_q_points      = Nx.size();
      const unsigned int n_dofs_per_cell = Nx[0].size();
      for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	{
	  Assert( Nx[q_point].size() == n_dofs_per_cell, ExcInternalError());
	  Assert( grad_Nx[q_point].size() == n_dofs_per_cell,
		  ExcInternalError());
	  for (unsigned int k = 0; k < n_dofs_per_cell; ++k)
	    {
	      Nx[q_point][k] = 0.0;
	      grad_Nx[q_point][k] = 0.0;
	    }
	}
    }

  };

 
  template <int dim>
  struct Solid<dim>::PerTaskData_UQPH
  {
    void reset() 
    {}
  };


  template <int dim>
  struct Solid<dim>::ScratchData_UQPH
  {
    const Vector<double>   &solution_total;
    const Vector<double>   &solution_delta;
    const Vector<double>   &growth_vector;

    std::vector<Tensor<2, dim> > solution_grads_u_total;
    std::vector<Tensor<2, dim> > growth_grads_u_total;

    FEValues<dim>                fe_values_ref;

    ScratchData_UQPH(const FiniteElement<dim> &fe_cell,
		     const QGauss<dim> &qf_cell,
		     const UpdateFlags uf_cell,
		     const Vector<double> &solution_total,
		     const Vector<double> &solution_delta,
		     const Vector<double> &growth_vector)
      :
      solution_total(solution_total),
      solution_delta(solution_delta),
      growth_vector(growth_vector),
      solution_grads_u_total(qf_cell.size()),
      growth_grads_u_total(qf_cell.size()),
      fe_values_ref(fe_cell, qf_cell, uf_cell)
    {}

    ScratchData_UQPH(const ScratchData_UQPH &rhs)
      :
      solution_total(rhs.solution_total),
      solution_delta(rhs.solution_delta),
      growth_vector(rhs.growth_vector),
      solution_grads_u_total(rhs.solution_grads_u_total),
      growth_grads_u_total(rhs.growth_grads_u_total),
      fe_values_ref(rhs.fe_values_ref.get_fe(),
		    rhs.fe_values_ref.get_quadrature(),
		    rhs.fe_values_ref.get_update_flags())
    {}

    void reset()
    {
      const unsigned int n_q_points = solution_grads_u_total.size();
      for (unsigned int q = 0; q < n_q_points; ++q)
	{
	  solution_grads_u_total[q] = 0.0;
	  growth_grads_u_total[q] = 0.0;
	}
    }
  };

  // - - - - - - - - - - - - - - - - - -
  template <int dim>
  bool
  Solid<dim>::cell_is_in_electrode_domain (const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    return (cell->material_id() == electrode_domain_id);
  }

  template <int dim>
  bool
  Solid<dim>::cell_is_in_electrolyte_domain (const typename Triangulation<dim>::active_cell_iterator &cell)
  {
    return (cell->material_id() == electrolyte_domain_id);
  }


  template <int dim>
  void Solid<dim>::make_grid(const int &grid_f)
  {
    if (grid_f == 0) {
      dealii::GridIn<dim> gridin;
      gridin.attach_triangulation(triangulation);

      std::string input;
      // std::cout << "Please enter the name of the input mesh file (gmsh format): ";
      // std::cin >> input;
      // input = "2DNotchedSample_LImetal_01_0.25HR_half.msh";
      
      input = "2DNotchedSample_LImetal_01_0.25HR_half.msh";
      

      std::ifstream f(input);
      gridin.read_msh(f);
      
      const double tol = 1.e-5;
      vol_reference = GridTools::volume(triangulation);
      const double yedge = 0.5;
      const double xedge = 1.0;

      for (typename Triangulation<dim>::active_cell_iterator cell =
	     triangulation.begin_active(); cell!= triangulation.end(); ++cell)
	{
	  if ( // cell_is_in_electrolyte_domain(cell)
	      cell->at_boundary() == true )
	    for (unsigned int face = 0;
		 face < GeometryInfo<dim>::faces_per_cell; ++face)
	      {	
		// faces on the external surface  
		if (std::fabs( cell->face(face)->center()[0] + 0.1*xedge) < tol)
		  cell->face(face)->set_boundary_id(Xnormal_negative_plane);
		else if (std::fabs( cell->face(face)->center()[0] - xedge) < tol)
		  cell->face(face)->set_boundary_id(Xnormal_positive_plane);
		else if (std::fabs( cell->face(face)->center()[1] + 0.0) < tol)
		  cell->face(face)->set_boundary_id(Ynormal_negative_plane);
		else if (std::fabs( cell->face(face)->center()[1] - yedge) < tol)
		  cell->face(face)->set_boundary_id(Ynormal_positive_plane);
	      }
	}
    }

    else if (grid_f == 1)
      {
	const double edge = 1.0;
	GridGenerator::hyper_rectangle(triangulation,
				       (dim==3 ? Point<dim>(-edge, -edge, -edge) : Point<dim>(-edge, -edge)),
				       (dim==3 ? Point<dim>(edge, edge, edge) : Point<dim>(edge, edge)),
				       true);

	const double tol = 1.e-5 * edge;
	
	for (typename Triangulation<dim>::active_cell_iterator cell =
	       triangulation.begin_active(); cell!= triangulation.end(); ++cell)
	  {
	    cell->set_material_id(0);
	    if ( cell->at_boundary() == true )
	      for (unsigned int face = 0;
		   face < GeometryInfo<dim>::faces_per_cell; ++face)
		{	
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
    
    // SCALE the entire mesh to the micrometer scale
    GridTools::scale(parameters.scale, triangulation);  

    
#if NON_UNIFORM_REFINEMENT
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell!= triangulation.end(); ++cell)
      {
	if ( cell_is_in_electrode_neighborhood(cell) || cell_is_in_electrolyte_neighborhood(cell) )
	  cell->set_refine_flag ();
      }
    triangulation.prepare_coarsening_and_refinement ();
    triangulation.execute_coarsening_and_refinement ();
    
    
    std::cout << "Triangulation:"
	      << "\n\t Number of active cells: " << triangulation.n_active_cells()
	      << "\n\t Number of vertices: " << triangulation.n_vertices()	
	      << std::endl;
#else
    triangulation.refine_global(std::max (1U, parameters.global_refinement)); 
#endif	    
    
   
    // Draw the mesh on a svg file to check if material IDs are correct
    // 
    std::ofstream out ("mesh.svg");
    dealii::GridOut grid_out;
    GridOutFlags::Svg svg_flags;
    svg_flags.coloring = GridOutFlags::Svg::Coloring::material_id;
    svg_flags.label_material_id = true;
    svg_flags.label_level_number = false;
    svg_flags.label_cell_index = false;
    grid_out.set_flags(svg_flags);
    grid_out.write_svg (triangulation, out);
    
    max_cell_diameter = GridTools::maximal_cell_diameter(triangulation);
    min_cell_diameter = GridTools::minimal_cell_diameter(triangulation);
    std::cout << "Grid:\n\t Reference area: \t" << vol_reference << std::endl;
    std::cout << "Maximum cell size \t" << max_cell_diameter << std::endl;
    std::cout << "Minimum cell size \t" << min_cell_diameter << std::endl;
    
  }
  

  template <int dim>
  void Solid<dim>::system_setup()
  {
    timer.enter_subsection("Setup system");

    dof_handler_ref.distribute_dofs(fe);
    DoFRenumbering::Cuthill_McKee(dof_handler_ref);

    std::cout << "Triangulation:"
	      << "\n\t Number of active cells: " << triangulation.n_active_cells()
	      << "\n\t Number of vertices: " << triangulation.n_vertices()	
	      << "\n\t Number of degrees of freedom: " << dof_handler_ref.n_dofs()
	      << std::endl;

    // Setup the sparsity pattern and tangent matrix
    tangent_matrix.clear();
    {
      DynamicSparsityPattern dsp(dof_handler_ref.n_dofs());

      DoFTools::make_sparsity_pattern(dof_handler_ref,
				      dsp,
				      constraints,
				      true);          // = keep_constrained_dofs

      sparsity_pattern.copy_from(dsp);

    }

    tangent_matrix.reinit(sparsity_pattern);
    
    system_rhs.reinit(dof_handler_ref.n_dofs());
  
    solution_n.reinit(dof_handler_ref.n_dofs());
    solution_current.reinit(dof_handler_ref.n_dofs());
    growth_vector.reinit(dof_handler_ref.n_dofs());
    
    setup_qph();

#if NONHOM_DIRICHLET
    impose_nonhomog_DirichletBC();
#endif

    timer.leave_subsection();
  }

#if NONHOM_DIRICHLET
  // @sect4{Solid::impose_nonhomog_DirichletBC}
  // This function is called in the system_setup and at the first Newton iteration
  // in order to impose the non-homogeneous Dirichlet boundary conditions.
  // The following operations are executed (being the constraints already filled) 
  // 1. static condensation of residual and stiffness matrix
  // 2. constraints are distributed to a temptative solution_delta 
  //    (increment at the end of the newton iteration) block vector
  // 3. the history point data is updated as function of the temptative solution  
  template <int dim>
  void
  Solid<dim>::impose_nonhomog_DirichletBC()
  { 
    Vector<double> Dirich_BC_solution(dof_handler_ref.n_dofs());
    Dirich_BC_solution.collect_sizes();
    constraints.distribute(Dirich_BC_solution);
    update_qph_incremental(Dirich_BC_solution);
  }
#endif

  
  template <int dim>
  void Solid<dim>::setup_qph()
  {
    std::cout << "    Setting up quadrature point data..." << std::endl;

    {
      triangulation.clear_user_data();
      {
	std::vector<PointHistory<dim> > tmp;
	tmp.swap(quadrature_point_history);
      }

      quadrature_point_history
	.resize(triangulation.n_active_cells() * n_q_points);

      unsigned int history_index = 0;
      for (typename Triangulation<dim>::active_cell_iterator cell =
	     triangulation.begin_active(); cell != triangulation.end();
	   ++cell)
	{
	  cell->set_user_pointer(&quadrature_point_history[history_index]);
	  history_index += n_q_points;
	}

      Assert(history_index == quadrature_point_history.size(),
	     ExcInternalError());
    }

    // Next we setup the initial quadrature
    // point data:
    for (typename Triangulation<dim>::active_cell_iterator cell =
	   triangulation.begin_active(); cell != triangulation.end(); ++cell)
      {
	PointHistory<dim> *lqph =
	  reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
	bool cell_material = cell_is_in_electrode_domain(cell);

	Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
	Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

	for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	  lqph[q_point].setup_lqp(parameters, cell_material);  // the PointHistory class knows how to interptret the II boolean variable passed along
      }
  }

 
  template <int dim>
  void Solid<dim>::update_qph_incremental(const Vector<double> &solution_delta)
  {
    timer.enter_subsection("Update QPH data");
    std::cout << " UQPH " << std::flush;

    const Vector<double> solution_total(get_total_solution(solution_delta));

    const UpdateFlags uf_UQPH(update_values | update_gradients);
    PerTaskData_UQPH per_task_data_UQPH;
    ScratchData_UQPH scratch_data_UQPH(fe, qf_cell, uf_UQPH, solution_total, solution_delta, growth_vector);

    // We then pass them and the one-cell update function to the WorkStream to
    // be processed:
    WorkStream::run(dof_handler_ref.begin_active(),
		    dof_handler_ref.end(),
		    *this,
		    &Solid::update_qph_incremental_one_cell,
		    &Solid::copy_local_to_global_UQPH,
		    scratch_data_UQPH,
		    per_task_data_UQPH);

    timer.leave_subsection();
  }


  // Now we describe how we extract data from the solution vector and pass it
  // along to each QP storage object for processing.
  template <int dim>
  void
  Solid<dim>::update_qph_incremental_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
					      ScratchData_UQPH &scratch,
					      PerTaskData_UQPH &/*data*/)
  {
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());

    Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
    Assert(lqph <= &quadrature_point_history.back(), ExcInternalError());

    Assert(scratch.growth_grads_u_total.size() == n_q_points,
     	   ExcInternalError());

    Assert(scratch.solution_grads_u_total.size() == n_q_points,
	   ExcInternalError());

    scratch.reset();

 
    scratch.fe_values_ref.reinit(cell);
    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.solution_total,
						       scratch.solution_grads_u_total);

    scratch.fe_values_ref[u_fe].get_function_gradients(scratch.growth_vector,
						       scratch.growth_grads_u_total);
   

    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      lqph[q_point].update_values(scratch.solution_grads_u_total[q_point],
				  scratch.growth_grads_u_total[q_point]);
  }


  // @sect4{Solid::solve_nonlinear_timestep}

  // The next function is the driver method for the Newton-Raphson scheme.
  // At its top we create a new vector to store the current Newton update step,
  // reset the error storage objects and print solver header.
  template <int dim>
  void
  Solid<dim>::solve_nonlinear_timestep(Vector<double> &solution_delta) 
  {
    std::cout << std::endl << "Timestep " << time.get_timestep() << " @ "
	      << time.current() << "s" << std::endl;

    Vector<double> newton_update(dof_handler_ref.n_dofs());

    error_residual.reset();
    error_residual_0.reset();
    error_residual_norm.reset();

    error_update.reset();
    error_update_0.reset();
    error_update_norm.reset();

    print_conv_header();


    unsigned int newton_iteration = 0;
    for (; newton_iteration < parameters.max_iterations_NR;
	 ++newton_iteration)
      {
	std::cout << " " << std::setw(2) << newton_iteration << " " << std::flush;

	tangent_matrix = 0.0;
	system_rhs = 0.0;

#if NONHOM_DIRICHLET  // ----------------------------------------------------------ADDed
	if (newton_iteration == 0) 
	  {
	    make_constraints(-1);
	    impose_nonhomog_DirichletBC();
	  }
#endif	// ------------------------------------------------------------------------ADDed

	assemble_system_rhs();

	// system_rhs.block(c_dof).add((1.0/time.get_delta_t()), tmp_vector.block(c_dof));

	// If we don't condense the residual looks that it is converging to a non-null value
#if NONHOM_DIRICHLET     
	constraints.condense(tangent_matrix, system_rhs);
#endif
	get_error_residual(error_residual);

	if (newton_iteration == 0) 
	  {
	    error_residual_0 = error_residual;
	    std::cout << "\n Residual norms :" << std::endl
		      << "Force: \t\t" << error_residual_0.u << std::endl
		      << "Flux: \t\t" <<  error_residual_0.c << std::endl
		      << "Chem. Pot. residual: \t\t" << error_residual_0.mu << "\n" << std::endl;
	  }

	error_residual_norm = error_residual;
	error_residual_norm.normalise(error_residual_0);

	if (newton_iteration > 0 
	    && ( error_update_norm.norm <= parameters.tol_u
		|| error_residual_norm.norm <= parameters.tol_f)
	    )
	  {
	    std::cout << " CONVERGED! " << std::endl;
	    print_conv_footer();
 
	    break;
	  }

	assemble_system_tangent();

	// tangent_matrix.block(mu_dof, mu_dof).copy_from( mass_matrix.block(mu_dof, mu_dof) );
	make_constraints(newton_iteration);
	constraints.condense(tangent_matrix, system_rhs); // eliminate constrained degrees of freedom (non-homogeneous Dirichlet BC)

	const std::pair<unsigned int, double>
	  lin_solver_output = solve_linear_system(newton_update);

	get_error_update(newton_update, error_update);
	if (newton_iteration == 0)
	  error_update_0 = error_update;

	error_update_norm = error_update;
	error_update_norm.normalise(error_update_0);

	solution_delta += newton_update;	
      
	update_qph_incremental(solution_delta);

	solution_current = solution_n;
	solution_current += solution_delta;

	std::cout << " |            " << std::fixed << std::setprecision(3) << std::setw(7)
		  << std::scientific << lin_solver_output.first << "  "
		  << lin_solver_output.second << "  " << error_residual_norm.norm
		  << "  " << error_update_norm.norm
		  << "  " << std::endl;
      }

    AssertThrow (newton_iteration <= parameters.max_iterations_NR,
		 ExcMessage("No convergence in nonlinear solver!"));
  }


  template <int dim>
  void Solid<dim>::print_conv_header()
  {
    static const unsigned int l_width = 155;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "                 SOLVER STEP                  "
	      << " |  LIN_IT   LIN_RES    RES_NORM    "
	      << " NU_NORM     "   << std::endl;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;
  }


  template <int dim>
  void Solid<dim>::print_conv_footer()
  {
    static const unsigned int l_width = 155;

    for (unsigned int i = 0; i < l_width; ++i)
      std::cout << "_";
    std::cout << std::endl;

    std::cout << "Relative errors:" << std::endl
	      << "Displacement:\t" << error_update.u / error_update_0.u << std::endl
	      << "Force: \t\t" << error_residual.u / error_residual_0.u << std::endl;
  }


  template <int dim>
  void Solid<dim>::get_error_residual(Errors &error_residual)
  {
    Vector<double> error_res(dof_handler_ref.n_dofs());

    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
	error_res(i) = system_rhs(i);

    error_residual.norm = error_res.l2_norm();
  }


  template <int dim>
  void Solid<dim>::get_error_update(const Vector<double> &newton_update,
				    Errors &error_update)
  {
    Vector<double> error_ud(dof_handler_ref.n_dofs());
    for (unsigned int i = 0; i < dof_handler_ref.n_dofs(); ++i)
      if (!constraints.is_constrained(i))
	error_ud(i) = newton_update(i);

    error_update.norm = error_ud.l2_norm();
  }

  template <int dim>
  Vector<double>
  Solid<dim>::get_total_solution(const Vector<double> &solution_delta) const
  {
    Vector<double> solution_total(solution_n);
    
    solution_total += solution_delta;
    return solution_total;
  }


  template <int dim>
  void Solid<dim>::assemble_system_tangent()
  {
    timer.enter_subsection("Assemble tangent matrix");
    std::cout << " ASM_K " << std::flush;

    tangent_matrix = 0.0;

    const UpdateFlags uf_cell(update_values    |
			      update_gradients |
			      update_JxW_values);

    PerTaskData_K per_task_data(dofs_per_cell);
    ScratchData_K scratch_data(fe, qf_cell, uf_cell);

    WorkStream::run(dof_handler_ref.begin_active(),
		    dof_handler_ref.end(),
		    *this,
		    &Solid::assemble_system_tangent_one_cell,
		    &Solid::copy_local_to_global_K,
		    scratch_data,
		    per_task_data);

    timer.leave_subsection();
  }

  template <int dim>
  void Solid<dim>::copy_local_to_global_K(const PerTaskData_K &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
	tangent_matrix.add(data.local_dof_indices[i],
			   data.local_dof_indices[j],
			   data.cell_matrix(i, j));
  }

  template <int dim>
  void
  Solid<dim>::assemble_system_tangent_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
					       ScratchData_K &scratch,
					       PerTaskData_K &data)
  {
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);

    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
   
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {		
	for (unsigned int k = 0; k < dofs_per_cell; ++k)
	  {
	    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point); 
	  }
      }


    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
	const Tensor<4, dim> dPK_dF          = lqph[q_point].get_Elast_Tens();

	const std::vector<Tensor<2, dim> >
	  &grad_Nx = scratch.grad_Nx[q_point];
	const double JxW = scratch.fe_values_ref.JxW(q_point);
	
	for (unsigned int i = 0; i < dofs_per_cell; ++i)
	  {
	    for (unsigned int j = 0; j < dofs_per_cell; ++j)
	      {
		data.cell_matrix(i, j) += contract3( grad_Nx[j], dPK_dF, grad_Nx[i] ) * JxW;
	      }
	  }
      }
  }


  template <int dim>
  void Solid<dim>::assemble_system_rhs()
  {
    timer.enter_subsection("Assemble system right-hand side");
    std::cout << " ASM_R " << std::flush;

    system_rhs = 0.0;

    const UpdateFlags uf_cell(update_values |
			      update_gradients |
			      update_JxW_values);
    const UpdateFlags uf_face(update_values |
			      update_normal_vectors |
			      update_JxW_values);

    PerTaskData_RHS per_task_data(dofs_per_cell);
    ScratchData_RHS scratch_data(fe, qf_cell, uf_cell, qf_face, uf_face);

    WorkStream::run(dof_handler_ref.begin_active(),
		    dof_handler_ref.end(),
		    *this,
		    &Solid::assemble_system_rhs_one_cell,
		    &Solid::copy_local_to_global_rhs,
		    scratch_data,
		    per_task_data);

    timer.leave_subsection();
  }



  template <int dim>
  void Solid<dim>::copy_local_to_global_rhs(const PerTaskData_RHS &data)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      system_rhs(data.local_dof_indices[i]) += data.cell_rhs(i);
  }

  template <int dim>
  void
  Solid<dim>::assemble_system_rhs_one_cell(const typename DoFHandler<dim>::active_cell_iterator &cell,
					   ScratchData_RHS &scratch,
					   PerTaskData_RHS &data)
  {
    data.reset();
    scratch.reset();
    scratch.fe_values_ref.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
  
    PointHistory<dim> *lqph =
      reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
    
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
	for (unsigned int k = 0; k < dofs_per_cell; ++k)
	  {
	    scratch.grad_Nx[q_point][k] = scratch.fe_values_ref[u_fe].gradient(k, q_point); 
	  }
      }
  
    for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
      {
	const Tensor<2, dim> PK              = lqph[q_point].get_PKstress();

	// std::cout<< ' PK ' << std::endl;
	// for (unsigned int i = 0; i<dim; ++i)
	//   for (unsigned int j = 0; j<dim; ++j) 
	//     std::cout << PK[i][j] << '\t';
	
	// std::cout<< std::endl;
	
	const std::vector<Tensor<2, dim> >
	  &grad_Nx = scratch.grad_Nx[q_point];
	const double JxW = scratch.fe_values_ref.JxW(q_point);

	// Note, by definition of the rhs as the negative of the residual,
	// these contributions are subtracted.
	for (unsigned int i = 0; i < dofs_per_cell; ++i)
	  {	
	    data.cell_rhs(i) -= double_contract(PK, grad_Nx[i]) * JxW;
	  }
      }

#if NEUMANN_BOUNDARY_COND
    for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      if (cell->face(face)->at_boundary() == true
      	  && cell->face(face)->boundary_id() ==  Xnormal_positive_plane )
      	{
      	  scratch.fe_face_values_ref.reinit(cell, face);

      	  for (unsigned int f_q_point = 0; f_q_point < n_q_points_f; ++f_q_point)
      	    {
    	      const Tensor<1, dim> &N = scratch.fe_face_values_ref.normal_vector(f_q_point);
	      
	      double  time_ramp = (time.current() / 100.0);
	       
	      static const double  p0        = -20.0e6;
	      if ( time.current() > 100.0 )
		time_ramp = 1.0;
	      double         pressure  = p0 * time_ramp;
	      const Tensor<1, dim> traction  = pressure * N;
	      
      	      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      	        {
		  const unsigned int component_i = fe.system_to_component_index(i).first;
		  const double Ni = scratch.fe_face_values_ref.shape_value(i, f_q_point);
		  const double JxW = scratch.fe_face_values_ref.JxW(f_q_point);
		  
		  data.cell_rhs(i) += (Ni * traction[component_i]) * JxW;
      	      	}
      	    }
      	}
#endif 
  }




  // ---------------------------------------------------------------------------


  template <int dim>
  void Solid<dim>::make_constraints(const int &it_nr)
  {
    std::cout << " CST " << std::flush;

    constraints.clear();
    const bool apply_dirichlet_bc = (it_nr == 0);
    const bool clear_dirichlet_bc = (it_nr > 0);
    const bool apply_nonhom_dirichlet_bc = (it_nr == -1);

   
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    const FEValuesExtractors::Scalar z_displacement(2);
    const FEValuesExtractors::Scalar concentration(3);


#if NON_UNIFORM_REFINEMENT
    DoFTools::make_hanging_node_constraints (dof_handler_ref,
					     constraints);
#endif

    {
      const int boundary_id = Xnormal_negative_plane;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
      else if (clear_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
    }
#if !NEUMANN_BOUNDARY_COND
     {
      const int boundary_id = Xnormal_positive_plane;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
      else if (clear_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(x_displacement));
    }
#endif
    {
      const int boundary_id = Ynormal_negative_plane;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
      else if (clear_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
    }
    {
      const int boundary_id = Ynormal_positive_plane;

      if (apply_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
      else if (clear_dirichlet_bc == true)
        VectorTools::interpolate_boundary_values(dof_handler_ref,
                                                 boundary_id,
                                                 ZeroFunction<dim>(n_components),
                                                 constraints,
                                                 fe.component_mask(y_displacement));
    }

    // Set the average Y displacement to zero on the part of the 
    // boundary domain having normal aligned with the Z axis
    // {
    //   std::set< types::boundary_id > selected_boundary_id;
    //   selected_boundary_id.insert(Ynormal_positive_plane);
      
    //   std::vector<bool> boundary_dofs (dof_handler_ref.n_dofs(), false);
    //   DoFTools::extract_boundary_dofs (dof_handler_ref,
    // 				       fe.component_mask(x_displacement),
    // 				       boundary_dofs,
    // 				       selected_boundary_id);

    //   const unsigned int first_boundary_dof
    // 	= std::distance (boundary_dofs.begin(),
    // 			 std::find (boundary_dofs.begin(),
    // 				    boundary_dofs.end(),
    // 				    true));

    //   constraints.add_line (first_boundary_dof);
    //   for (unsigned int i=first_boundary_dof+1; i<boundary_dofs.size(); ++i)
    // 	if (boundary_dofs[i] == true)
    // 	  constraints.add_entry (first_boundary_dof, i, -1);
    // }

  
#if NONHOM_DIRICHLET

#endif
   
    constraints.close();
  }


  // @sect4{Solid::solve_linear_system}
  template <int dim>
  std::pair<unsigned int, double>
  Solid<dim>::solve_linear_system(Vector<double> &newton_update)
  {
    unsigned int lin_it = 0;
    double lin_res = 0.0;
    {
      timer.enter_subsection("Linear solver");
      std::cout << " SLV " << std::flush;

      if (parameters.type_lin == "UMFPACK")
	{
	  // direct solver 

	  SparseDirectUMFPACK A_direct;
	  A_direct.initialize(tangent_matrix);
	  A_direct.vmult(newton_update, system_rhs);

	  lin_it = 1;
	  lin_res = 0.0;
	}
      // else if (parameters.type_lin == "MUMPS")
      // 	{
      // 	  double put_initial_error_here = 1.0;
      // 	  SolverControl cn; //(dof_handler_ref.n_dofs(), 1e-16*put_initial_error_here);
      // 	  PETScWrappers::SparseDirectMUMPS mumps_solver(cn);
      // 	  mumps_solver.solve (tangent_matrix, newton_update, system_rhs);
      // 	}
      else
	Assert (false, ExcMessage("Linear solver type not implemented"));

      timer.leave_subsection();
    }

    // Now that we have the displacement update, distribute the constraints
    // back to the Newton update:
    constraints.distribute(newton_update);

    return std::make_pair(lin_it, lin_res);
  }


  template <int dim>
  void Solid<dim>::output_results() const
  {
    DataOut<dim> data_out;
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim,
				    DataComponentInterpretation::component_is_part_of_vector);

    std::vector<std::string> solution_name(dim, "displacement");

    data_out.attach_dof_handler(dof_handler_ref);
    data_out.add_data_vector(solution_n,
			     solution_name,
			     DataOut<dim>::type_dof_data,
			     data_component_interpretation);

    // ---------------------------------------------------------------------->>>>> 
    Vector<double> Hydrostatic_stress (triangulation.n_active_cells());
    Vector<double> Cauchy_stress_XX (triangulation.n_active_cells());
    Vector<double> Cauchy_stress_YY (triangulation.n_active_cells());
    Vector<double> vonMises_stress (triangulation.n_active_cells());
    // std::vector<types::material_id> material_int (triangulation.n_active_cells());
  
    {
      typename Triangulation<dim>::active_cell_iterator
	cell = triangulation.begin_active(),
	endc = triangulation.end();
      for (unsigned int index=0; cell!=endc; ++cell, ++index)
	{
	  PointHistory<dim> *lqph =
	    reinterpret_cast<PointHistory<dim>*>(cell->user_pointer());
	  
	  Assert(lqph >= &quadrature_point_history.front(), ExcInternalError());
	  Assert(lqph <= &quadrature_point_history.back(), ExcInternalError()); 
	  
	  for (unsigned int q=0; q < qf_cell.size(); ++q)
	    {
	      Hydrostatic_stress (index) += qf_cell.weight(q) * (lqph[q].get_Cauchy_stress()[0][0] +
								 lqph[q].get_Cauchy_stress()[1][1] + 
								 lqph[q].get_Cauchy_stress()[2][2]) / 3;
	      Cauchy_stress_XX (index) += qf_cell.weight(q) * lqph[q].get_Cauchy_stress()[0][0];
	      Cauchy_stress_YY (index) += qf_cell.weight(q) * lqph[q].get_Cauchy_stress()[1][1];

	      vonMises_stress (index) += qf_cell.weight(q) * lqph[q].get_vonMises_stress();
	    }
	  // MATERIAL ID
	  // material_int[cell->index()] = cell->material_id();
	}
    }
    data_out.add_data_vector (Hydrostatic_stress, "Hydrostatic_Cauchy_stress");
    data_out.add_data_vector (Cauchy_stress_XX, "Cauchy_stress_XX");
    data_out.add_data_vector (Cauchy_stress_YY, "Cauchy_stress_YY");
    data_out.add_data_vector (vonMises_stress, "vonMises_stress");
    // MATERIAL ID
    // const Vector<double> materialID(material_int.begin(),
    // 				    material_int.end());
    // data_out.add_data_vector (materialID, "material_ID");
    
    data_out.build_patches(degree);
    
    std::ostringstream filename;
    filename << "Mechanical_solver-" << time.get_timestep() << ".vtu";

    std::ofstream output(filename.str().c_str());
    data_out.write_vtu(output);

  }



  // . . . . . . . . . . . GROWTH MODEL . . . . . . . . . . . .
#if GROWTH

  template <int dim>
  void Solid<dim>::setup_system_gr ()
  {
    dof_handler_gr.distribute_dofs (fe_gr);
    growth_vector.reinit (dof_handler_gr.n_dofs());
    system_rhs_gr.reinit (dof_handler_gr.n_dofs());
    
    
    constraints_gr.clear ();
    
    std::set<types::boundary_id> normal_flux_boundaries;
    double velocity = 0.01 * parameters.scale;
    normal_flux_boundaries.insert (Xnormal_positive_plane);
    
    typename FunctionMap<dim>::type      boundary_functions;
    ConstantFunction<dim>                constant_velocity_bc (velocity,numVariables);
    boundary_functions[0] = &constant_velocity_bc;
    const MappingQ<dim> mapping (parameters.quad_order);
    
    // VectorTools::compute_nonzero_normal_flux_constraints	( dof_handler_gr,
    // 								  0,                       //  first_vector_component,
    // 								  normal_flux_boundaries,
    // 								  boundary_functions,
    // 								  constraints_gr,
    // 								  mapping 
    // 								  );
    
    const FEValuesExtractors::Scalar x_displacement(0);
    const FEValuesExtractors::Scalar y_displacement(1);
    
    {
      VectorTools::interpolate_boundary_values (dof_handler_gr,
						Xnormal_negative_plane,
						ZeroFunction<dim>(numVariables),
						constraints_gr,
						fe_gr.component_mask(x_displacement));
      
      VectorTools::interpolate_boundary_values (dof_handler_gr,
						Ynormal_negative_plane,
						ZeroFunction<dim>(numVariables),
						constraints_gr,
						fe_gr.component_mask(y_displacement));
      
      VectorTools::interpolate_boundary_values (dof_handler_gr,
						Ynormal_positive_plane,
						ZeroFunction<dim>(numVariables),
						constraints_gr,
						fe_gr.component_mask(y_displacement));
    }
    
    {
      VectorTools::interpolate_boundary_values (dof_handler_gr,
    					      Xnormal_positive_plane,
    					      ConstantFunction<dim>(velocity, numVariables),
    					      constraints_gr,
    					      fe_gr.component_mask(x_displacement));
    }
    
    constraints_gr.close ();
    
    
    DynamicSparsityPattern dsp(dof_handler_gr.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_gr,
				    dsp,
				    constraints_gr,
				    /*keep_constrained_dofs = */ true);
    sparsity_pattern_gr.copy_from(dsp);
    system_matrix_gr.reinit (sparsity_pattern_gr);
    
  }
  
  
  template <int dim>
  void Solid<dim>::assemble_system_gr ()
  {
    QGauss<dim>  quadrature_formula(parameters.quad_order);
    
    
    // const RightHandSide<dim> right_hand_side;
    
    FEValues<dim> fe_values (fe_gr, quadrature_formula,
			     update_values   | update_gradients |
			     update_quadrature_points | update_JxW_values);
    
    const unsigned int   dofs_per_cell = fe_gr.dofs_per_cell;
    const unsigned int   n_q_points    = quadrature_formula.size();
    
    FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
    Vector<double>       cell_rhs (dofs_per_cell);
    
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    
    typename DoFHandler<dim>::active_cell_iterator
      cell = dof_handler_gr.begin_active(),
      endc = dof_handler_gr.end();
    
    const double coefficient = 1.e-16;
    
    for (; cell!=endc; ++cell)
      {
	fe_values.reinit (cell);
	cell_matrix = 0;
	cell_rhs = 0;
	
	for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      const unsigned int component_i = fe_gr.system_to_component_index(i).first;	
	      
	      for (unsigned int j=0; j<dofs_per_cell; ++j)
		{
		  // const unsigned int j_group = fe.system_to_base_index(j).first.first;
		  const unsigned int component_j = fe_gr.system_to_component_index(j).first;	
		  
		  if (component_i == component_j)
		    cell_matrix(i,j) += (coefficient *
					 fe_values.shape_grad (i, q_index) *
					 fe_values.shape_grad (j, q_index) *
					 fe_values.JxW (q_index));
		}
	      // cell_rhs(i) += (fe_values.shape_value (i, q_index) *
	      //                right_hand_side.value (fe_values.quadrature_point (q_index)) *
	      //                fe_values.JxW (q_index));
	      
	    }
	
	
	cell->get_dof_indices (local_dof_indices);
	constraints_gr.distribute_local_to_global(cell_matrix, cell_rhs,
						  local_dof_indices,
						  system_matrix_gr, system_rhs_gr);
      }
    
  }
  
  
  template <int dim>
  void Solid<dim>::solve_gr ()
  {
    SolverControl           solver_control (5000, 1e-12);
    SolverCG<>              solver (solver_control);
    solver.solve (system_matrix_gr, growth_vector, system_rhs_gr,
		  PreconditionIdentity());
    
    // We have made one addition, though: since we suppress output from the
    // linear solvers, we have to print the number of iterations by hand.
    std::cout << "   " << solver_control.last_step()
	      << " CG iterations needed to obtain convergence."
	      << std::endl;
    
    constraints_gr.distribute(growth_vector);
  }
  
  
  
  template <int dim>
  void Solid<dim>::output_results_gr () const
  {
    DataOut<dim> data_out;
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(numVariables,
				    DataComponentInterpretation::component_is_part_of_vector);
    
    
    std::vector<std::string> solution_name(numVariables, "displacement");

    data_out.attach_dof_handler(dof_handler_gr);
    data_out.add_data_vector(growth_vector,
			     solution_name,
			     DataOut<dim>::type_dof_data,
			     data_component_interpretation);
  
    data_out.build_patches ();
  
    std::ofstream output (dim == 2 ?
			  "solution-2d.vtk" :
			  "solution-1d.vtk");
    data_out.write_vtk (output);
  }
#endif
}
  
int main (int argc, char *argv[])
{
  using namespace dealii;
  using namespace Step44;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, dealii::numbers::invalid_unsigned_int); 
  // NB: third argument
  // dealii::numbers::invalid_unsigned_int 
  // is used to have the number of processors determined by TBB 

  try
    {
      deallog.depth_console(0);

      Solid<2> solid_2d("parameters.prm");  
      solid_2d.run();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Exception on processing: " << std::endl << exc.what()
		<< std::endl << "Aborting!" << std::endl
		<< "----------------------------------------------------"
		<< std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      std::cerr << "Unknown exception!" << std::endl << "Aborting!"
		<< std::endl
		<< "----------------------------------------------------"
		<< std::endl;
      return 1;
    }
  
  return 0;
}


