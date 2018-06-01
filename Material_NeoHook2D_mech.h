#ifndef MATERIAL_NEOHOOK_H
#define MATERIAL_NEOHOOK_H

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include "standard_tensors.h"
#include "Parameters.h"

template <int dim>
class Material_NeoHook
{
 public:
  Material_NeoHook(const double E0,
		   const double nu);
  
  void update_material_data(const dealii::Tensor<2, 3> &F,
			    const dealii::Tensor<2, 3> &F_growth);
	
  
  dealii::Tensor<2, 3> get_PKstress() const;
  dealii::Tensor<2, 3> get_Cauchy_stress() const;
  double get_vonMises_stress() const;
  dealii::Tensor<4, 3> get_Elast_Tens_el() const;
  dealii::Tensor<4, 3> get_Elast_Tens() const;
  dealii::Tensor<2, 3> get_F_inv_transp() const;
  dealii::Tensor<2, 3> get_C_inv() const;
  double get_D_RT() const;
  double get_det_F() const;
  
  protected:
    double E;
    double mmu;
    double kappa;
    double lambda;
    const double E0;
    double nu;
    
    double det_F;
    double logJ_e;
    double det_F_el;
    double dFa_1T_dc;
    dealii::Tensor<2, 3> F_inv_transp;
    dealii::Tensor<2, 3> F_el;
    dealii::Tensor<2, 3> F_an;
    dealii::Tensor<2, 3> F_el_inv;
    dealii::Tensor<2, 3> F_an_inv;
    dealii::Tensor<2, 3> C_inv;
};


// @sect3{Compressible neo-Hookean material}


template <int dim>
Material_NeoHook<dim>::Material_NeoHook(const double E0,
					const double nu)
  :
  E0(E0),      
  nu(nu)
{
  Assert(E0 > 0, dealii::ExcInternalError());
}

// ~Material_NeoHook()
// {}

// We update the material model with various deformation dependent data
// based on $F$ and the Li concentration and the chemical potential.
template <int dim>
void Material_NeoHook<dim>::update_material_data(const dealii::Tensor<2, 3> &F,
						 const dealii::Tensor<2, 3> &F_growth)
						 
{
  E = E0 ;
  mmu = E/(2*(1+nu));
  kappa = (2.0 * mmu * (1.0 + nu)) / (3.0 * (1.0 - 2.0 * nu));  
  lambda = nu * E / ((1+nu) * (1-2*nu));    
  det_F = determinant(F);
  F_inv_transp = transpose(invert(F));
  F_an = F_growth;
  F_an_inv = invert(F_an) * invert(F_growth);
  F_el = F * F_an_inv;
  det_F_el = determinant(F_el);
  logJ_e = log(det_F_el);
  F_el_inv = invert(F_el); 
  C_inv = invert(transpose(F) * F);
  
  
  Assert(det_F > 0, dealii::ExcInternalError());
}
 
// This function determines the 1st Piola Kirchhoff stress
template <int dim>
dealii::Tensor<2, 3> Material_NeoHook<dim>::get_PKstress() const
{
  return ((lambda*logJ_e - mmu) * transpose(F_el_inv) + mmu*F_el) * transpose(F_an_inv); 
}

// This function determines the Cauchy stress

template <int dim>
dealii::Tensor<2, 3> Material_NeoHook<dim>::get_Cauchy_stress() const
{
  return (1/det_F_el) * get_PKstress() * transpose(F_el);   // check this
}

template <int dim>
double Material_NeoHook<dim>::get_vonMises_stress() const
{
  dealii::Tensor<2, 3> deviat_stress =  get_Cauchy_stress() - (1.0/3.0) * StandardTensors<dim>::trace( get_Cauchy_stress() ) * StandardTensors<3>::I;
  return std::sqrt( 1.5 * StandardTensors<dim>::scalar_product (deviat_stress,  deviat_stress) );
}

// The fourth-order elasticity tensor $ \frac{\dee \Psi}{\dee F^e \dee F^e} $
template <int dim>
dealii::Tensor<4, 3> Material_NeoHook<dim>::get_Elast_Tens_el() const
{
  dealii::Tensor<4, 3> A = StandardTensors<3>::outer_productIV(transpose(F_el_inv), transpose(F_el_inv));
  A *= lambda;
  dealii::Tensor<4, 3> B = StandardTensors<3>::mytensor_product(F_el_inv, F_el_inv);
  B *= - (lambda*logJ_e - mmu);
  dealii::Tensor<4, 3> C = StandardTensors<3>::IdentityIV();
  C *= mmu;
  
  return A + B + C;
}

// The fourth-order elasticity tensor in the reference setting $ \frac{\dee \Psi}{\dee F \dee F} $
template <int dim>
dealii::Tensor<4, 3> Material_NeoHook<dim>::get_Elast_Tens() const
{
  return StandardTensors<3>::tensor_product_2_4_2( F_an_inv, get_Elast_Tens_el(), transpose(F_an_inv) ); 
}

template <int dim>
dealii::Tensor<2, 3> Material_NeoHook<dim>::get_F_inv_transp() const
{
  return F_inv_transp;
}

template <int dim>
dealii::Tensor<2, 3> Material_NeoHook<dim>::get_C_inv() const
{
  return C_inv;
}

template <int dim>  
double Material_NeoHook<dim>::get_det_F() const
{
 return det_F;
}

#endif
