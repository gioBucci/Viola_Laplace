#ifndef POINTHISTORY_H
#define POINTHISTORY_H

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include "standard_tensors.h"
#include "Material_NeoHook2D_mech.h"
#include "Parameters.h"
 
template <int dim>
class PointHistory
{
 public:
  PointHistory();
  
  virtual ~PointHistory();
  
  void setup_lqp (const Parameters::AllParameters &parameters, bool electrode_material);

  void update_values (const dealii::Tensor<2, dim> &grad_GR,
		      const dealii::Tensor<2, dim> &grad_u_n);
 
  const double get_vonMises_stress() const;
  const dealii::Tensor<2, dim> &get_F() const;
  const dealii::Tensor<2, dim> &get_PKstress() const;
  const dealii::Tensor<2, 3> &get_Cauchy_stress() const;
  const dealii::Tensor<4, dim> &get_Elast_Tens() const;
  const double &get_det_F() const;
  const dealii::Tensor<2, dim> &get_F_inv_transp() const;
  const dealii::Tensor<2, dim> &get_C_inv() const;

 private:
  Material_NeoHook<dim> *material;

  dealii::Tensor<2, dim> F;
  dealii::Tensor<2, dim> PKstress;
  dealii::Tensor<2, 3> Cauchy_stress;
  double vonMises_stress;
  dealii::Tensor<4, dim> Elast_Tens;
  double det_F;
  dealii::Tensor<2, dim> F_inv_transp;
  dealii::Tensor<2, dim> C_inv;
};


template <int dim>
PointHistory<dim>::PointHistory()
  :
  material(NULL),
  F(StandardTensors<dim>::I),
  PKstress(dealii::Tensor<2, dim>()),
  Cauchy_stress(dealii::Tensor<2, 3>()),
  vonMises_stress (0.0),
  Elast_Tens(dealii::Tensor<4, dim>()),
  det_F(0.0),
  F_inv_transp(StandardTensors<dim>::I),
  C_inv(StandardTensors<dim>::I)
{}

template <int dim>
PointHistory<dim>::~PointHistory()
{
  delete material;
  material = NULL;
}

template <int dim>
void PointHistory<dim>::setup_lqp (const Parameters::AllParameters &parameters, bool electrode_material)
{
  if (electrode_material)
    material = new Material_NeoHook<dim>(parameters.E0,
					 parameters.nu);
  else
    material = new Material_NeoHook<dim>(parameters.E0_el,
					 parameters.nu_el);

  update_values(dealii::Tensor<2, dim>(),dealii::Tensor<2, dim>());
}


template <int dim>
void PointHistory<dim>::update_values ( const dealii::Tensor<2, dim> &grad_u_n,
					const dealii::Tensor<2, dim> &grad_GR )
{
   const dealii::Tensor<2, 3> F
    = (StandardTensors<3>::I +
       StandardTensors<dim>::extend_dim(grad_u_n));
     const dealii::Tensor<2, 3> F_gr
    = (StandardTensors<3>::I +
       StandardTensors<dim>::extend_dim(grad_GR));
     material->update_material_data(F, F_gr );
     
  PKstress = StandardTensors<dim>::reduce_dim(material->get_PKstress());
  Cauchy_stress = material->get_Cauchy_stress();
  vonMises_stress = material->get_vonMises_stress();
  Elast_Tens = StandardTensors<dim>::reduceIV_dim(material->get_Elast_Tens());
  det_F = material->get_det_F();
  F_inv_transp = StandardTensors<dim>::reduce_dim(material->get_F_inv_transp());
  C_inv = StandardTensors<dim>::reduce_dim(material->get_C_inv());
}
    

template <int dim>
const dealii::Tensor<2, dim> &PointHistory<dim>::get_F() const
{
  return F;
}

template <int dim>
const dealii::Tensor<2, dim> &PointHistory<dim>::get_PKstress() const
{
  return PKstress;
}

template <int dim>
const dealii::Tensor<2, 3> &PointHistory<dim>::get_Cauchy_stress() const
{
  return Cauchy_stress;
}

template <int dim>
const double PointHistory<dim>::get_vonMises_stress() const
{
  return vonMises_stress;
}

template <int dim>
const dealii::Tensor<4, dim> &PointHistory<dim>::get_Elast_Tens() const
{
  return Elast_Tens;
}

template <int dim>
const dealii::Tensor<2, dim> &PointHistory<dim>::get_F_inv_transp() const
{
  return F_inv_transp;
} 

template <int dim>
const dealii::Tensor<2, dim> &PointHistory<dim>::get_C_inv() const
{
  return C_inv;
} 

template <int dim>
const double &PointHistory<dim>::get_det_F() const
{
  return det_F;
}

#endif
