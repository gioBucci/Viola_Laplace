#ifndef STANDARD_TENSORS_H
#define STANDARD_TENSORS_H

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

template <int dim>
class StandardTensors
{
 public:
    
  static const dealii::Tensor<2, dim> I;
  /* static const dealii::SymmetricTensor<4, dim> IxI; */
  /* static const dealii::SymmetricTensor<4, dim> II; */
  /* static const dealii::SymmetricTensor<4, dim> dev_P; */

  static dealii::Tensor<2, dim> diagonalTensor(const double diag_src[]);
  static dealii::Tensor<4,dim> IdentityIV ();
  static dealii::Tensor<4,dim> outer_productIV(const dealii::Tensor<2,dim> &src1, const dealii::Tensor<2,dim> &src2);
  static double double_contract_2_2 (const dealii::Tensor<2,dim> &src1, const dealii::Tensor<2,dim> &src2); 
  static dealii::Tensor<2,dim> double_contract_4_2 (const dealii::Tensor<4,dim> &src1, const dealii::Tensor<2,dim> &src2);
  static dealii::Tensor<2,dim> double_contract_2_4 (const dealii::Tensor<2,dim> &src1, const dealii::Tensor<4,dim> &src2);
  static dealii::Tensor<4,dim> tensor_product_2_4_2 (const dealii::Tensor<2,dim> &src1, const dealii::Tensor<4,dim> &src2, const dealii::Tensor<2,dim> &src3);
  static dealii::Tensor<4,dim> mytensor_product(const dealii::Tensor<2,dim> &src1, const dealii::Tensor<2,dim> &src2);
  static dealii::Tensor<1,dim> extract_Tangent (const dealii::Tensor<1,dim> &src1);
  static dealii::Tensor<1,dim> cross_product (const dealii::Tensor<1,dim> &src1, const dealii::Tensor<1,dim> &src2);
  static dealii::Tensor<2,dim> outer_product (const dealii::Tensor<1,dim> &src1, const dealii::Tensor<1,dim> &src2);
  static dealii::Tensor<2,3> extend_dim (const dealii::Tensor<2,dim> &src);
  static dealii::Tensor<2,dim> reduce_dim (const dealii::Tensor<2,3> &src);
  static dealii::Tensor<4,dim> reduceIV_dim (const dealii::Tensor<4,3> &src);

  static double trace (const dealii::Tensor<2,3> &src);
  static double scalar_product (const dealii::Tensor<2,3> &src1, const dealii::Tensor<2,3> &src2);
};


// @sect3{Some standard tensors}
// Now we define some frequently used second and fourth-order tensors and tensor operators:
  
template <int dim>
const dealii::Tensor<2, dim>
  StandardTensors<dim>::I = dealii::unit_symmetric_tensor<dim>();
  
// template <int dim>
// const dealii::SymmetricTensor<4, dim>
// StandardTensors<dim>::IxI = dealii::outer_product(I, I);
  
// template <int dim>
// const dealii::SymmetricTensor<4, dim>
// StandardTensors<dim>::II = dealii::identity_tensor<dim>();
  
// template <int dim>
// const dealii::SymmetricTensor<4, dim>
// StandardTensors<dim>::dev_P = dealii::deviator_tensor<dim>();

template <int dim>
dealii::Tensor<2,dim>
  StandardTensors<dim>::diagonalTensor (const double diag_src[])
{
  dealii::Tensor<2,dim> dst;
  for (unsigned int i=1; i<dim; ++i)
    dst[i][i] = diag_src[i];
  return dst;
}

template <int dim>
dealii::Tensor<4,dim>
  StandardTensors<dim>::IdentityIV ()
{
  dealii::Tensor<4,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  if ( i==k && j==l)
	    dst[i][j][k][l] = 1.0;
	  else 
	    dst[i][j][k][l] = 0.0;
  return dst;
}

template <int dim>
dealii::Tensor<4,dim>
  StandardTensors<dim>::outer_productIV (const dealii::Tensor<2,dim> &src1,
					 const dealii::Tensor<2,dim> &src2)
{
  dealii::Tensor<4,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  dst[i][j][k][l] = src1[i][j] * src2[k][l];
  return dst;
}

template <int dim>
double
StandardTensors<dim>::double_contract_2_2 (const dealii::Tensor<2,dim> &src1,
					   const dealii::Tensor<2,dim> &src2)
{
  double dst = 0.0;
  
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
	  dst += src1[i][j] * src2[i][j];
  return dst;
}

template <int dim>
dealii::Tensor<2,dim>
  StandardTensors<dim>::double_contract_4_2 (const dealii::Tensor<4,dim> &src1,
					     const dealii::Tensor<2,dim> &src2)
{
  dealii::Tensor<2,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  dst[i][j] += src1[i][j][k][l] * src2[k][l];
  return dst;
}

template <int dim>
dealii::Tensor<2,dim>
  StandardTensors<dim>::double_contract_2_4 (const dealii::Tensor<2,dim> &src1,
					     const dealii::Tensor<4,dim> &src2)
{
  dealii::Tensor<2,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  dst[i][j] += src1[k][l] * src2[k][l][i][j];
  return dst;
}

template <int dim>
dealii::Tensor<4,dim>
  StandardTensors<dim>::tensor_product_2_4_2 (const dealii::Tensor<2,dim> &src1,
					      const dealii::Tensor<4,dim> &src2,
					      const dealii::Tensor<2,dim> &src3)
{
  dealii::Tensor<4,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  for (unsigned int h=0; h<dim; ++h)
	    for (unsigned int m=0; m<dim; ++m)
	      dst[i][j][k][l] += src1[i][h] * src2[h][j][k][m] * src3[m][l];
  return dst;
}

template <int dim>
dealii::Tensor<4,dim>
  StandardTensors<dim>::mytensor_product (const dealii::Tensor<2,dim> &src1,
					  const dealii::Tensor<2,dim> &src2)
{
  dealii::Tensor<4,dim> dst;

  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
	  dst[i][j][k][l] = src1[l][i] * src2[j][k];
  return dst;
}

template <int dim>
dealii::Tensor<1,dim>
  StandardTensors<dim>::extract_Tangent (const dealii::Tensor<1,dim> &src1)
{
  Assert (dim>1, dealii::ExcInternalError());

  dealii::Tensor<1,dim> dst;
  dealii::Tensor<1,dim> e1;
  dealii::Tensor<1,dim> e2;
  e1[0] = 1.0;
  e2[1] = 1.0;
 
  if(dim==3)
    {
      dst = e1 - contract(src1, e1) * src1;
      if (dst.norm() < 1.0e-8)
	dst = e2 - contract(src1, e2) * src1;
      
      dst /= dst.norm();
    }
  else if(dim==2)
    {
      dst = - contract(src1,e2)*e1 + contract(src1,e1)*e2;
    }

  return dst;
}

template <int dim>
dealii::Tensor<1,dim>
  StandardTensors<dim>::cross_product (const dealii::Tensor<1,dim> &src1,
				       const dealii::Tensor<1,dim> &src2)
{
  Assert (dim==3, dealii::ExcInternalError());
  
  dealii::Tensor<1,dim> dst;
  dst[0] = src1[1]*src2[2] - src1[2]*src2[1];
  dst[1] = src1[2]*src2[0] - src1[0]*src2[2];
  dst[2] = src1[0]*src2[1] - src1[1]*src2[0];
  
  return dst;
}

template <int dim>
dealii::Tensor<2,dim>
  StandardTensors<dim>::outer_product (const dealii::Tensor<1,dim> &src1,
				       const dealii::Tensor<1,dim> &src2)
{
  dealii::Tensor<2,dim> dst;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      dst[i][j] = src1[i] * src2[j];
  
  return dst;
}

template <int dim>
dealii::Tensor<2,3>
  StandardTensors<dim>::extend_dim (const dealii::Tensor<2,dim> &src)
{
  dealii::Tensor<2,3> dst;

      for (unsigned int i=0; i<dim; ++i)
	for (unsigned int j=0; j<dim; ++j)
	  dst[i][j] = src[i][j];

      /* if (dim==2) */
      /* 	dst[2][2] = 1.0; */

  return dst;
}

template <int dim>
dealii::Tensor<2,dim>
  StandardTensors<dim>::reduce_dim (const dealii::Tensor<2,3> &src)
{
  dealii::Tensor<2,dim> dst;
  
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      dst[i][j] = src[i][j];
  
  return dst;
}

template <int dim>
dealii::Tensor<4,dim>
  StandardTensors<dim>::reduceIV_dim (const dealii::Tensor<4,3> &src)
{
  dealii::Tensor<4,dim> dst;
  
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
	for (unsigned int l=0; l<dim; ++l)
      dst[i][j][k][l] = src[i][j][k][l];
  
  return dst;
}

template <int dim>
double 
StandardTensors<dim>::trace (const dealii::Tensor<2,3> &src)
{
  double dst = src[0][0];
  for (unsigned int i=1; i<dim; ++i)
    dst += src[i][i];
  return dst;
}

template <int dim>
double
StandardTensors<dim>::scalar_product (const dealii::Tensor<2,3> &src1,
				      const dealii::Tensor<2,3> &src2)
{
  double dst = 0.0;
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      dst += src1[i][j] * src2[i][j];
  return dst;
}

#endif

