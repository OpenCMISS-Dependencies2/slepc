/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      SLEPc - Scalable Library for Eigenvalue Problem Computations
      Copyright (c) 2002-2007, Universidad Politecnica de Valencia, Spain

      This file is part of SLEPc. See the README file for conditions of use
      and additional information.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include "slepc.h" /*I "slepc.h" I*/
#include <stdlib.h>

#undef __FUNCT__  
#define __FUNCT__ "SlepcVecSetRandom"
/*@
   SlepcVecSetRandom - Sets all components of a vector to random numbers which
   follow a uniform distribution in [0,1).

   Collective on Vec

   Input/Output Parameter:
.  x  - the vector

   Note:
   This operation is equivalent to VecSetRandom - the difference is that the
   vector generated by SlepcVecSetRandom is the same irrespective of the size
   of the communicator.

   Level: developer

.seealso: VecSetRandom()
@*/
PetscErrorCode SlepcVecSetRandom(Vec x)
{
  PetscErrorCode ierr;
  PetscInt       i,n,low,high;
  PetscScalar    *px,t;
#if defined(PETSC_HAVE_DRAND48)
  static unsigned short seed[3] = { 1, 3, 2 };
#endif
  
  PetscFunctionBegin;
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(x,&low,&high);CHKERRQ(ierr);
  ierr = VecGetArray(x,&px);CHKERRQ(ierr);
  for (i=0;i<n;i++) {
#if defined(PETSC_HAVE_DRAND48)
    t = erand48(seed);
#elif defined(PETSC_HAVE_RAND)
    t = rand()/(double)((unsigned int)RAND_MAX+1);
#else
    t = 0.5;
#endif
    if (i>=low && i<high) px[i-low] = t;
  }
  ierr = VecRestoreArray(x,&px);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcIsHermitian"
/*@
   SlepcIsHermitian - Checks if a matrix is Hermitian or not.

   Collective on Mat

   Input parameter:
.  A  - the matrix

   Output parameter:
.  is  - flag indicating if the matrix is Hermitian

   Notes: 
   The result of Ax and A^Hx (with a random x) is compared, but they 
   could be equal also for some non-Hermitian matrices.

   This routine will not work with matrix formats MATSEQSBAIJ or MATMPISBAIJ,
   or when PETSc is configured with complex scalars.
   
   Level: developer

@*/
PetscErrorCode SlepcIsHermitian(Mat A,PetscTruth *is)
{
  PetscErrorCode ierr;
  PetscInt       M,N,m,n;
  Vec            x,w1,w2;
  MPI_Comm       comm;
  PetscReal      norm;
  PetscTruth     has;

  PetscFunctionBegin;

#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
  ierr = PetscTypeCompare((PetscObject)A,MATMPISBAIJ,is);CHKERRQ(ierr);
  if (*is) PetscFunctionReturn(0);
#endif

  *is = PETSC_FALSE;
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  if (M!=N) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);
  ierr = MatHasOperation(A,MATOP_MULT_TRANSPOSE,&has);CHKERRQ(ierr);
  if (!has) PetscFunctionReturn(0);

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = SlepcVecSetRandom(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w1);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&w2);CHKERRQ(ierr);
  ierr = MatMult(A,x,w1);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,x,w2);CHKERRQ(ierr);
  ierr = VecConjugate(w2);CHKERRQ(ierr);
  ierr = VecAXPY(w2,-1.0,w1);CHKERRQ(ierr);
  ierr = VecNorm(w2,NORM_2,&norm);CHKERRQ(ierr);
  if (norm<1.0e-6) *is = PETSC_TRUE;
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(w1);CHKERRQ(ierr);
  ierr = VecDestroy(w2);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)

#undef __FUNCT__  
#define __FUNCT__ "SlepcAbsEigenvalue"
/*@C
   SlepcAbsEigenvalue - Returns the absolute value of a complex number given
   its real and imaginary parts.

   Not collective

   Input parameters:
+  x  - the real part of the complex number
-  y  - the imaginary part of the complex number

   Notes: 
   This function computes sqrt(x**2+y**2), taking care not to cause unnecessary
   overflow. It is based on LAPACK's DLAPY2.

   Level: developer

@*/
PetscReal SlepcAbsEigenvalue(PetscScalar x,PetscScalar y)
{
  PetscReal xabs,yabs,w,z,t;
  PetscFunctionBegin;
  xabs = PetscAbsReal(x);
  yabs = PetscAbsReal(y);
  w = PetscMax(xabs,yabs);
  z = PetscMin(xabs,yabs);
  if (z == 0.0) PetscFunctionReturn(w);
  t = z/w;
  PetscFunctionReturn(w*sqrt(1.0+t*t));  
}

#endif

#undef __FUNCT__  
#define __FUNCT__ "SlepcMatConvertSeqDense"
/*@C
   SlepcMatConvertSeqDense - Converts a parallel matrix to another one in sequential 
   dense format replicating the values in every processor.

   Collective

   Input parameters:
+  A  - the source matrix
-  B  - the target matrix

   Level: developer
   
@*/
PetscErrorCode SlepcMatConvertSeqDense(Mat mat,Mat *newmat)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  PetscMPIInt    size;
  MPI_Comm       comm;
  Mat            *M;
  IS             isrow, iscol;
  PetscTruth     flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(newmat,2);

  ierr = PetscObjectGetComm((PetscObject)mat,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  if (size > 1) {
    /* assemble full matrix on every processor */
    ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,m,0,1,&isrow);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,n,0,1,&iscol);CHKERRQ(ierr);
    ierr = MatGetSubMatrices(mat,1,&isrow,&iscol,MAT_INITIAL_MATRIX,&M);CHKERRQ(ierr);
    ierr = ISDestroy(isrow);CHKERRQ(ierr);
    ierr = ISDestroy(iscol);CHKERRQ(ierr);

    /* Fake support for "inplace" convert */
    if (*newmat == mat) {
      ierr = MatDestroy(mat);CHKERRQ(ierr);
    }
    *newmat = *M;
    ierr = PetscFree(M);CHKERRQ(ierr);     
  
    /* convert matrix to MatSeqDense */
    ierr = PetscTypeCompare((PetscObject)*newmat,MATSEQDENSE,&flg); CHKERRQ(ierr);
    if (!flg) {
      ierr = MatConvert(*newmat,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
    } 
  } else {
    /* convert matrix to MatSeqDense */
    ierr = MatConvert(mat,MATSEQDENSE,MAT_INITIAL_MATRIX,newmat);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "SlepcCheckOrthogonality"
/*@
   SlepcCheckOrthogonality - Checks (or prints) the level of orthogonality
   of a set of vectors.

   Collective on Vec

   Input parameters:
+  V  - a set of vectors
.  nv - number of V vectors
.  W  - an alternative set of vectors (optional)
.  nw - number of W vectors
-  B  - matrix defining the inner product (optional)

   Output parameter:
.  lev - level of orthogonality (optional)

   Notes: 
   This function computes W'*V and prints the result. It is intended to check
   the level of bi-orthogonality of the vectors in the two sets. If W is equal
   to PETSC_NULL then V is used, thus checking the orthogonality of the V vectors.

   If matrix B is provided then the check uses the B-inner product, W'*B*V.

   If lev is not PETSC_NULL, it will contain the level of orthogonality
   computed as ||W'*V - I|| in the Frobenius norm. Otherwise, the matrix W'*V
   is printed.

   Level: developer

@*/
PetscErrorCode SlepcCheckOrthogonality(Vec *V,PetscInt nv,Vec *W,PetscInt nw,Mat B,PetscScalar *lev)
{
  PetscErrorCode ierr;
  int            i,j;
  PetscScalar    *vals;
  Vec            w;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (nv<=0 || nw<=0) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)V[0],&comm);CHKERRQ(ierr);
  ierr = PetscMalloc(nv*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
  if (B) { ierr = VecDuplicate(V[0],&w);CHKERRQ(ierr); }
  if (lev) *lev = 0.0;
  for (i=0;i<nw;i++) {
    if (B) {
      if (W) { ierr = MatMultTranspose(B,W[i],w);CHKERRQ(ierr); }
      else { ierr = MatMultTranspose(B,V[i],w);CHKERRQ(ierr); }
    }
    else {
      if (W) w = W[i];
      else w = V[i];
    }
    ierr = VecMDot(w,nv,V,vals);CHKERRQ(ierr);
    for (j=0;j<nv;j++) {
      if (lev) *lev += (j==i)? (vals[j]-1.0)*(vals[j]-1.0): vals[j]*vals[j];
      else { 
#ifndef PETSC_USE_COMPLEX
        ierr = PetscPrintf(comm," %12g  ",vals[j]);CHKERRQ(ierr); 
#else
        ierr = PetscPrintf(comm," %12g%+12gi ",PetscRealPart(vals[j]),PetscImaginaryPart(vals[j]));CHKERRQ(ierr);     
#endif
      }
    }
    if (!lev) { ierr = PetscPrintf(comm,"\n");CHKERRQ(ierr); }
  }
  ierr = PetscFree(vals);CHKERRQ(ierr);
  if (B) { ierr = VecDestroy(w);CHKERRQ(ierr); }
  if (lev) *lev = PetscSqrtScalar(*lev);
  PetscFunctionReturn(0);
}


