/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Square root function  sqrt(x)
*/

#include <slepc/private/fnimpl.h>      /*I "slepcfn.h" I*/
#include <slepcblaslapack.h>

PetscErrorCode FNEvaluateFunction_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(x>=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Function not defined in the requested value");
#endif
  *y = PetscSqrtScalar(x);
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateDerivative_Sqrt(FN fn,PetscScalar x,PetscScalar *y)
{
  PetscFunctionBegin;
  PetscCheck(x!=0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#if !defined(PETSC_USE_COMPLEX)
  PetscCheck(x>0.0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Derivative not defined in the requested value");
#endif
  *y = 1.0/(2.0*PetscSqrtScalar(x));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Schur(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&T));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmSchur(fn,n,T,n,PETSC_FALSE));
  PetscCall(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMatVec_Sqrt_Schur(FN fn,Mat A,Vec v)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;
  Mat            B;

  PetscFunctionBegin;
  PetscCall(FN_AllocateWorkMat(fn,A,&B));
  PetscCall(MatDenseGetArray(B,&T));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmSchur(fn,n,T,n,PETSC_TRUE));
  PetscCall(MatDenseRestoreArray(B,&T));
  PetscCall(MatGetColumnVector(B,v,0));
  PetscCall(FN_FreeWorkMat(fn,&B));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&T));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmDenmanBeavers(fn,n,T,n,PETSC_FALSE));
  PetscCall(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&Ba));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmNewtonSchulz(fn,n,Ba,n,PETSC_FALSE));
  PetscCall(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#define MAXIT 50

/*
   Computes the principal square root of the matrix A using the
   Sadeghi iteration. A is overwritten with sqrtm(A).
 */
PetscErrorCode FNSqrtmSadeghi(FN fn,PetscBLASInt n,PetscScalar *A,PetscBLASInt ld)
{
  PetscScalar    *M,*M2,*G,*X=A,*work,work1,sqrtnrm;
  PetscScalar    szero=0.0,sone=1.0,smfive=-5.0,s1d16=1.0/16.0;
  PetscReal      tol,Mres=0.0,nrm,rwork[1],done=1.0;
  PetscInt       i,it;
  PetscBLASInt   N,*piv=NULL,info,lwork=0,query=-1,one=1,zero=0;
  PetscBool      converged=PETSC_FALSE;
  unsigned int   ftz;

  PetscFunctionBegin;
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;
  PetscCall(SlepcSetFlushToZero(&ftz));

  /* query work size */
  PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,A,&ld,piv,&work1,&query,&info));
  PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(work1),&lwork));

  PetscCall(PetscMalloc5(N,&M,N,&M2,N,&G,lwork,&work,n,&piv));
  PetscCall(PetscArraycpy(M,A,N));

  /* scale M */
  nrm = LAPACKlange_("fro",&n,&n,M,&n,rwork);
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    PetscStackCallBLAS("LAPACKlascl",LAPACKlascl_("G",&zero,&zero,&nrm,&done,&N,&one,M,&N,&info));
    SlepcCheckLapackInfo("lascl",info);
    tol *= nrm;
  }
  PetscCall(PetscInfo(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol));

  /* X = I */
  PetscCall(PetscArrayzero(X,N));
  for (i=0;i<n;i++) X[i+i*ld] = 1.0;

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M,&ld,M,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&smfive,M,&one,M2,&one));
    for (i=0;i<n;i++) M2[i+i*ld] += 15.0;
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&s1d16,M,&ld,M2,&ld,&szero,G,&ld));
    for (i=0;i<n;i++) G[i+i*ld] += 5.0/16.0;

    /* X = X*G */
    PetscCall(PetscArraycpy(M2,X,N));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,M2,&ld,G,&ld,&szero,X,&ld));

    /* M = M*inv(G*G) */
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,G,&ld,&szero,M2,&ld));
    PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&n,&n,M2,&ld,piv,&info));
    SlepcCheckLapackInfo("getrf",info);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,M2,&ld,piv,work,&lwork,&info));
    SlepcCheckLapackInfo("getri",info);

    PetscCall(PetscArraycpy(G,M,N));
    PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&n,&n,&n,&sone,G,&ld,M2,&ld,&szero,M,&ld));

    /* check ||I-M|| */
    PetscCall(PetscArraycpy(M2,M,N));
    for (i=0;i<n;i++) M2[i+i*ld] -= 1.0;
    Mres = LAPACKlange_("fro",&n,&n,M2,&n,rwork);
    PetscCheck(!PetscIsNanReal(Mres),PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    PetscCall(PetscInfo(fn,"it: %" PetscInt_FMT " res: %g\n",it,(double)Mres));
    PetscCall(PetscLogFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n));
  }

  PetscCheck(Mres<=tol,PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations",MAXIT);

  /* undo scaling */
  if (nrm>1.0) PetscStackCallBLAS("BLASscal",BLASscal_(&N,&sqrtnrm,A,&one));

  PetscCall(PetscFree5(M,M2,G,work,piv));
  PetscCall(SlepcResetFlushToZero(&ftz));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
#include "../src/sys/classes/fn/impls/cuda/fnutilcuda.h"
#include <slepccublas.h>

#if defined(PETSC_HAVE_MAGMA)
#include <slepcmagma.h>

/*
 * Matrix square root by Sadeghi iteration. CUDA version.
 * Computes the principal square root of the matrix T using the
 * Sadeghi iteration. T is overwritten with sqrtm(T).
 */
PetscErrorCode FNSqrtmSadeghi_CUDAm(FN fn,PetscBLASInt n,PetscScalar *A,PetscBLASInt ld)
{
  PetscScalar        *d_X,*d_M,*d_M2,*d_G,*d_work,alpha;
  const PetscScalar  szero=0.0,sone=1.0,smfive=-5.0,s15=15.0,s1d16=1.0/16.0;
  PetscReal          tol,Mres=0.0,nrm,sqrtnrm;
  PetscInt           it,nb,lwork;
  PetscBLASInt       *piv,N;
  const PetscBLASInt one=1,zero=0;
  PetscBool          converged=PETSC_FALSE;
  cublasHandle_t     cublasv2handle;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA)); /* For CUDA event timers */
  PetscCall(PetscCUBLASGetHandle(&cublasv2handle));
  magma_init();
  N = n*n;
  tol = PetscSqrtReal((PetscReal)n)*PETSC_MACHINE_EPSILON/2;

  PetscCall(PetscMalloc1(n,&piv));
  PetscCallCUDA(cudaMalloc((void **)&d_X,sizeof(PetscScalar)*N));
  PetscCallCUDA(cudaMalloc((void **)&d_M,sizeof(PetscScalar)*N));
  PetscCallCUDA(cudaMalloc((void **)&d_M2,sizeof(PetscScalar)*N));
  PetscCallCUDA(cudaMalloc((void **)&d_G,sizeof(PetscScalar)*N));

  nb = magma_get_xgetri_nb(n);
  lwork = nb*n;
  PetscCallCUDA(cudaMalloc((void **)&d_work,sizeof(PetscScalar)*lwork));
  PetscCall(PetscLogGpuTimeBegin());

  /* M = A */
  PetscCallCUDA(cudaMemcpy(d_M,A,sizeof(PetscScalar)*N,cudaMemcpyHostToDevice));

  /* scale M */
  PetscCallCUBLAS(cublasXnrm2(cublasv2handle,N,d_M,one,&nrm));
  if (nrm>1.0) {
    sqrtnrm = PetscSqrtReal(nrm);
    alpha = 1.0/nrm;
    PetscCallCUBLAS(cublasXscal(cublasv2handle,N,&alpha,d_M,one));
    tol *= nrm;
  }
  PetscCall(PetscInfo(fn,"||A||_F = %g, new tol: %g\n",(double)nrm,(double)tol));

  /* X = I */
  PetscCallCUDA(cudaMemset(d_X,zero,sizeof(PetscScalar)*N));
  PetscCall(set_diagonal(n,d_X,ld,sone));

  for (it=0;it<MAXIT && !converged;it++) {

    /* G = (5/16)*I + (1/16)*M*(15*I-5*M+M*M) */
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M,ld,d_M,ld,&szero,d_M2,ld));
    PetscCallCUBLAS(cublasXaxpy(cublasv2handle,N,&smfive,d_M,one,d_M2,one));
    PetscCall(shift_diagonal(n,d_M2,ld,s15));
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&s1d16,d_M,ld,d_M2,ld,&szero,d_G,ld));
    PetscCall(shift_diagonal(n,d_G,ld,5.0/16.0));

    /* X = X*G */
    PetscCallCUDA(cudaMemcpy(d_M2,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_M2,ld,d_G,ld,&szero,d_X,ld));

    /* M = M*inv(G*G) */
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_G,ld,&szero,d_M2,ld));
    /* magma */
    PetscCallMAGMA(magma_xgetrf_gpu,n,n,d_M2,ld,piv);
    PetscCallMAGMA(magma_xgetri_gpu,n,d_M2,ld,piv,d_work,lwork);
    /* magma */
    PetscCallCUDA(cudaMemcpy(d_G,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    PetscCallCUBLAS(cublasXgemm(cublasv2handle,CUBLAS_OP_N,CUBLAS_OP_N,n,n,n,&sone,d_G,ld,d_M2,ld,&szero,d_M,ld));

    /* check ||I-M|| */
    PetscCallCUDA(cudaMemcpy(d_M2,d_M,sizeof(PetscScalar)*N,cudaMemcpyDeviceToDevice));
    PetscCall(shift_diagonal(n,d_M2,ld,-1.0));
    PetscCallCUBLAS(cublasXnrm2(cublasv2handle,N,d_M2,one,&Mres));
    PetscCheck(!PetscIsNanReal(Mres),PETSC_COMM_SELF,PETSC_ERR_FP,"The computed norm is not-a-number");
    if (Mres<=tol) converged = PETSC_TRUE;
    PetscCall(PetscInfo(fn,"it: %" PetscInt_FMT " res: %g\n",it,(double)Mres));
    PetscCall(PetscLogGpuFlops(8.0*n*n*n+2.0*n*n+2.0*n*n*n/3.0+4.0*n*n*n/3.0+2.0*n*n*n+2.0*n*n));
  }

  PetscCheck(Mres<=tol,PETSC_COMM_SELF,PETSC_ERR_LIB,"SQRTM not converged after %d iterations", MAXIT);

  if (nrm>1.0) PetscCallCUBLAS(cublasXscal(cublasv2handle,N,&sqrtnrm,d_X,one));
  PetscCallCUDA(cudaMemcpy(A,d_X,sizeof(PetscScalar)*N,cudaMemcpyDeviceToHost));
  PetscCall(PetscLogGpuTimeEnd());

  PetscCallCUDA(cudaFree(d_X));
  PetscCallCUDA(cudaFree(d_M));
  PetscCallCUDA(cudaFree(d_M2));
  PetscCallCUDA(cudaFree(d_G));
  PetscCallCUDA(cudaFree(d_work));
  PetscCall(PetscFree(piv));

  magma_finalize();
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&Ba));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmSadeghi(fn,n,Ba,n));
  PetscCall(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_CUDA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_NS_CUDA(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&Ba));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmNewtonSchulz_CUDA(fn,n,Ba,n,PETSC_FALSE));
  PetscCall(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_MAGMA)
PetscErrorCode FNEvaluateFunctionMat_Sqrt_DBP_CUDAm(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *T;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&T));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmDenmanBeavers_CUDAm(fn,n,T,n,PETSC_FALSE));
  PetscCall(MatDenseRestoreArray(B,&T));
  PetscFunctionReturn(0);
}

PetscErrorCode FNEvaluateFunctionMat_Sqrt_Sadeghi_CUDAm(FN fn,Mat A,Mat B)
{
  PetscBLASInt   n=0;
  PetscScalar    *Ba;
  PetscInt       m;

  PetscFunctionBegin;
  if (A!=B) PetscCall(MatCopy(A,B,SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArray(B,&Ba));
  PetscCall(MatGetSize(A,&m,NULL));
  PetscCall(PetscBLASIntCast(m,&n));
  PetscCall(FNSqrtmSadeghi_CUDAm(fn,n,Ba,n));
  PetscCall(MatDenseRestoreArray(B,&Ba));
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */

PetscErrorCode FNView_Sqrt(FN fn,PetscViewer viewer)
{
  PetscBool      isascii;
  char           str[50];
  const char     *methodname[] = {
                  "Schur method for the square root",
                  "Denman-Beavers (product form)",
                  "Newton-Schulz iteration",
                  "Sadeghi iteration"
#if defined(PETSC_HAVE_CUDA)
                 ,"Newton-Schulz iteration CUDA"
#if defined(PETSC_HAVE_MAGMA)
                 ,"Denman-Beavers (product form) CUDA/MAGMA",
                  "Sadeghi iteration CUDA/MAGMA"
#endif
#endif
  };
  const int      nmeth=sizeof(methodname)/sizeof(methodname[0]);

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii));
  if (isascii) {
    if (fn->beta==(PetscScalar)1.0) {
      if (fn->alpha==(PetscScalar)1.0) PetscCall(PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(x)\n"));
      else {
        PetscCall(SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE));
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Square root: sqrt(%s*x)\n",str));
      }
    } else {
      PetscCall(SlepcSNPrintfScalar(str,sizeof(str),fn->beta,PETSC_TRUE));
      if (fn->alpha==(PetscScalar)1.0) PetscCall(PetscViewerASCIIPrintf(viewer,"  Square root: %s*sqrt(x)\n",str));
      else {
        PetscCall(PetscViewerASCIIPrintf(viewer,"  Square root: %s",str));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
        PetscCall(SlepcSNPrintfScalar(str,sizeof(str),fn->alpha,PETSC_TRUE));
        PetscCall(PetscViewerASCIIPrintf(viewer,"*sqrt(%s*x)\n",str));
        PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      }
    }
    if (fn->method<nmeth) PetscCall(PetscViewerASCIIPrintf(viewer,"  computing matrix functions with: %s\n",methodname[fn->method]));
  }
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode FNCreate_Sqrt(FN fn)
{
  PetscFunctionBegin;
  fn->ops->evaluatefunction          = FNEvaluateFunction_Sqrt;
  fn->ops->evaluatederivative        = FNEvaluateDerivative_Sqrt;
  fn->ops->evaluatefunctionmat[0]    = FNEvaluateFunctionMat_Sqrt_Schur;
  fn->ops->evaluatefunctionmat[1]    = FNEvaluateFunctionMat_Sqrt_DBP;
  fn->ops->evaluatefunctionmat[2]    = FNEvaluateFunctionMat_Sqrt_NS;
  fn->ops->evaluatefunctionmat[3]    = FNEvaluateFunctionMat_Sqrt_Sadeghi;
#if defined(PETSC_HAVE_CUDA)
  fn->ops->evaluatefunctionmat[4]    = FNEvaluateFunctionMat_Sqrt_NS_CUDA;
#if defined(PETSC_HAVE_MAGMA)
  fn->ops->evaluatefunctionmat[5]    = FNEvaluateFunctionMat_Sqrt_DBP_CUDAm;
  fn->ops->evaluatefunctionmat[6]    = FNEvaluateFunctionMat_Sqrt_Sadeghi_CUDAm;
#endif /* PETSC_HAVE_MAGMA */
#endif /* PETSC_HAVE_CUDA */
  fn->ops->evaluatefunctionmatvec[0] = FNEvaluateFunctionMatVec_Sqrt_Schur;
  fn->ops->view                      = FNView_Sqrt;
  PetscFunctionReturn(0);
}
