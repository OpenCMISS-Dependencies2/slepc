
/*                       
       This implements the Arnoldi method with explicit restart and
       deflation.
*/
#include "src/eps/epsimpl.h"
#include "slepcblaslapack.h"

typedef PetscTruth logical;
typedef PetscBLASInt integer;
typedef PetscScalar doublereal;
typedef PetscBLASInt ftnlen;

extern int dlaqr3_(logical *wantt, logical *wantz, integer *n, 
	integer *ilo, integer *ihi, doublereal *h__, integer *ldh, doublereal 
	*wr, doublereal *wi, integer *iloz, integer *ihiz, doublereal *z__, 
	integer *ldz, doublereal *work, integer *info);
        
#undef __FUNCT__  
#define __FUNCT__ "EPSSetUp_ARNOLDI"
static int EPSSetUp_ARNOLDI(EPS eps)
{
  int         ierr, N;

  PetscFunctionBegin;
  ierr = VecGetSize(eps->vec_initial,&N);CHKERRQ(ierr);
  if (eps->ncv) {
    if (eps->ncv<eps->nev) SETERRQ(1,"The value of ncv must be at least nev"); 
  }
  else eps->ncv = PetscMax(2*eps->nev,eps->nev+8);
  if (!eps->max_it) eps->max_it = PetscMax(100,N);
  if (!eps->tol) eps->tol = 1.e-7;

  ierr = EPSAllocateSolution(eps);CHKERRQ(ierr);
  ierr = EPSDefaultGetWork(eps,eps->ncv+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSBasicArnoldi"
int  EPSBasicArnoldi(EPS eps,PetscScalar *H,Vec *V,int k,int m,Vec f,PetscReal *beta)
{
  int         ierr,j;
  PetscReal   norm;
  PetscScalar t;

  PetscFunctionBegin;
  for (j=k;j<m-1;j++) {
    ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
    ierr = (*eps->orthog)(eps,j+1,V,f,H+m*j,&norm);CHKERRQ(ierr);
    if (norm<1e-8) SETERRQ(1,"Breakdown in Arnoldi method");
    H[(m+1)*j+1] = norm;
    t = 1 / norm;
    ierr = VecScale(&t,f);CHKERRQ(ierr);
    ierr = VecCopy(f,V[j+1]);CHKERRQ(ierr);
  }
  ierr = STApply(eps->OP,V[j],f);CHKERRQ(ierr);
  ierr = (*eps->orthog)(eps,j+1,V,f,H+m*j,beta);CHKERRQ(ierr);
  if (norm<1e-8) SETERRQ(1,"Breakdown in Arnoldi method");
  t = 1 / *beta;
  ierr = VecScale(&t,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "EPSSolve_ARNOLDI"
static int  EPSSolve_ARNOLDI(EPS eps)
{
  int         ierr,i,ilo,info,mout,ncv=eps->ncv,int_one=1;
  PetscTruth  bool_true = PETSC_TRUE;
  Vec         f=eps->work[ncv];
  PetscScalar *H,*U,*work,t;
  PetscReal   norm,beta;
#if defined(PETSC_USE_COMPLEX)
  PetscReal   *rwork;
#endif

  PetscFunctionBegin;
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMemzero(H,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#else
  ierr = PetscMalloc(3*ncv*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*sizeof(PetscReal),&rwork);CHKERRQ(ierr);
#endif

  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = VecNorm(eps->V[0],NORM_2,&norm);CHKERRQ(ierr);
  t = 1 / norm;
  ierr = VecScale(&t,eps->V[0]);CHKERRQ(ierr);
  
  eps->nconv = 0;
  eps->its = 0;
  while (eps->its<eps->max_it) {
    eps->its = eps->its + 1;
  /* [H,V,f,beta] = karnoldi(es,H,V,nconv+1,m) % Arnoldi factorization */
    ierr = EPSBasicArnoldi(eps,H,eps->V,eps->nconv,ncv,f,&beta);CHKERRQ(ierr);
  /* U = eye(m,m) */
    ierr = PetscMemzero(U,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    for (i=0;i<ncv;i++) { U[i*(ncv+1)] = 1.0; }
  /* [T,wr0,wi0,U] = laqr3(H,U,nconv+1,ncv) */
    ilo = eps->nconv+1;
    dlaqr3_(&bool_true,&bool_true,&ncv,&ilo,&ncv,H,&ncv,eps->eigr,eps->eigi,&int_one,&ncv,U,&ncv,work,&info);
  /* V(:,idx) = V*U(:,idx) */
    ierr = EPSReverseProjection(eps,eps->V,U,eps->nconv,ncv,eps->work);CHKERRQ(ierr);
  /* [Y,dummy] = eig(H) */
#if !defined(PETSC_USE_COMPLEX)
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,U,&ncv,&ncv,&mout,work,&ierr,1,1);
#else
    LAtrevc_("R","B",PETSC_NULL,&ncv,H,&ncv,PETSC_NULL,&ncv,U,&ncv,&ncv,&mout,work,rwork,&ierr,1,1);
#endif
  /* rsd = beta*abs(Y(m,:)) */
    for (i=eps->nconv;i<ncv;i++) { 
      eps->errest[i] = beta*PetscAbsScalar(U[i*ncv+ncv-1]); 
      if (eps->errest[i] < eps->tol) eps->nconv = i + 1;
    }
    EPSMonitor(eps,eps->its,eps->nconv,eps->eigr,eps->eigi,eps->errest,ncv);
    if (eps->nconv >= eps->nev) break;
  }
  
  if( eps->nconv >= eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(U);CHKERRQ(ierr);
  ierr = PetscFree(work);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}



















#if 0
  int         ierr, i, j, k, m, maxit=eps->max_it, ncv = eps->ncv;
  int         lwork, ilo, mout;
  Vec         w;
  PetscReal   norm, tol=eps->tol;
  PetscScalar alpha, *H, *Y, *S, *pV, *work;
#if defined(PETSC_USE_COMPLEX)
  PetscReal   *rwork;
#endif

  PetscFunctionBegin;
  w  = eps->work[0];
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&H);CHKERRQ(ierr);
  ierr = PetscMemzero(H,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&U);CHKERRQ(ierr);



  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&Y);CHKERRQ(ierr);
  ierr = PetscMalloc(ncv*ncv*sizeof(PetscScalar),&S);CHKERRQ(ierr);

  ierr = VecCopy(eps->vec_initial,eps->V[0]);CHKERRQ(ierr);
  ierr = VecNorm(eps->V[0],NORM_2,&norm);CHKERRQ(ierr);
  if (norm==0.0) SETERRQ( 1,"Null initial vector" );
  alpha = 1.0/norm;
  ierr = VecScale(&alpha,eps->V[0]);CHKERRQ(ierr);

  eps->its = 0;
  m = ncv-1; /* m is the number of Arnoldi vectors, one less than
                the available vectors because one is needed for v_{m+1} */
  k = 0;     /* k is the number of locked vectors */

  while (eps->its<maxit) {
    /* Perform a Schur-Rayleigh-Ritz projection */
    EPSBasicArnoldi(eps,

    /* compute the projected matrix, H, with the basic Arnoldi method */
    for (j=k;j<m;j++) {

      /* w = OP v_j */
      ierr = STApply(eps->OP,eps->V[j],eps->V[j+1]);CHKERRQ(ierr);

      /* orthogonalize wrt previous vectors */
      ierr = (*eps->orthog)(eps,j+1,eps->V,eps->V[j+1],&H[0+ncv*j],&norm);CHKERRQ(ierr);

      /* h_{j+1,j} = ||w||_2 */
      if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
      H[j+1+ncv*j] = norm;
      alpha = 1.0/norm;
      ierr = VecScale(&alpha,eps->V[j+1]);CHKERRQ(ierr);

    }

    /* At this point, H has the following structure

              | *   * | *   *   *   * |
              |     * | *   *   *   * |
              | ------|-------------- |
          H = |       | *   *   *   * |
              |       | *   *   *   * |
              |       |     *   *   * |
              |       |         *   * |

       that is, a mxm upper Hessenberg matrix whose kxk principal submatrix
       is (quasi-)triangular.
     */

    /* reduce H to (real) Schur form, H = S \tilde{H} S'  */
    lwork = m;
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
    ilo = k+1;
#if !defined(PETSC_USE_COMPLEX)
    LAhseqr_("S","I",&m,&ilo,&m,H,&ncv,eps->eigr,eps->eigi,S,&ncv,work,&lwork,&ierr,1,1);
#else
    LAhseqr_("S","I",&m,&ilo,&m,H,&ncv,eps->eigr,S,&ncv,work,&lwork,&ierr,1,1);
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);
 
    /* compute eigenvectors y_i */
    ierr = PetscMemcpy(Y,S,ncv*ncv*sizeof(PetscScalar));CHKERRQ(ierr);
    lwork = 3*m;
    ierr = PetscMalloc(lwork*sizeof(PetscScalar),&work);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    LAtrevc_("R","B",PETSC_NULL,&m,H,&ncv,Y,&ncv,Y,&ncv,&ncv,&mout,work,&ierr,1,1);
#else
    ierr = PetscMalloc(2*m*sizeof(PetscScalar),&rwork);CHKERRQ(ierr);
    LAtrevc_("R","B",PETSC_NULL,&m,H,&ncv,Y,&ncv,Y,&ncv,&ncv,&mout,work,rwork,&ierr,1,1);
    ierr = PetscFree(rwork);CHKERRQ(ierr);
#endif
    ierr = PetscFree(work);CHKERRQ(ierr);

    /* compute error estimates */
    for (j=k;j<m;j++) {
      /* errest_j = h_{m+1,m} |e_m' y_j| */
      eps->errest[j] = PetscRealPart(H[m+ncv*(m-1)]) 
                     * PetscAbsScalar(Y[(m-1)+ncv*j]);
    }

    /* compute Ritz vectors */
    ierr = EPSReverseProjection(eps,k,m-k,S);CHKERRQ(ierr);

    /* lock converged Ritz pairs */
    for (j=k;j<m;j++) {
      if (eps->errest[j]<tol) {
        if (j>k) {
          ierr = EPSSwapEigenpairs(eps,k,j);CHKERRQ(ierr);
        }
        ierr = (*eps->orthog)(eps,k,eps->V,eps->V[k],PETSC_NULL,&norm);CHKERRQ(ierr);
        if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
        alpha = 1.0/norm;
        ierr = VecScale(&alpha,eps->V[k]);CHKERRQ(ierr);
        /* h_{i,k} = v_i' OP v_k, i=1..k */
        for (i=0;i<=k;i++) {
          ierr = STApply(eps->OP,eps->V[k],w);CHKERRQ(ierr);
          ierr = VecDot(w,eps->V[i],H+i+ncv*k);CHKERRQ(ierr);
        }
        H[k+1+ncv*k] = 0.0;
        k = k + 1;
      }
    }
    eps->nconv = k;

    /* select next wanted eigenvector as restart vector */
    ierr = EPSSortEigenvalues(m-k,eps->eigr+k,eps->eigi+k,eps->which,1,&i);CHKERRQ(ierr);
    ierr = EPSSwapEigenpairs(eps,k,k+i);CHKERRQ(ierr);

    /* orthogonalize u_k wrt previous vectors */
    ierr = (*eps->orthog)(eps,k,eps->V,eps->V[k],PETSC_NULL,&norm);CHKERRQ(ierr);

    /* normalize new initial vector */
    if (norm==0.0) SETERRQ( 1,"Breakdown in Arnoldi method" );
    alpha = 1.0/norm;
    ierr = VecScale(&alpha,eps->V[k]);CHKERRQ(ierr);

    EPSMonitor(eps,eps->its + 1,eps->nconv,eps->eigr,eps->eigi,eps->errest,m); 
    eps->its = eps->its + 1;

    if (eps->nconv>=eps->nev) break;

  }

  ierr = PetscFree(H);CHKERRQ(ierr);
  ierr = PetscFree(Y);CHKERRQ(ierr);
  ierr = PetscFree(S);CHKERRQ(ierr);

  if( eps->its==maxit ) eps->its = eps->its - 1;
  if( eps->nconv == eps->nev ) eps->reason = EPS_CONVERGED_TOL;
  else eps->reason = EPS_DIVERGED_ITS;
#if defined(PETSC_USE_COMPLEX)
  for (i=0;i<eps->nconv;i++) eps->eigi[i]=0.0;
#endif

  PetscFunctionReturn(0);
#endif

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "EPSCreate_ARNOLDI"
int EPSCreate_ARNOLDI(EPS eps)
{
  PetscFunctionBegin;
  eps->data                      = (void *) 0;
  eps->ops->setfromoptions       = 0;
  eps->ops->setup                = EPSSetUp_ARNOLDI;
  eps->ops->solve                = EPSSolve_ARNOLDI;
  eps->ops->destroy              = EPSDestroy_Default;
  eps->ops->backtransform        = EPSBackTransform_Default;
  eps->computevectors            = EPSComputeVectors_Default;
  PetscFunctionReturn(0);
}
EXTERN_C_END

