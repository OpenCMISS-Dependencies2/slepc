/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2020, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   This file implements a wrapper to eigensolvers in EVSL.
*/

#include <slepc/private/epsimpl.h>    /*I "slepceps.h" I*/
#include <evsl.h>

typedef struct {
  PetscBool         initialized;
  Mat               A;           /* problem matrix */
  Vec               x,y;         /* auxiliary vectors */
  PetscReal         *sli;        /* slice bounds */
  PetscInt          nev;         /* approximate number of wanted eigenvalues in each slice */
  PetscLayout       map;         /* used to distribute slices among MPI processes */
  PetscBool         estimrange;  /* the filter range was not set by the user */
  /* user parameters */
  PetscInt          nslices;     /* number of slices */
  PetscReal         lmin,lmax;   /* numerical range (min and max eigenvalue) */
  EPSEVSLDOSMethod  dos;         /* DOS method, either KPM or Lanczos */
  PetscInt          nvec;        /* number of sample vectors used for DOS */
  PetscInt          deg;         /* polynomial degree used for DOS (KPM only) */
  PetscInt          steps;       /* number of Lanczos steps used for DOS (Lanczos only) */
  PetscInt          npoints;     /* number of sample points used for DOS (Lanczos only) */
  PetscInt          max_deg;     /* maximum degree allowed for the polynomial */
  PetscReal         thresh;      /* threshold for accepting polynomial */
  EPSEVSLDamping    damping;     /* type of damping (for polynomial and for DOS-KPM) */
} EPS_EVSL;

static void AMatvec_EVSL(double *xa,double *ya,void *data)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)data;
  Vec            x = ctx->x,y = ctx->y;
  Mat            A = ctx->A;

  PetscFunctionBegin;
  ierr = VecPlaceArray(x,(PetscScalar*)xa);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecPlaceArray(y,(PetscScalar*)ya);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = MatMult(A,x,y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecResetArray(x);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  ierr = VecResetArray(y);CHKERRABORT(PetscObjectComm((PetscObject)A),ierr);
  PetscFunctionReturnVoid();
}

PetscErrorCode EPSSetUp_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;
  PetscMPIInt    size,rank;
  PetscBool      isshift;
  PetscScalar    *vinit;
  PetscReal      *mu,ecount,xintv[4],*xdos,*ydos;
  Vec            v0;
  Mat            A;
  PetscRandom    rnd;

  PetscFunctionBegin;
  EPSCheckStandard(eps);
  EPSCheckHermitian(eps);
  ierr = PetscObjectTypeCompare((PetscObject)eps->st,STSHIFT,&isshift);CHKERRQ(ierr);
  if (!isshift) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver does not support spectral transformations");

  if (ctx->initialized) EVSLFinish();
  EVSLStart();
  ctx->initialized=PETSC_TRUE;

  /* get number of slices per process */
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank);CHKERRMPI(ierr);
  if (!ctx->nslices) ctx->nslices = size;
  ierr = PetscLayoutDestroy(&ctx->map);CHKERRQ(ierr);
  ierr = PetscLayoutCreateFromSizes(PetscObjectComm((PetscObject)eps),PETSC_DECIDE,ctx->nslices,1,&ctx->map);CHKERRQ(ierr);

  /* get matrix and prepare auxiliary vectors */
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = STGetMatrix(eps->st,0,&A);CHKERRQ(ierr);
  if (size==1) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    ctx->A = A;
  } else {
    ierr = MatCreateRedundantMatrix(A,0,PETSC_COMM_SELF,MAT_INITIAL_MATRIX,&ctx->A);CHKERRQ(ierr);
  }
  SetAMatvec(eps->n,&AMatvec_EVSL,(void*)ctx);
  if (!ctx->x) {
    ierr = MatCreateVecsEmpty(ctx->A,&ctx->x,&ctx->y);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->x);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)eps,(PetscObject)ctx->y);CHKERRQ(ierr);
  }
  EPSCheckUnsupported(eps,EPS_FEATURE_ARBITRARY | EPS_FEATURE_REGION | EPS_FEATURE_STOPPING);
  EPSCheckIgnored(eps,EPS_FEATURE_EXTRACTION | EPS_FEATURE_CONVERGENCE);

  if (!eps->which) eps->which=EPS_ALL;
  if (eps->which!=EPS_ALL || eps->inta==eps->intb) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_SUP,"This solver requires setting an interval with EPSSetInterval()");

  /* estimate numerical range */
  if (ctx->estimrange || ctx->lmin == PETSC_MIN_REAL || ctx->lmax == PETSC_MAX_REAL) {
    ierr = MatCreateVecs(ctx->A,&v0,NULL);CHKERRQ(ierr);
    if (!eps->V) { ierr = EPSGetBV(eps,&eps->V);CHKERRQ(ierr); }
    ierr = BVGetRandomContext(eps->V,&rnd);CHKERRQ(ierr);
    ierr = VecSetRandom(v0,rnd);CHKERRQ(ierr);
    ierr = VecGetArray(v0,&vinit);CHKERRQ(ierr);
    ierr = LanTrbounds(50,200,eps->tol,vinit,1,&ctx->lmin,&ctx->lmax,NULL);CHKERRQ(ierr);
    ierr = VecRestoreArray(v0,&vinit);CHKERRQ(ierr);
    ierr = VecDestroy(&v0);CHKERRQ(ierr);
    ctx->estimrange = PETSC_TRUE;   /* estimate if called again with another matrix */
  }
  if (ctx->lmin > eps->inta || ctx->lmax < eps->intb) SETERRQ4(PetscObjectComm((PetscObject)eps),1,"The requested interval [%g,%g] must be contained in the numerical range [%g,%g]",(double)eps->inta,(double)eps->intb,(double)ctx->lmin,(double)ctx->lmax);
  xintv[0] = eps->inta;
  xintv[1] = eps->intb;
  xintv[2] = ctx->lmin;
  xintv[3] = ctx->lmax;

  /* estimate number of eigenvalues in the interval */
  if (ctx->dos == EPS_EVSL_DOS_KPM) {
    ierr = PetscMalloc1(ctx->deg+1,&mu);CHKERRQ(ierr);
    if (!rank) { ierr = kpmdos(ctx->deg,(int)ctx->damping,ctx->nvec,xintv,mu,&ecount);CHKERRQ(ierr); }
    ierr = MPI_Bcast(mu,ctx->deg+1,MPIU_REAL,0,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
  } else if (ctx->dos == EPS_EVSL_DOS_LANCZOS) {
    ierr = PetscMalloc2(ctx->npoints,&xdos,ctx->npoints,&ydos);CHKERRQ(ierr);
    if (!rank) { ierr = LanDos(ctx->nvec,PetscMin(ctx->steps,eps->n/2),ctx->npoints,xdos,ydos,&ecount,xintv);CHKERRQ(ierr); }
    ierr = MPI_Bcast(xdos,ctx->npoints,MPIU_REAL,0,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
    ierr = MPI_Bcast(ydos,ctx->npoints,MPIU_REAL,0,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Invalid DOS method");
  ierr = MPI_Bcast(&ecount,1,MPIU_REAL,0,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);

  ierr = PetscInfo1(eps,"Estimated eigenvalue count in the interval: %g\n",ecount);CHKERRQ(ierr);
  eps->ncv = (PetscInt)PetscCeilReal(1.5*ecount);

  /* slice the spectrum */
  ierr = PetscFree(ctx->sli);CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx->nslices+1,&ctx->sli);CHKERRQ(ierr);
  if (ctx->dos == EPS_EVSL_DOS_KPM) {
    ierr = spslicer(ctx->sli,mu,ctx->deg,xintv,ctx->nslices,10*(PetscInt)ecount);CHKERRQ(ierr);
    ierr = PetscFree(mu);CHKERRQ(ierr);
  } else if (ctx->dos == EPS_EVSL_DOS_LANCZOS) {
    spslicer2(xdos,ydos,ctx->nslices,ctx->npoints,ctx->sli);
    ierr = PetscFree2(xdos,ydos);CHKERRQ(ierr);
  }

  /* approximate number of eigenvalues wanted in each slice */
  ctx->nev = (PetscInt)(1.0 + ecount/(PetscReal)ctx->nslices) + 2;

  if (eps->mpd!=PETSC_DEFAULT) { ierr = PetscInfo(eps,"Warning: parameter mpd ignored\n");CHKERRQ(ierr); }
  if (eps->max_it==PETSC_DEFAULT) eps->max_it = 1;
  ierr = EPSAllocateSolution(eps,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSolve_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;
  PetscInt       i,j,k=0,sl,mlan,nevout,*ind,nevmax,rstart,rend,*nevloc,*disp,N;
  PetscReal      *res,xintv[4],*errest;
  PetscScalar    *lam,*X,*Y,*vinit,*eigr;
  PetscMPIInt    size,rank;
  PetscRandom    rnd;
  Vec            v,w,v0,x;
  VecScatter     vs;
  IS             is;
  polparams      pol;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)eps),&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)eps),&rank);CHKERRMPI(ierr);
  ierr = PetscLayoutGetRange(ctx->map,&rstart,&rend);CHKERRQ(ierr);
  nevmax = (rend-rstart)*ctx->nev;
  ierr = MatCreateVecs(ctx->A,&v0,NULL);CHKERRQ(ierr);
  ierr = BVGetRandomContext(eps->V,&rnd);CHKERRQ(ierr);
  ierr = VecSetRandom(v0,rnd);CHKERRQ(ierr);
  ierr = VecGetArray(v0,&vinit);CHKERRQ(ierr);
  ierr = PetscMalloc5(size,&nevloc,size+1,&disp,nevmax,&eigr,nevmax,&errest,nevmax*eps->n,&X);CHKERRQ(ierr);
  mlan = PetscMin(PetscMax(5*ctx->nev,300),eps->n);
  for (sl=rstart; sl<rend; sl++) {
    xintv[0] = ctx->sli[sl];
    xintv[1] = ctx->sli[sl+1];
    xintv[2] = ctx->lmin;
    xintv[3] = ctx->lmax;
    ierr = PetscInfo3(ctx->A,"Subinterval %D: [%.4e, %.4e]\n",sl+1,xintv[0],xintv[1]);CHKERRQ(ierr);
    set_pol_def(&pol);
    pol.max_deg    = ctx->max_deg;
    pol.damping    = (int)ctx->damping;
    pol.thresh_int = ctx->thresh;
    find_pol(xintv,&pol);
    ierr = PetscInfo4(ctx->A,"Polynomial [type = %D], deg %D, bar %e gam %e\n",pol.type,pol.deg,pol.bar,pol.gam);CHKERRQ(ierr);
    ierr = ChebLanNr(xintv,mlan,eps->tol,vinit,&pol,&nevout,&lam,&Y,&res,NULL);CHKERRQ(ierr);
    if (k+nevout>nevmax) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Too low estimation of eigenvalue count, try modifying the sampling parameters");
    free_pol(&pol);
    ierr = PetscInfo1(ctx->A,"Computed %D eigenvalues\n",nevout);CHKERRQ(ierr);
    ierr = PetscMalloc1(nevout,&ind);CHKERRQ(ierr);
    sort_double(nevout,lam,ind);
    for (i=0;i<nevout;i++) {
      eigr[i+k]   = lam[i];
      errest[i+k] = res[ind[i]];
      ierr = PetscArraycpy(X+(i+k)*eps->n,Y+ind[i]*eps->n,eps->n);CHKERRQ(ierr);
    }
    k += nevout;
    if (lam) evsl_Free(lam);
    if (Y)   evsl_Free_device(Y);
    if (res) evsl_Free(res);
    ierr = PetscFree(ind);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v0,&vinit);CHKERRQ(ierr);
  ierr = VecDestroy(&v0);CHKERRQ(ierr);

  /* gather eigenvalues computed by each MPI process */
  ierr = MPI_Allgather(&k,1,MPIU_INT,nevloc,1,MPIU_INT,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
  eps->nev = nevloc[0];
  disp[0]  = 0;
  for (i=1;i<size;i++) {
    eps->nev += nevloc[i];
    disp[i]   = disp[i-1]+nevloc[i-1];
  }
  disp[size] = disp[size-1]+nevloc[size-1];
  if (eps->nev>eps->ncv) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_LIB,"Too low estimation of eigenvalue count, try modifying the sampling parameters");
  ierr = MPI_Allgatherv(eigr,k,MPIU_SCALAR,eps->eigr,nevloc,disp,MPIU_SCALAR,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
  ierr = MPI_Allgatherv(errest,k,MPIU_REAL,eps->errest,nevloc,disp,MPIU_REAL,PetscObjectComm((PetscObject)eps));CHKERRMPI(ierr);
  eps->nconv  = eps->nev;
  eps->its    = 1;
  eps->reason = EPS_CONVERGED_TOL;

  /* scatter computed eigenvectors and store them in eps->V */
  ierr = BVCreateVec(eps->V,&w);CHKERRQ(ierr);
  for (i=0;i<size;i++) {
    N = (rank==i)? eps->n: 0;
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x);CHKERRQ(ierr);
    ierr = VecSetFromOptions(x);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&is);CHKERRQ(ierr);
    ierr = VecScatterCreate(x,is,w,is,&vs);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    for (j=disp[i];j<disp[i+1];j++) {
      ierr = BVGetColumn(eps->V,j,&v);CHKERRQ(ierr);
      if (rank==i) { ierr = VecPlaceArray(x,X+(j-disp[i])*eps->n);CHKERRQ(ierr); }
      ierr = VecScatterBegin(vs,x,v,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(vs,x,v,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      if (rank==i) { ierr = VecResetArray(x);CHKERRQ(ierr); }
      ierr = BVRestoreColumn(eps->V,j,&v);CHKERRQ(ierr);
    }
    ierr = VecScatterDestroy(&vs);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&w);CHKERRQ(ierr);
  ierr = PetscFree5(nevloc,disp,eigr,errest,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetSlices_EVSL(EPS eps,PetscInt nslices)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (nslices == PETSC_DECIDE || nslices == PETSC_DEFAULT) nslices = 0;
  else if (nslices<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"Number of slices must be 1 at least");
  if (ctx->nslices != nslices) {
    ctx->nslices = nslices;
    eps->state   = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetSlices - Set the number of slices in which the interval must be
   subdivided.

   Logically Collective on eps

   Input Parameters:
+  eps     - the eigensolver context
-  nslices - the number of slices

   Options Database Key:
.  -eps_evsl_slices <n> - set the number of slices to n

   Notes:
   By default, one slice per MPI process is used. Depending on the number of
   eigenvalues, using more slices may be beneficial, but very narrow subintervals
   imply higher polynomial degree.

   Level: intermediate

.seealso: EPSEVSLGetSlices()
@*/
PetscErrorCode EPSEVSLSetSlices(EPS eps,PetscInt nslices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,nslices,2);
  ierr = PetscTryMethod(eps,"EPSEVSLSetSlices_C",(EPS,PetscInt),(eps,nslices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetSlices_EVSL(EPS eps,PetscInt *nslices)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  *nslices = ctx->nslices;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetSlices - Gets the number of slices in which the interval must be
   subdivided.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  nslices - the number of slices

   Level: intermediate

.seealso: EPSEVSLSetSlices()
@*/
PetscErrorCode EPSEVSLGetSlices(EPS eps,PetscInt *nslices)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(nslices,2);
  ierr = PetscUseMethod(eps,"EPSEVSLGetSlices_C",(EPS,PetscInt*),(eps,nslices));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetRange_EVSL(EPS eps,PetscReal lmin,PetscReal lmax)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (lmin>lmax) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_WRONG,"Badly defined interval, must be lmin<lmax");
  if (ctx->lmin != lmin || ctx->lmax != lmax) {
    ctx->lmin  = lmin;
    ctx->lmax  = lmax;
    eps->state = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetRange - Defines the numerical range (or field of values) of the problem,
   that is, the interval containing all eigenvalues.

   Logically Collective on eps

   Input Parameters:
+  eps  - the eigensolver context
.  lmin - left end of the interval
-  lmax - right end of the interval

   Options Database Key:
.  -eps_evsl_range <a,b> - set [a,b] as the numerical range

   Notes:
   The filter will be most effective if the numerical range is tight, that is, lmin
   and lmax are good approximations to the leftmost and rightmost eigenvalues,
   respectively. If not set by the user, an approximation is computed internally.

   The wanted computational interval specified via EPSSetInterval() must be
   contained in the numerical range.

   Level: intermediate

.seealso: EPSEVSLGetRange(), EPSSetInterval()
@*/
PetscErrorCode EPSEVSLSetRange(EPS eps,PetscReal lmin,PetscReal lmax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveReal(eps,lmin,2);
  PetscValidLogicalCollectiveReal(eps,lmax,3);
  ierr = PetscTryMethod(eps,"EPSEVSLSetRange_C",(EPS,PetscReal,PetscReal),(eps,lmin,lmax));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetRange_EVSL(EPS eps,PetscReal *lmin,PetscReal *lmax)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (lmin) *lmin = ctx->lmin;
  if (lmax) *lmax = ctx->lmax;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetRange - Gets the interval containing all eigenvalues.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameters:
+  lmin - left end of the interval
-  lmax - right end of the interval

   Level: intermediate

.seealso: EPSEVSLSetRange()
@*/
PetscErrorCode EPSEVSLGetRange(EPS eps,PetscReal *lmin,PetscReal *lmax)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSEVSLGetRange_C",(EPS,PetscReal*,PetscReal*),(eps,lmin,lmax));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetDOSParameters_EVSL(EPS eps,EPSEVSLDOSMethod dos,PetscInt nvec,PetscInt deg,PetscInt steps,PetscInt npoints)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ctx->dos = dos;
  if (nvec == PETSC_DECIDE || nvec == PETSC_DEFAULT) ctx->nvec = 80;
  else if (nvec<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The nvec argument must be > 0");
  else ctx->nvec = nvec;
  switch (dos) {
    case EPS_EVSL_DOS_KPM:
      if (deg == PETSC_DECIDE || deg == PETSC_DEFAULT) ctx->deg = 300;
      else if (deg<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The deg argument must be > 0");
      else ctx->deg = deg;
      break;
    case EPS_EVSL_DOS_LANCZOS:
      if (steps == PETSC_DECIDE || steps == PETSC_DEFAULT) ctx->steps = 40;
      else if (steps<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The steps argument must be > 0");
      else ctx->steps = steps;
      if (npoints == PETSC_DECIDE || npoints == PETSC_DEFAULT) ctx->npoints = 200;
      else if (npoints<1) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The npoints argument must be > 0");
      else ctx->npoints = npoints;
      break;
  }
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetDOSParameters - Defines the parameters used for computing the
   density of states (DOS) in the EVSL solver.

   Logically Collective on eps

   Input Parameters:
+  eps     - the eigensolver context
.  dos     - DOS method, either KPM or Lanczos
.  nvec    - number of sample vectors
.  deg     - polynomial degree (KPM only)
.  steps   - number of Lanczos steps (Lanczos only)
-  npoints - number of sample points (Lanczos only)

   Options Database Keys:
+  -eps_evsl_dos_method <dos> - set the DOS method, either kpm or lanczos
.  -eps_evsl_dos_nvec <n> - set the number of sample vectors
.  -eps_evsl_dos_degree <n> - set the polynomial degree
.  -eps_evsl_dos_steps <n> - set the number of Lanczos steps
-  -eps_evsl_dos_npoints <n> - set the number of sample points

   Notes:
   The density of states (or spectral density) can be approximated with two
   methods: kernel polynomial method (KPM) or Lanczos. Some parameters for
   these methods can be set by the user with this function, with some of
   them being relevant for one of the methods only.

   Level: intermediate

.seealso: EPSEVSLGetDOSParameters()
@*/
PetscErrorCode EPSEVSLSetDOSParameters(EPS eps,EPSEVSLDOSMethod dos,PetscInt nvec,PetscInt deg,PetscInt steps,PetscInt npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,dos,2);
  PetscValidLogicalCollectiveInt(eps,nvec,3);
  PetscValidLogicalCollectiveInt(eps,deg,4);
  PetscValidLogicalCollectiveInt(eps,steps,5);
  PetscValidLogicalCollectiveInt(eps,npoints,6);
  ierr = PetscTryMethod(eps,"EPSEVSLSetDOSParameters_C",(EPS,EPSEVSLDOSMethod,PetscInt,PetscInt,PetscInt,PetscInt),(eps,dos,nvec,deg,steps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetDOSParameters_EVSL(EPS eps,EPSEVSLDOSMethod *dos,PetscInt *nvec,PetscInt *deg,PetscInt *steps,PetscInt *npoints)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (dos)     *dos     = ctx->dos;
  if (nvec)    *nvec    = ctx->nvec;
  if (deg)     *deg     = ctx->deg;
  if (steps)   *steps   = ctx->steps;
  if (npoints) *npoints = ctx->npoints;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetDOSParameters - Gets the parameters used for computing the
   density of states (DOS) in the EVSL solver.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameters:
+  dos     - DOS method, either KPM or Lanczos
.  nvec    - number of sample vectors
.  deg     - polynomial degree (KPM only)
.  steps   - number of Lanczos steps (Lanczos only)
-  npoints - number of sample points (Lanczos only)

   Level: intermediate

.seealso: EPSEVSLSetDOSParameters()
@*/
PetscErrorCode EPSEVSLGetDOSParameters(EPS eps,EPSEVSLDOSMethod *dos,PetscInt *nvec,PetscInt *deg,PetscInt *steps,PetscInt *npoints)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSEVSLGetDOSParameters_C",(EPS,EPSEVSLDOSMethod*,PetscInt*,PetscInt*,PetscInt*,PetscInt*),(eps,dos,nvec,deg,steps,npoints));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetPolParameters_EVSL(EPS eps,PetscInt max_deg,PetscReal thresh)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (max_deg == PETSC_DECIDE || max_deg == PETSC_DEFAULT) ctx->max_deg = 10000;
  else if (max_deg<3) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The max_deg argument must be > 2");
  else ctx->max_deg = max_deg;
  if (thresh == PETSC_DECIDE || thresh == PETSC_DEFAULT) ctx->thresh = 0.8;
  else if (thresh<0.0) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_OUTOFRANGE,"The thresh argument must be > 0.0");
  else ctx->thresh = thresh;
  eps->state = EPS_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetPolParameters - Defines the parameters used for building the
   building the polynomial in the EVSL solver.

   Logically Collective on eps

   Input Parameters:
+  eps     - the eigensolver context
.  max_deg - maximum degree allowed for the polynomial
-  thresh  - threshold for accepting polynomial

   Options Database Keys:
+  -eps_evsl_pol_max_deg <d> - set maximum polynomial degree
-  -eps_evsl_pol_thresh <t> - set the threshold

   Level: intermediate

.seealso: EPSEVSLGetPolParameters()
@*/
PetscErrorCode EPSEVSLSetPolParameters(EPS eps,PetscInt max_deg,PetscReal thresh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveInt(eps,max_deg,2);
  PetscValidLogicalCollectiveReal(eps,thresh,3);
  ierr = PetscTryMethod(eps,"EPSEVSLSetPolParameters_C",(EPS,PetscInt,PetscReal),(eps,max_deg,thresh));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetPolParameters_EVSL(EPS eps,PetscInt *max_deg,PetscReal *thresh)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (max_deg) *max_deg = ctx->max_deg;
  if (thresh)  *thresh  = ctx->thresh;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetPolParameters - Gets the parameters used for building the
   polynomial in the EVSL solver.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameters:
+  max_deg - the maximum degree of the polynomial
-  thresh  - the threshold

   Level: intermediate

.seealso: EPSEVSLSetPolParameters()
@*/
PetscErrorCode EPSEVSLGetPolParameters(EPS eps,PetscInt *max_deg,PetscReal *thresh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  ierr = PetscUseMethod(eps,"EPSEVSLGetPolParameters_C",(EPS,PetscInt*,PetscReal*),(eps,max_deg,thresh));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLSetDamping_EVSL(EPS eps,EPSEVSLDamping damping)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (ctx->damping != damping) {
    ctx->damping = damping;
    eps->state   = EPS_STATE_INITIAL;
  }
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLSetDamping - Set the type of damping to be used in EVSL.

   Logically Collective on eps

   Input Parameters:
+  eps     - the eigensolver context
-  damping - the type of damping

   Options Database Key:
.  -eps_evsl_damping <n> - set the type of damping

   Notes:
   Damping is applied when building the polynomial to be used when solving the
   eigenproblem, and also during estimation of DOS with the KPM method.

   Level: intermediate

.seealso: EPSEVSLGetDamping(), EPSEVSLSetDOSParameters()
@*/
PetscErrorCode EPSEVSLSetDamping(EPS eps,EPSEVSLDamping damping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidLogicalCollectiveEnum(eps,damping,2);
  ierr = PetscTryMethod(eps,"EPSEVSLSetDamping_C",(EPS,EPSEVSLDamping),(eps,damping));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSEVSLGetDamping_EVSL(EPS eps,EPSEVSLDamping *damping)
{
  EPS_EVSL *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  *damping = ctx->damping;
  PetscFunctionReturn(0);
}

/*@
   EPSEVSLGetDamping - Gets the type of damping.

   Not Collective

   Input Parameter:
.  eps - the eigensolver context

   Output Parameter:
.  damping - the type of damping

   Level: intermediate

.seealso: EPSEVSLSetDamping()
@*/
PetscErrorCode EPSEVSLGetDamping(EPS eps,EPSEVSLDamping *damping)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(eps,EPS_CLASSID,1);
  PetscValidIntPointer(damping,2);
  ierr = PetscUseMethod(eps,"EPSEVSLGetDamping_C",(EPS,EPSEVSLDamping*),(eps,damping));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSView_EVSL(EPS eps,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      isascii;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  numerical range = [%g,%g]\n",(double)ctx->lmin,(double)ctx->lmax);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  number of slices = %D\n",ctx->nslices);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  type of damping = %s\n",EPSEVSLDampings[ctx->damping]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  computing DOS with %s: nvec=%D, ",EPSEVSLDOSMethods[ctx->dos],ctx->nvec);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    switch (ctx->dos) {
      case EPS_EVSL_DOS_KPM:
        ierr = PetscViewerASCIIPrintf(viewer,"degree=%D\n",ctx->deg);CHKERRQ(ierr);
        break;
      case EPS_EVSL_DOS_LANCZOS:
        ierr = PetscViewerASCIIPrintf(viewer,"steps=%D, npoints=%D\n",ctx->steps,ctx->npoints);CHKERRQ(ierr);
        break;
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  polynomial parameters: max degree = %D, threshold = %g\n",ctx->max_deg,(double)ctx->thresh);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EPSSetFromOptions_EVSL(PetscOptionItems *PetscOptionsObject,EPS eps)
{
  PetscErrorCode   ierr;
  PetscReal        array[2]={0,0},th;
  PetscInt         k,i1,i2,i3,i4;
  PetscBool        flg,flg1;
  EPSEVSLDOSMethod dos;
  EPSEVSLDamping   damping;
  EPS_EVSL         *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"EPS EVSL Options");CHKERRQ(ierr);

    k = 2;
    ierr = PetscOptionsRealArray("-eps_evsl_range","Interval containing all eigenvalues (two real values separated with a comma without spaces)","EPSEVSLSetRange",array,&k,&flg);CHKERRQ(ierr);
    if (flg) {
      if (k<2) SETERRQ(PetscObjectComm((PetscObject)eps),PETSC_ERR_ARG_SIZ,"Must pass two values in -eps_evsl_range (comma-separated without spaces)");
      ierr = EPSEVSLSetRange(eps,array[0],array[1]);CHKERRQ(ierr);
    }

    ierr = PetscOptionsInt("-eps_evsl_slices","Number of slices","EPSEVSLSetSlices",ctx->nslices,&i1,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSEVSLSetSlices(eps,i1);CHKERRQ(ierr); }

    ierr = PetscOptionsEnum("-eps_evsl_damping","Type of damping","EPSEVSLSetDamping",EPSEVSLDampings,(PetscEnum)ctx->damping,(PetscEnum*)&damping,&flg);CHKERRQ(ierr);
    if (flg) { ierr = EPSEVSLSetDamping(eps,damping);CHKERRQ(ierr); }

    ierr = EPSEVSLGetDOSParameters(eps,&dos,&i1,&i2,&i3,&i4);CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-eps_evsl_dos_method","Method to compute the DOS","EPSEVSLSetDOSParameters",EPSEVSLDOSMethods,(PetscEnum)ctx->dos,(PetscEnum*)&dos,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_evsl_dos_nvec","Number of sample vectors for DOS","EPSEVSLSetDOSParameters",i1,&i1,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_degree","Polynomial degree used for DOS","EPSEVSLSetDOSParameters",i2,&i2,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_steps","Number of Lanczos steps in DOS","EPSEVSLSetDOSParameters",i3,&i3,&flg1);CHKERRQ(ierr);
    flg = flg || flg1;
    ierr = PetscOptionsInt("-eps_evsl_dos_npoints","Number of sample points used for DOS","EPSEVSLSetDOSParameters",i4,&i4,&flg1);CHKERRQ(ierr);
    if (flg || flg1) { ierr = EPSEVSLSetDOSParameters(eps,dos,i1,i2,i3,i4);CHKERRQ(ierr); }

    ierr = EPSEVSLGetPolParameters(eps,&i1,&th);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-eps_evsl_pol_max_deg","Maximum degree allowed for the polynomial","EPSEVSLSetPolParameters",i1,&i1,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-eps_evsl_pol_threshold","Threshold for accepting polynomial","EPSEVSLSetPolParameters",th,&th,&flg1);CHKERRQ(ierr);
    if (flg || flg1) { ierr = EPSEVSLSetPolParameters(eps,i1,th);CHKERRQ(ierr); }

  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSDestroy_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  if (ctx->initialized) EVSLFinish();
  ierr = PetscLayoutDestroy(&ctx->map);CHKERRQ(ierr);
  ierr = PetscFree(ctx->sli);CHKERRQ(ierr);
  ierr = PetscFree(eps->data);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetRange_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetRange_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetSlices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetSlices_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDOSParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDOSParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetPolParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetPolParameters_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDamping_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDamping_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode EPSReset_EVSL(EPS eps)
{
  PetscErrorCode ierr;
  EPS_EVSL       *ctx = (EPS_EVSL*)eps->data;

  PetscFunctionBegin;
  ierr = MatDestroy(&ctx->A);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->x);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx->y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode EPSCreate_EVSL(EPS eps)
{
  EPS_EVSL       *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(eps,&ctx);CHKERRQ(ierr);
  eps->data = (void*)ctx;

  ctx->nslices = 0;
  ctx->lmin    = PETSC_MIN_REAL;
  ctx->lmax    = PETSC_MAX_REAL;
  ctx->dos     = EPS_EVSL_DOS_KPM;
  ctx->nvec    = 80;
  ctx->deg     = 300;
  ctx->steps   = 40;
  ctx->npoints = 200;
  ctx->max_deg = 10000;
  ctx->thresh  = 0.8;
  ctx->damping = EPS_EVSL_DAMPING_SIGMA;

  eps->categ = EPS_CATEGORY_OTHER;

  eps->ops->solve          = EPSSolve_EVSL;
  eps->ops->setup          = EPSSetUp_EVSL;
  eps->ops->setupsort      = EPSSetUpSort_Basic;
  eps->ops->setfromoptions = EPSSetFromOptions_EVSL;
  eps->ops->destroy        = EPSDestroy_EVSL;
  eps->ops->reset          = EPSReset_EVSL;
  eps->ops->view           = EPSView_EVSL;
  eps->ops->backtransform  = EPSBackTransform_Default;
  eps->ops->setdefaultst   = EPSSetDefaultST_NoFactor;

  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetRange_C",EPSEVSLSetRange_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetRange_C",EPSEVSLGetRange_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetSlices_C",EPSEVSLSetSlices_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetSlices_C",EPSEVSLGetSlices_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDOSParameters_C",EPSEVSLSetDOSParameters_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDOSParameters_C",EPSEVSLGetDOSParameters_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetPolParameters_C",EPSEVSLSetPolParameters_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetPolParameters_C",EPSEVSLGetPolParameters_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLSetDamping_C",EPSEVSLSetDamping_EVSL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)eps,"EPSEVSLGetDamping_C",EPSEVSLGetDamping_EVSL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
