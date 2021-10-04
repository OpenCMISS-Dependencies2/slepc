/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   SLEPc singular value solver: "randomized"

   Method: RSVD

   Algorithm:

       Randomized singular value decomposition.

   References:

       [1] N. Halko, P.-G. Martinsson, and J. A. Tropp, "Finding
           structure with randomness: Probabilistic algorithms for
           constructing approximate matrix decompositions", SIAM Rev.,
           53(2):217-288, 2011.

       [2] P.-G. Martinsson, and S. Voronin, "A Randomized Blocked Algorithm
           for Efficiently Computing Rank-revealing Factorizations of Matrices"
           SIAM J. Sci. Comput., 38(5):S485–S507, 2016.

*/

#include <slepc/private/svdimpl.h>                /*I "slepcsvd.h" I*/

typedef struct {
  PetscInt q; /* steps for power iteration */

  /* fixed precision version?  */
  PetscBool fixedprecision;

  /* adaptive rank (fixed precision) */
  PetscInt  r;
  PetscInt  blocksize;
  PetscReal eps;
} SVD_RANDOMIZED;

PetscErrorCode SVDSetUp_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  PetscInt       M,N;

  PetscFunctionBegin;
  if (svd->which!=SVD_LARGEST) SETERRQ(PetscObjectComm((PetscObject)svd),PETSC_ERR_SUP,"This solver supports only largest singular values");
  ierr = MatGetSize(svd->A,&M,&N);CHKERRQ(ierr);
  if (svd->ncv==PETSC_DEFAULT) svd->ncv = PetscMin(svd->nsv + 20,PetscMin(M,N)); /* 20 as generous default oversampling */
  ierr = SVDSetDimensions_Default(svd);CHKERRQ(ierr);
  if (svd->max_it==PETSC_DEFAULT) svd->max_it = PetscMax(N/svd->ncv,100);
  svd->leftbasis = PETSC_TRUE;
  svd->mpd = svd->ncv;
  ierr = SVDAllocateSolution(svd,0);CHKERRQ(ierr);
  ierr = DSSetType(svd->ds,DSSVD);CHKERRQ(ierr);
  ierr = DSAllocate(svd->ds,svd->ncv);CHKERRQ(ierr);
  ierr = SVDSetWorkVecs(svd,1,1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDRandomizedResidualNorm(SVD svd,PetscInt i,PetscScalar sigma,PetscReal *res)
{
  PetscErrorCode ierr;
  PetscReal      norm1,norm2;
  Vec            u,v,wu,wv;

  PetscFunctionBegin;
  wu = svd->swapped? svd->workr[0]: svd->workl[0];
  wv = svd->swapped? svd->workl[0]: svd->workr[0];
  if (svd->conv!=SVD_CONV_MAXIT) {
    ierr = BVGetColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVGetColumn(svd->U,i,&u);CHKERRQ(ierr);
    /* norm1 = ||A*v-sigma*u||_2 */
    ierr = MatMult(svd->A,v,wu);CHKERRQ(ierr);
    ierr = VecAXPY(wu,-sigma,u);CHKERRQ(ierr);
    ierr = VecNorm(wu,NORM_2,&norm1);CHKERRQ(ierr);
    /* norm2 = ||A^T*u-sigma*v||_2 */
    ierr = MatMult(svd->AT,u,wv);CHKERRQ(ierr);
    ierr = VecAXPY(wv,-sigma,v);CHKERRQ(ierr);
    ierr = VecNorm(wv,NORM_2,&norm2);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->V,i,&v);CHKERRQ(ierr);
    ierr = BVRestoreColumn(svd->U,i,&u);CHKERRQ(ierr);
    *res = PetscSqrtReal(norm1*norm1+norm2*norm2);
  } else {
    *res = 1.0;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDSolve_Randomized_Subspace(SVD svd)
{
  PetscErrorCode ierr;
  PetscScalar    *w;
  PetscReal      res=1.0;
  PetscInt       q,i,k=0;
  Mat            A,U,V;
  SVD_RANDOMIZED *svdr = (SVD_RANDOMIZED*)svd->data;

  PetscFunctionBegin;
  /* Form random matrix, G. Complete the initial basis with random vectors */
  ierr = BVSetActiveColumns(svd->V,svd->nini,svd->ncv);CHKERRQ(ierr);
  ierr = BVSetRandomNormal(svd->V);CHKERRQ(ierr);
  ierr = PetscCalloc1(svd->ncv,&w);CHKERRQ(ierr);

  /* Subspace Iteration */
  do {
    svd->its++;
    ierr = BVSetActiveColumns(svd->V,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,svd->nconv,svd->ncv);CHKERRQ(ierr);
    /* Form AG */
    ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
    /* Orthogonalization Q=qr(AG)*/
    ierr = BVOrthogonalize(svd->U,NULL);CHKERRQ(ierr);
    /* Power iteration */
    for (q = 0; q < svdr->q; q++) {
      ierr = BVMatMult(svd->U,svd->AT,svd->V);CHKERRQ(ierr);
      ierr = BVOrthogonalize(svd->V,NULL);CHKERRQ(ierr);
      ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
      ierr = BVOrthogonalize(svd->U,NULL);CHKERRQ(ierr);
    }
    /* Form B^*= A^*Q */
    ierr = BVMatMult(svd->U,svd->AT,svd->V);CHKERRQ(ierr);

    ierr = DSSetDimensions(svd->ds,svd->ncv,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = DSSVDSetDimensions(svd->ds,svd->ncv);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = MatZeroEntries(A);CHKERRQ(ierr);
    ierr = BVOrthogonalize(svd->V,A);CHKERRQ(ierr);
    ierr = DSRestoreMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
    ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
    ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
    ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
    ierr = DSGetMat(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->U,V,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = BVMultInPlace(svd->V,U,svd->nconv,svd->ncv);CHKERRQ(ierr);
    ierr = MatDestroy(&U);CHKERRQ(ierr);
    ierr = MatDestroy(&V);CHKERRQ(ierr);
    /* Check convergence */
    k = 0;
    for (i=svd->nconv;i<svd->ncv;i++) {
      res = 0.0;
      svd->sigma[i] = PetscRealPart(w[i]);
      if (svd->its < svd->max_it) {
        ierr = SVDRandomizedResidualNorm(svd,i,w[i],&res);CHKERRQ(ierr);
      }
      ierr = (*svd->converged)(svd,svd->sigma[i],res,&svd->errest[i],svd->convergedctx);CHKERRQ(ierr);
      if (svd->errest[i] < svd->tol) k++;
      else break;
    }
    if (svd->conv == SVD_CONV_MAXIT && svd->its >= svd->max_it) {
      k = svd->nsv;
      for (i=0;i<svd->ncv;i++) svd->sigma[i] = PetscRealPart(w[i]);
    }
    ierr = (*svd->stopping)(svd,svd->its,svd->max_it,svd->nconv+k,svd->nsv,&svd->reason,svd->stoppingctx);CHKERRQ(ierr);
    svd->nconv += k;
    ierr = SVDMonitor(svd,svd->its,svd->nconv,svd->sigma,svd->errest,svd->ncv);CHKERRQ(ierr);
  } while (svd->reason == SVD_CONVERGED_ITERATING);
  ierr = PetscFree(w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SVDSolve_Randomized_Adaptive(SVD svd)
{
  PetscErrorCode    ierr;
  PetscScalar       *w;
  PetscReal         res=1.0;
  PetscInt          q,i,k=0;
  Mat               A,U,V;
  SVD_RANDOMIZED    *svdr = (SVD_RANDOMIZED*)svd->data;
  PetscReal         scale = 10 * PetscSqrtReal(2.0/PETSC_PI);
  PetscInt          b = svdr->blocksize, r = svdr->r;
  PetscReal         eps = svdr->eps;
  PetscReal         me = 0.0;
  Mat               MW,RW;
  BV                W,AW;
  const PetscScalar *rw;
  PetscRandom       rand;

  PetscFunctionBegin;
  /* Form random matrix, G. Complete the initial basis with random vectors */
  ierr = BVSetActiveColumns(svd->V,svd->nini,svd->ncv);CHKERRQ(ierr);
  ierr = BVSetRandomNormal(svd->V);CHKERRQ(ierr);
  ierr = PetscCalloc1(svd->ncv,&w);CHKERRQ(ierr);

  /* Initialize error indicator AW -> They must share the same random number generator */
  ierr = BVGetRandomContext(svd->V,&rand);CHKERRQ(ierr);
  ierr = BVDuplicateResize(svd->V,r,&W);CHKERRQ(ierr);
  ierr = BVDuplicateResize(svd->U,r,&AW);CHKERRQ(ierr);
  ierr = BVSetRandomContext(W,rand);CHKERRQ(ierr);
  ierr = BVSetRandomNormal(W);CHKERRQ(ierr);
  ierr = BVMatMult(W,svd->A,AW);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,svd->ncv,r,NULL,&MW);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,svd->ncv,svd->ncv,NULL,&RW);CHKERRQ(ierr);

  do { /* Move in batches */
    svd->its++;
    b = PetscMin(b,svd->ncv - svd->nconv);
    ierr = BVSetActiveColumns(svd->V,svd->nconv,svd->nconv + b);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,svd->nconv,svd->nconv + b);CHKERRQ(ierr);

    /* A * Omega */
    ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
    /* Q = Orth (A * Omega) */
    ierr = BVOrthogonalize(svd->U,RW);CHKERRQ(ierr);
    /* Power iteration */
    for (q = 0; q < svdr->q; q++) {
      ierr = BVMatMult(svd->U,svd->AT,svd->V);CHKERRQ(ierr);
      ierr = BVOrthogonalize(svd->V,NULL);CHKERRQ(ierr);
      ierr = BVMatMult(svd->V,svd->A,svd->U);CHKERRQ(ierr);
      ierr = BVOrthogonalize(svd->U,RW);CHKERRQ(ierr);
    }
    /* XXX Does BV guarantee I will not need to perform step 8 in randQB_pb, Fig 4 of [2]? */

    /* XXX filter out linearly dependent vectors? */
    ierr = MatDenseGetArrayRead(RW,&rw);CHKERRQ(ierr);
    for (i = svd->nconv, k = 0; i < svd->nconv + b; i++) {
      if (rw[i*(svd->ncv +1)] > PETSC_SMALL) k++;
    }
    ierr = MatDenseRestoreArrayRead(RW,&rw);CHKERRQ(ierr);

    /* B^* = A^* * Q */
    ierr = BVSetActiveColumns(svd->V,svd->nconv,svd->nconv + k);CHKERRQ(ierr);
    ierr = BVSetActiveColumns(svd->U,svd->nconv,svd->nconv + k);CHKERRQ(ierr);
    ierr = BVMatMult(svd->U,svd->AT,svd->V);CHKERRQ(ierr);
    if (k != b && svdr->q) { /* rank deficient with power iteration, restore random V columns */
      ierr = BVSetActiveColumns(svd->V,svd->nconv,svd->nconv + b);CHKERRQ(ierr);
      ierr = BVSetRandomNormal(svd->V);CHKERRQ(ierr);
    }

    /* Update error indicator */
    ierr = BVDot(AW,svd->U,MW);CHKERRQ(ierr);
    ierr = BVMult(AW,-1.0,1.0,svd->U,MW);CHKERRQ(ierr);

    /* Randomized norm */
    me = 0.0;
    for (i = 0; i < r; i++) {
      PetscReal err;
      ierr = BVNormColumn(AW,i,NORM_2,&err);CHKERRQ(ierr);
      printf("  ERR %d %g -> %g\n",i,err,err*scale);
      me = PetscMax(me,err);
    }
    printf("  ME %g*%g < %g ? %d\n",me,scale,eps,me * scale < eps);
    printf("  STEP %d CONV %d NVC %d B %d K %d\n",svd->its,svd->nconv,svd->ncv,b,k);
    svd->nconv += k;
    if (me * scale < eps) svd->reason = SVD_CONVERGED_TOL;
    /* XXX Monitor and stopping? */
  } while (svd->reason == SVD_CONVERGED_ITERATING && svd->nconv < svd->ncv);
  ierr = BVDestroy(&W);CHKERRQ(ierr);
  ierr = BVDestroy(&AW);CHKERRQ(ierr);

  /* Compute SVD of B and update U and V (XXX is DS setup properly?) */
  ierr = BVSetActiveColumns(svd->V,0,svd->nconv);CHKERRQ(ierr);
  ierr = BVSetActiveColumns(svd->U,0,svd->nconv);CHKERRQ(ierr);
  ierr = DSSetDimensions(svd->ds,svd->nconv,0,0);CHKERRQ(ierr);
  ierr = DSSVDSetDimensions(svd->ds,svd->nconv);CHKERRQ(ierr);
  ierr = DSGetMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = BVOrthogonalize(svd->V,A);CHKERRQ(ierr);
  ierr = DSRestoreMat(svd->ds,DS_MAT_A,&A);CHKERRQ(ierr);
  ierr = DSSetState(svd->ds,DS_STATE_RAW);CHKERRQ(ierr);
  ierr = DSSolve(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSSort(svd->ds,w,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DSSynchronize(svd->ds,w,NULL);CHKERRQ(ierr);
  ierr = DSGetMat(svd->ds,DS_MAT_U,&U);CHKERRQ(ierr);
  ierr = DSGetMat(svd->ds,DS_MAT_V,&V);CHKERRQ(ierr);
  ierr = BVMultInPlace(svd->U,V,0,svd->nconv);CHKERRQ(ierr);
  ierr = BVMultInPlace(svd->V,U,0,svd->nconv);CHKERRQ(ierr);
  ierr = MatDestroy(&U);CHKERRQ(ierr);
  ierr = MatDestroy(&V);CHKERRQ(ierr);

  for (i = 0; i < svd->nconv; i++) {
    svd->sigma[i] = PetscRealPart(w[i]);
    ierr = SVDRandomizedResidualNorm(svd,i,w[i],&res);CHKERRQ(ierr);
    printf("SIGMA/RES %d %g %g\n",i,w[i],res);
  }
  ierr = PetscFree(w);CHKERRQ(ierr);
  ierr = MatDestroy(&MW);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDSolve_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  SVD_RANDOMIZED *svdr = (SVD_RANDOMIZED*)svd->data;

  PetscFunctionBegin;
  if (svdr->fixedprecision) {
    ierr = SVDSolve_Randomized_Adaptive(svd);CHKERRQ(ierr);
  } else {
    ierr = SVDSolve_Randomized_Subspace(svd);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* XXX Missing API, please add it as you wish */

PetscErrorCode SVDSetFromOptions_Randomized(PetscOptionItems *PetscOptionsObject,SVD svd)
{
  PetscErrorCode ierr;
  SVD_RANDOMIZED *svdr = (SVD_RANDOMIZED*)svd->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"SVD Randomized options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-svd_randomized_psteps","Number of steps of the power iteration method","",svdr->q,&svdr->q,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-svd_randomized_fixedprecision","Use the fixed precision version","",svdr->fixedprecision,&svdr->fixedprecision,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-svd_randomized_eps","Tolerance for the fixed precision version","",svdr->eps,&svdr->eps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-svd_randomized_bs","Blocksize for the fixed precision blocking version","",svdr->blocksize,&svdr->blocksize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-svd_randomized_r","Number of random samples used in norm estimation for the fixed precision version","",svdr->r,&svdr->r,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SVDDestroy_Randomized(SVD svd)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(svd->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

SLEPC_EXTERN PetscErrorCode SVDCreate_Randomized(SVD svd)
{
  PetscErrorCode ierr;
  SVD_RANDOMIZED *svdr;

  PetscFunctionBegin;
  ierr = PetscNewLog(svd,&svdr);CHKERRQ(ierr);
  svd->data = (void*)svdr;

  /* XXX Some defaults? */
  svdr->r = 5;
  svdr->blocksize = 16;
  svdr->eps = 1.e-4;

  /* XXX one would argue that these randomized methods are used when memory is a precious commodity:
     do not explicitly create the transposed matrix */
  svd->impltrans = PETSC_TRUE;

  svd->ops->setup          = SVDSetUp_Randomized;
  svd->ops->solve          = SVDSolve_Randomized;
  svd->ops->setfromoptions = SVDSetFromOptions_Randomized;
  svd->ops->destroy        = SVDDestroy_Randomized;
  PetscFunctionReturn(0);
}

