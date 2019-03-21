/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2018, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/
/*
   Full basis for the linearization of the rational approximation of non-linear eigenproblems
*/

#include <slepc/private/nepimpl.h>         /*I "slepcnep.h" I*/
#include "nleigs.h"

static PetscErrorCode MatMult_FullBasis_Sinvert(Mat M,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  NEP_NLEIGS        *ctx;
  NEP               nep;
  const PetscScalar *px;
  PetscScalar       *beta,*s,*xi,*coeffs,*t,*py,sigma;
  PetscInt          nmat,d,i,m;
  Vec               xx,xxx,yy,yyy,w,ww;

  PetscFunctionBegin;
  ierr = MatShellGetContext(M,(void**)&nep);CHKERRQ(ierr);
  ctx = (NEP_NLEIGS*)nep->data;
  beta = ctx->beta; s = ctx->s; xi = ctx->xi;
  sigma = ctx->shifts[0];
  nmat = ctx->nmat;
  d = nmat-1;
  m = nep->nloc;
  ierr = PetscMalloc2(ctx->nmat,&coeffs,ctx->nmat,&t);CHKERRQ(ierr);
  xx = ctx->w[0]; xxx = ctx->w[1]; yy = ctx->w[2]; yyy=ctx->w[3];
  w = nep->work[0]; ww = nep->work[1];
  ierr = EPSGetTarget(ctx->eps,&sigma);CHKERRQ(ierr);
  ierr = VecGetArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecGetArray(y,&py);CHKERRQ(ierr);
  ierr = VecPlaceArray(xx,px+(d-1)*m);CHKERRQ(ierr);
  ierr = VecPlaceArray(xxx,px+(d-2)*m);CHKERRQ(ierr);
  ierr = VecPlaceArray(yy,py+(d-2)*m);CHKERRQ(ierr);
  ierr = VecCopy(xxx,yy);CHKERRQ(ierr);
  ierr = VecAXPY(yy,beta[d-1]/xi[d-2],xx);CHKERRQ(ierr);
  ierr = VecScale(yy,1.0/(s[d-2]-sigma));
  ierr = VecResetArray(xx);CHKERRQ(ierr);
  ierr = VecResetArray(xxx);CHKERRQ(ierr);
  ierr = VecResetArray(yy);CHKERRQ(ierr);
  for (i=d-3;i>=0;i--) {
    ierr = VecPlaceArray(xx,px+(i+1)*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(xxx,px+i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(yy,py+i*m);CHKERRQ(ierr);
    ierr = VecPlaceArray(yyy,py+(i+1)*m);CHKERRQ(ierr);
    ierr = VecCopy(xxx,yy);CHKERRQ(ierr);
    ierr = VecAXPY(yy,beta[i+1]/xi[i],xx);CHKERRQ(ierr);
    ierr = VecAXPY(yy,-beta[i+1]*(1.0-sigma/xi[i]),yyy);CHKERRQ(ierr);
    ierr = VecScale(yy,1.0/(s[i]-sigma));
    ierr = VecResetArray(xx);CHKERRQ(ierr);
    ierr = VecResetArray(xxx);CHKERRQ(ierr);
    ierr = VecResetArray(yy);CHKERRQ(ierr);
    ierr = VecResetArray(yyy);CHKERRQ(ierr);
  }
  ierr = NEPNLEIGSEvalNRTFunct(nep,d-1,sigma,t);CHKERRQ(ierr);  
  if (nep->fui==NEP_USER_INTERFACE_SPLIT) {
    /* TODO */
  } else {
    ierr = VecPlaceArray(xx,px+(d-1)*m);CHKERRQ(ierr);
    ierr = MatMult(ctx->D[d],xx,w);CHKERRQ(ierr);
    ierr = VecScale(w,-1.0/beta[d]);CHKERRQ(ierr);
    ierr = VecResetArray(xx);CHKERRQ(ierr);
    for (i=0;i<d-1;i++) {
      ierr = VecPlaceArray(yy,py+i*m);CHKERRQ(ierr);
      ierr = MatMult(ctx->D[i],yy,ww);CHKERRQ(ierr);
      ierr = VecResetArray(yy);CHKERRQ(ierr);
      ierr = VecAXPY(w,-1.0,ww);CHKERRQ(ierr);
    }
    ierr = VecPlaceArray(yy,py+(d-1)*m);CHKERRQ(ierr);
    ierr = KSPSolve(ctx->ksp[0],w,yy);CHKERRQ(ierr);
    for (i=0;i<d-1;i++) {
      ierr = VecPlaceArray(yyy,py+i*m);CHKERRQ(ierr);
      ierr = VecAXPY(yyy,t[i],yy);CHKERRQ(ierr);
      ierr = VecResetArray(yyy);CHKERRQ(ierr);
    }
    ierr = VecScale(yy,t[d-1]);CHKERRQ(ierr);
    ierr = VecResetArray(yy);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(x,&px);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&py);CHKERRQ(ierr);
  ierr = PetscFree2(coeffs,t);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode BackTransform_FullBasis(ST st,PetscInt n,PetscScalar *eigr,PetscScalar *eigi)
{
  PetscErrorCode ierr;
  NEP            nep;

  PetscFunctionBegin;
  ierr = STShellGetContext(st,(void**)&nep);CHKERRQ(ierr);
  ierr = NEPNLEIGSBackTransform((PetscObject)nep,n,eigr,eigi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode Apply_FullBasis(ST st,Vec x,Vec y)
{
  PetscErrorCode ierr;
  NEP            nep;
  NEP_NLEIGS     *ctx;

  PetscFunctionBegin;
  ierr = STShellGetContext(st,(void**)&nep);CHKERRQ(ierr);
  ctx = (NEP_NLEIGS*)nep->data;
  ierr = MatMult(ctx->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSetUp_NLEIGS_FullBasis(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;
  ST             st;
  Mat            Q;
  PetscInt       i=0,deg=ctx->nmat-1;
  PetscBool      trackall,istrivial,ks;
  PetscScalar    *epsarray,*neparray;
  Vec            veps,w=NULL;

  PetscFunctionBegin;
  if (!ctx->eps) { ierr = NEPNLEIGSGetEPS(nep,&ctx->eps);CHKERRQ(ierr); }
  ierr = EPSGetST(ctx->eps,&st);CHKERRQ(ierr);
  ierr = EPSSetTarget(ctx->eps,nep->target);CHKERRQ(ierr);
  ierr = STSetDefaultShift(st,nep->target);CHKERRQ(ierr);
  if (!((PetscObject)(ctx->eps))->type_name) {
    ierr = EPSSetType(ctx->eps,EPSKRYLOVSCHUR);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectTypeCompare((PetscObject)ctx->eps,EPSKRYLOVSCHUR,&ks);CHKERRQ(ierr);
    if (!ks) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Full-basis option only implemented for Krylov-Schur");
  }
  ierr = STSetType(st,STSHELL);CHKERRQ(ierr);
  ierr = STShellSetContext(st,(PetscObject)nep);CHKERRQ(ierr);
  ierr = STShellSetBackTransform(st,BackTransform_FullBasis);CHKERRQ(ierr);
  ierr = KSPGetOperators(ctx->ksp[0],&Q,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecsEmpty(Q,&ctx->w[0],&ctx->w[1]);CHKERRQ(ierr);
  ierr = MatCreateVecsEmpty(Q,&ctx->w[2],&ctx->w[3]);CHKERRQ(ierr);
  ierr = PetscLogObjectParents(nep,6,ctx->w);CHKERRQ(ierr);
  ierr = MatCreateShell(PetscObjectComm((PetscObject)nep),deg*nep->nloc,deg*nep->nloc,deg*nep->n,deg*nep->n,nep,&ctx->A);CHKERRQ(ierr);
  ierr = MatShellSetOperation(ctx->A,MATOP_MULT,(void(*)(void))MatMult_FullBasis_Sinvert);CHKERRQ(ierr);
  ierr = STShellSetApply(st,Apply_FullBasis);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->A);CHKERRQ(ierr);
  ierr = EPSSetOperators(ctx->eps,ctx->A,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(ctx->eps,EPS_NHEP);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(ctx->eps,EPS_LARGEST_MAGNITUDE);CHKERRQ(ierr);
  ierr = RGIsTrivial(nep->rg,&istrivial);CHKERRQ(ierr);
  if (!istrivial) { ierr = EPSSetRG(ctx->eps,nep->rg);CHKERRQ(ierr);}
  ierr = EPSSetDimensions(ctx->eps,nep->nev,nep->ncv?nep->ncv:PETSC_DEFAULT,nep->mpd?nep->mpd:PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = EPSSetTolerances(ctx->eps,nep->tol==PETSC_DEFAULT?SLEPC_DEFAULT_TOL:nep->tol,nep->max_it?nep->max_it:PETSC_DEFAULT);CHKERRQ(ierr);
  /* Transfer the trackall option from pep to eps */
  ierr = NEPGetTrackAll(nep,&trackall);CHKERRQ(ierr);
  ierr = EPSSetTrackAll(ctx->eps,trackall);CHKERRQ(ierr);

  /* process initial vector */
  if (nep->nini<0) {
    ierr = VecCreateMPI(PetscObjectComm((PetscObject)ctx->eps),deg*nep->nloc,deg*nep->n,&veps);CHKERRQ(ierr);
    ierr = VecGetArray(veps,&epsarray);CHKERRQ(ierr);
    for (i=0;i<deg;i++) {
      if (i<-nep->nini) {
        ierr = VecGetArray(nep->IS[i],&neparray);CHKERRQ(ierr);
        ierr = PetscMemcpy(epsarray+i*nep->nloc,neparray,nep->nloc*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = VecRestoreArray(nep->IS[i],&neparray);CHKERRQ(ierr);
      } else {
        if (!w) { ierr = VecDuplicate(nep->IS[0],&w);CHKERRQ(ierr); }
        ierr = VecSetRandom(w,NULL);CHKERRQ(ierr);
        ierr = VecGetArray(w,&neparray);CHKERRQ(ierr);
        ierr = PetscMemcpy(epsarray+i*nep->nloc,neparray,nep->nloc*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = VecRestoreArray(w,&neparray);CHKERRQ(ierr);
      }
    }
    ierr = VecRestoreArray(veps,&epsarray);CHKERRQ(ierr);
    ierr = EPSSetInitialSpace(ctx->eps,1,&veps);CHKERRQ(ierr);
    ierr = VecDestroy(&veps);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = SlepcBasisDestroy_Private(&nep->nini,&nep->IS);CHKERRQ(ierr);
  }

  ierr = EPSSetUp(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetDimensions(ctx->eps,NULL,&nep->ncv,&nep->mpd);CHKERRQ(ierr);
  ierr = EPSGetTolerances(ctx->eps,NULL,&nep->max_it);CHKERRQ(ierr);
  ierr = NEPAllocateSolution(nep,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   NEPNLEIGSExtract_None - Extracts the first block of the basis
   and normalizes the columns.
*/
static PetscErrorCode NEPNLEIGSExtract_None(NEP nep,EPS eps)
{
  PetscErrorCode    ierr;
  PetscInt          i;
  const PetscScalar *px;
  Mat               A;
  Vec               xr,xi=NULL,w;
  PetscReal         norm;

  PetscFunctionBegin;
  ierr = EPSGetOperators(eps,&A,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&xr,NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDuplicate(xr,&xi);CHKERRQ(ierr);
#endif
  w = nep->work[0];
  for (i=0;i<nep->nconv;i++) {
    ierr = EPSGetEigenvector(eps,i,xr,xi);CHKERRQ(ierr);
    ierr = VecGetArrayRead(xr,&px);CHKERRQ(ierr);
    ierr = VecPlaceArray(w,px);CHKERRQ(ierr);
    ierr = BVInsertVec(nep->V,i,w);CHKERRQ(ierr);
    ierr = BVNormColumn(nep->V,i,NORM_2,&norm);CHKERRQ(ierr);
    ierr = BVScaleColumn(nep->V,i,1.0/norm);CHKERRQ(ierr);
    ierr = VecResetArray(w);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(xr,&px);CHKERRQ(ierr);
  }
  ierr = VecDestroy(&xr);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  ierr = VecDestroy(&xi);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode NEPSolve_NLEIGS_FullBasis(NEP nep)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx = (NEP_NLEIGS*)nep->data;
  PetscInt       i;
  PetscScalar    eigi=0.0;

  PetscFunctionBegin;
  ierr = EPSSolve(ctx->eps);CHKERRQ(ierr);
  ierr = EPSGetConverged(ctx->eps,&nep->nconv);CHKERRQ(ierr);
  ierr = EPSGetIterationNumber(ctx->eps,&nep->its);CHKERRQ(ierr);
  ierr = EPSGetConvergedReason(ctx->eps,(EPSConvergedReason*)&nep->reason);CHKERRQ(ierr);

  /* recover eigenvalues */
  for (i=0;i<nep->nconv;i++) { 
    ierr = EPSGetEigenpair(ctx->eps,i,&nep->eigr[i],&eigi,NULL,NULL);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
    if (eigi!=0.0) SETERRQ(PetscObjectComm((PetscObject)nep),PETSC_ERR_SUP,"Complex value requires complex arithmetic");
#endif 
  }
  ierr = NEPNLEIGSExtract_None(nep,ctx->eps);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPNLEIGSSetEPS_NLEIGS(NEP nep,EPS eps)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)eps);CHKERRQ(ierr);
  ierr = EPSDestroy(&ctx->eps);CHKERRQ(ierr);
  ctx->eps = eps;
  ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps);CHKERRQ(ierr);
  nep->state = NEP_STATE_INITIAL;
  PetscFunctionReturn(0);
}

/*@
   NEPNLEIGSSetEPS - Associate an eigensolver object (EPS) to the NLEIGS solver.

   Collective on NEP

   Input Parameters:
+  nep - nonlinear eigenvalue solver
-  eps - the eigensolver object

   Level: advanced

.seealso: NEPNLEIGSGetEPS()
@*/
PetscErrorCode NEPNLEIGSSetEPS(NEP nep,EPS eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidHeaderSpecific(eps,EPS_CLASSID,2);
  PetscCheckSameComm(nep,1,eps,2);
  ierr = PetscTryMethod(nep,"NEPNLEIGSSetEPS_C",(NEP,EPS),(nep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode EPSMonitor_NLEIGS(EPS eps,PetscInt its,PetscInt nconv,PetscScalar *eigr,PetscScalar *eigi,PetscReal *errest,PetscInt nest,void *ctx)
{
  NEP            nep = (NEP)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = NEPMonitor(nep,its,nconv,eigr,eigi,errest,nest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode NEPNLEIGSGetEPS_NLEIGS(NEP nep,EPS *eps)
{
  PetscErrorCode ierr;
  NEP_NLEIGS     *ctx=(NEP_NLEIGS*)nep->data;

  PetscFunctionBegin;
  if (!ctx->eps) {
    ierr = EPSCreate(PetscObjectComm((PetscObject)nep),&ctx->eps);CHKERRQ(ierr);
    ierr = PetscObjectIncrementTabLevel((PetscObject)ctx->eps,(PetscObject)nep,1);CHKERRQ(ierr);
    ierr = EPSSetOptionsPrefix(ctx->eps,((PetscObject)nep)->prefix);CHKERRQ(ierr);
    ierr = EPSAppendOptionsPrefix(ctx->eps,"nep_nleigs_");CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)nep,(PetscObject)ctx->eps);CHKERRQ(ierr);
    ierr = PetscObjectSetOptions((PetscObject)ctx->eps,((PetscObject)nep)->options);CHKERRQ(ierr);
    ierr = EPSMonitorSet(ctx->eps,EPSMonitor_NLEIGS,nep,NULL);CHKERRQ(ierr);
  }
  *eps = ctx->eps;
  PetscFunctionReturn(0);
}

/*@
   NEPNLEIGSGetEPS - Retrieve the eigensolver object (EPS) associated
   to the nonlinear eigenvalue solver.

   Not Collective

   Input Parameter:
.  nep - nonlinear eigenvalue solver

   Output Parameter:
.  eps - the eigensolver object

   Level: advanced

.seealso: NEPNLEIGSSetEPS()
@*/
PetscErrorCode NEPNLEIGSGetEPS(NEP nep,EPS *eps)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(nep,NEP_CLASSID,1);
  PetscValidPointer(eps,2);
  ierr = PetscUseMethod(nep,"NEPNLEIGSGetEPS_C",(NEP,EPS*),(nep,eps));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

