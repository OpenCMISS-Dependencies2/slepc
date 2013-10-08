/*

   SLEPc quadratic eigensolver: "stoar"

   Method: S-TOAR

   Algorithm:

       Symmetric Two-Level Orthogonal Arnoldi.

   References:

       [1] C. Campos and J.E. Roman, "A thick-restart Q-Lanczos method
           for quadratic eigenvalue problems", in preparation, 2013.

   Last update: Oct 2013

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#include <slepc-private/qepimpl.h>         /*I "slepcqep.h" I*/
#include <slepcblaslapack.h>

typedef struct {
  PetscScalar  *S; 
  PetscInt     d;
  PetscInt     ld;
  PetscScalar  *qK;
  PetscReal    *qM;
} QEP_STOAR;

#undef __FUNCT__
#define __FUNCT__ "QEPSetUp_STOAR"
PetscErrorCode QEPSetUp_STOAR(QEP qep)
{
  PetscErrorCode ierr;
  PetscBool      sinv;
  QEP_STOAR      *ctx;
  ST             st;
  Mat            M;
  PetscInt       ld;
  IP             ip;

  PetscFunctionBegin;
  if (qep->ncv) { /* ncv set */
    if (qep->ncv<qep->nev) SETERRQ(PetscObjectComm((PetscObject)qep),1,"The value of ncv must be at least nev");
  } else if (qep->mpd) { /* mpd set */
    qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd);
  } else { /* neither set: defaults depend on nev being small or large */
    if (qep->nev<500) qep->ncv = PetscMin(qep->n,PetscMax(2*qep->nev,qep->nev+15));
    else {
      qep->mpd = 500;
      qep->ncv = PetscMin(qep->n,qep->nev+qep->mpd);
    }
  }
  if (!qep->mpd) qep->mpd = qep->ncv;
  if (qep->ncv>qep->nev+qep->mpd) SETERRQ(PetscObjectComm((PetscObject)qep),1,"The value of ncv must not be larger than nev+mpd");
  if (!qep->max_it) qep->max_it = PetscMax(100,2*qep->n/qep->ncv); /* -- */
  if (!qep->which) {
    ierr = PetscObjectTypeCompare((PetscObject)qep->st,STSINVERT,&sinv);CHKERRQ(ierr);
    if (sinv) qep->which = QEP_TARGET_MAGNITUDE;
    else qep->which = QEP_LARGEST_MAGNITUDE;
  }
  if (qep->problem_type!=QEP_HERMITIAN) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_SUP,"Requested method is only available for Hermitian problems");
  if (qep->sfactor_set && qep->sfactor!=1.0) SETERRQ(PetscObjectComm((PetscObject)qep),PETSC_ERR_SUP,"Requested method is not jet available with scaling");
  else qep->sfactor = 1.0;
  ierr = QEPAllocateSolution(qep,2);CHKERRQ(ierr);
  ierr = QEPSetWorkVecs(qep,4);CHKERRQ(ierr);
  ld = qep->ncv+2;
  ierr = DSSetType(qep->ds,DSGHIEP);CHKERRQ(ierr);
  ierr = DSSetCompact(qep->ds,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DSAllocate(qep->ds,ld);CHKERRQ(ierr);
  ierr = PetscNewLog(qep,QEP_STOAR,&ctx);CHKERRQ(ierr);
  ierr = QEPGetIP(qep,&ip);CHKERRQ(ierr);
  ierr = IPSetType(ip,IPINDEFINITE);CHKERRQ(ierr);
  ierr = QEPGetST(qep,&st);CHKERRQ(ierr);
  ierr = STSetUp(st);CHKERRQ(ierr);
  ierr = STGetNumMatrices(st,&ctx->d);CHKERRQ(ierr);
  ctx->d--;
  ctx->ld = ld;
  ierr = STGetBilinearForm(st,&M);CHKERRQ(ierr);
  ierr = IPSetMatrix(qep->ip,M);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = PetscMalloc(ctx->d*ld*ld*sizeof(PetscScalar),&ctx->S);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->S,ctx->d*ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscMalloc(ld*sizeof(PetscReal),&ctx->qM);CHKERRQ(ierr);
  ierr = PetscMalloc(ld*ld*sizeof(PetscScalar),&ctx->qK);CHKERRQ(ierr);
  ierr = PetscMemzero(ctx->qK,ld*ld*sizeof(PetscScalar));CHKERRQ(ierr);
  qep->data = ctx;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPSTOARNorm"
/*
  Compute B-norm of v=[v1;v2] whith  B=diag(-qep->T[0],qep->T[2]) 
*/
static PetscErrorCode QEPSTOARNorm(QEP qep,PetscInt j,PetscReal *norm,PetscScalar *w,PetscInt lw)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx = (QEP_STOAR*)qep->data;
  PetscBLASInt   n_,one=1,ld_;
  PetscScalar    sone=1.0,szero=0.0,*sp,*sq;
  PetscInt       lwa,n,i,lds=ctx->d*ctx->ld;

  PetscFunctionBegin;
  n = j+2;
  lwa = n;
  if (lw<lwa) SETERRQ1(PETSC_COMM_SELF,1,"Wrong value of lw",lw);
  sp = ctx->S+lds*j;
  sq = sp+ctx->ld;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,sp,&one,&szero,w,&one));
  *norm = 0.0;
  for (i=0;i<n;i++) *norm += PetscRealPart(w[i]*sp[i]+sq[i]*sq[i]*(*(ctx->qM+i)));
  *norm = (*norm>0.0)?PetscSqrtReal(*norm):-PetscSqrtReal(-*norm);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPSTOAROrth2"
/*
 Computes GS orthogonalization  x = [z;x] - [Sp;Sq]*y,
 where y = Omega\([Sp;Sq]'*[qK zeros(size(qK,1)) ;zeros(size(qK,1)) qM]*[z;x]).
 n: Column from S to be orthogonalized against previous columns.
*/
static PetscErrorCode QEPSTOAROrth2(QEP qep,PetscInt k,PetscReal *Omega,PetscScalar *y,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx=(QEP_STOAR*)qep->data;
  PetscBLASInt   n_,lds_,k_,one=1,ld_;
  PetscScalar    *S=ctx->S,sonem=-1.0,sone=1.0,szero=0.0,*tp,*tq,*xp,*xq,*c;
  PetscInt       lwa,nwu=0,i,lds=ctx->d*ctx->ld,n;
  
  PetscFunctionBegin;
  n = k+1;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr); /* Size of qK and qM */
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(k,&k_);CHKERRQ(ierr); /* Number of vectors to orthogonalize against them */
  lwa = 3*n;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
  tp = work+nwu;
  nwu += n;
  tq = work+nwu;
  nwu += n;
  c = work+nwu;
  nwu += k;
  xp = S+k*lds;
  xq = S+ctx->ld+k*lds;
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,xp,&one,&szero,tp,&one));
  for (i=0;i<n;i++) tq[i] = *(ctx->qM+i)*xq[i];
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,y,&one));
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,y,&one));
  for (i=0;i<n-1;i++) y[i] /= Omega[i];
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,y,&one,&sone,xp,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,y,&one,&sone,xq,&one));
  /* twice */
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&n_,&sone,ctx->qK,&ld_,xp,&one,&szero,tp,&one));
  for (i=0;i<n;i++) tq[i] = *(ctx->qM+i)*xq[i];
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,ctx->S,&lds_,tp,&one,&szero,c,&one));
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&k_,&sone,S+ctx->ld,&lds_,tq,&one,&sone,c,&one));
  for (i=0;i<k;i++) c[i] /= Omega[i];
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S,&lds_,c,&one,&sone,xp,&one));
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&k_,&sonem,S+ctx->ld,&lds_,c,&one,&sone,xq,&one));
  for (i=0;i<k;i++) y[i] += c[i];
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSTOARqKupdate"
PetscErrorCode QEPSTOARqKupdate(QEP qep,PetscInt j,Vec wv)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx=(QEP_STOAR*)qep->data;
  PetscInt       i,ld=ctx->ld;
  PetscScalar    *qK=ctx->qK;
  Vec            *V = qep->V;

  PetscFunctionBegin;
  if (!wv) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",3);
  ierr = STMatMult(qep->st,0,V[j],wv);CHKERRQ(ierr);
  ierr = VecMDot(wv,j+1,V,qK+j*ld);CHKERRQ(ierr);
  for (i=0;i<=j;i++) {
    qK[i+j*ld] = -PetscConj(qK[i+j*ld]);
    qK[j+i*ld] = qK[i+ld*j];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSTOARrun"
/*
  Compute a run of Lanczos iterations
*/
static PetscErrorCode QEPSTOARrun(QEP qep,PetscReal *a,PetscReal *b,PetscReal *omega,Vec *V,PetscInt k,PetscInt *M,PetscBool *breakdown,PetscScalar *work,PetscInt nw,Vec *t_,PetscInt nwv)
{
  PetscErrorCode     ierr;
  QEP_STOAR          *ctx=(QEP_STOAR*)qep->data;
  PetscInt           i,j,m=*M,nwu=0,lwa;
  PetscInt           lds=ctx->d*ctx->ld,offq=ctx->ld;
  Vec                v=t_[0],t=t_[1],q=t_[2];
  PetscReal          norm;
  PetscScalar        *y,*S=ctx->S;

  PetscFunctionBegin;
  *breakdown = PETSC_FALSE; /* ////////////// */
  if (!t_||nwv<3) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",12);
  lwa = (ctx->ld)*4;
  if (!work||nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
  y = work;
  nwu += ctx->ld;
  for (j=k;j<m;j++) {
    /* apply operator */
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    ierr = VecMAXPY(v,j+2,S+j*lds,V);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,0,v,t);CHKERRQ(ierr);
    ierr = VecZeroEntries(v);CHKERRQ(ierr);
    ierr = VecMAXPY(v,j+2,S+offq+j*lds,V);CHKERRQ(ierr);
    ierr = STMatMult(qep->st,1,v,q);CHKERRQ(ierr);
    ierr = VecAXPY(t,qep->sfactor,q);CHKERRQ(ierr);
    ierr = STMatSolve(qep->st,2,t,q);CHKERRQ(ierr);
    ierr = VecScale(q,-1.0/(qep->sfactor*qep->sfactor));CHKERRQ(ierr);

    /* orthogonalize */
    ierr = IPPseudoOrthogonalize(qep->ip,j+2,qep->V,ctx->qM,q,S+offq+(j+1)*lds,&norm,NULL);CHKERRQ(ierr);
    for (i=0;i<j+2;i++) *(S+offq+(j+1)*lds+i) *= *(ctx->qM+i);
    *(S+offq+(j+1)*lds+j+2) = norm;
    ierr = VecScale(q,1.0/norm);CHKERRQ(ierr);
    ierr = VecCopy(q,V[j+2]);CHKERRQ(ierr);
    for (i=0;i<=j+1;i++) *(S+(j+1)*lds+i) = *(S+offq+j*lds+i);
   
    /* Update qK and qM */
    *(ctx->qM+j+2) = (norm > 0)?1.0:-1.0;
    ierr = QEPSTOARqKupdate(qep,j+2,t);CHKERRQ(ierr);

    /* Level-2 orthogonalization */
    ierr = QEPSTOAROrth2(qep,j+1,omega,y,work+nwu,lwa-nwu);CHKERRQ(ierr);
    a[j] = PetscRealPart(y[j])/omega[j];
    ierr = QEPSTOARNorm(qep,j+1,&norm,work+nwu,lwa-nwu);CHKERRQ(ierr);
    omega[j+1] = (norm > 0)?1.0:-1.0;
    for (i=0;i<=j+2;i++) {
      S[i+(j+1)*lds] /= norm;
      S[i+offq+(j+1)*lds] /= norm;
    }
    b[j] = PetscAbsReal(norm);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSTOARTrunc2"
PetscErrorCode QEPSTOARTrunc2(QEP qep,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx=(QEP_STOAR*)qep->data;
  PetscInt       lwa,nwu=0,lrwa,nrwu=0,j,i,off1,off2,lds=ctx->ld*ctx->d,i0,i1;
  PetscScalar    *V,*S1,*S2,*St,*tau,*lapackw,*qK=ctx->qK,*qKt;
  PetscScalar     mone=-1.0,one=1.0,zero=0.0,rt;
  PetscReal      *e,*d,*qM=ctx->qM;
  PetscBLASInt   cs1_,rs1_,cs1p1,cs1t2,lds_,info,lw_,ione=1,ld_;

  PetscFunctionBegin;
  lwa = cs1*cs1*7+(3+rs1)*cs1+rs1+ctx->ld*ctx->ld;
  lrwa = 8*cs1;
  if (!work||nw<lwa){
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",5);
  }
  if (!rwork||nrw<lrwa){
    if (nrw<lrwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",8);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",7);
  }
  S1 = ctx->S;
  S2 = ctx->S+ctx->ld;
  qKt = work+nwu;
  nwu += ctx->ld*ctx->ld;
  V = work+nwu;
  nwu += cs1*cs1*4;
  St = work+nwu;
  nwu += (cs1+1)*rs1;
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1+1,&cs1p1);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*2,&cs1t2);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  for (j=0;j<cs1;j++) {
    ierr = PetscMemcpy(St+j*rs1,S1+j*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (j=0;j<rs1;j++) {
    if (qM[j]<0) {
      PetscStackCall("BLASscal",BLASscal_(&cs1_,&mone,St+j,&rs1));
    }
  }
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&cs1_,&cs1_,&rs1_,&one,S1,&lds_,St,&rs1_,&zero,V,&cs1t2));
  for (j=0;j<cs1;j++) {
    ierr = PetscMemcpy(St+j*rs1,S2+j*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (j=0;j<rs1;j++) {
    if (qM[j]<0) {
      PetscStackCall("BLASscal",BLASscal_(&cs1_,&mone,St+j,&rs1_));
    }
  }
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&cs1_,&cs1_,&rs1_,&one,S1,&lds_,St,&rs1_,&zero,V+cs1*cs1t2,&cs1t2));
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&cs1_,&cs1_,&rs1_,&one,S2,&lds_,St,&rs1_,&zero,V+cs1*cs1t2+cs1,&cs1t2));
  tau = work+nwu;
  nwu += 2*cs1;
  d = rwork+nrwu;
  nrwu += 2*cs1;
  e = rwork+nrwu;
  nrwu += 2*cs1-1;
  lapackw = work+nwu;
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKsytrd",LAPACKsytrd_("U",&cs1t2,V,&cs1t2,d,e,tau,lapackw,&lw_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSYTRD %d",info);
  PetscStackCallBLAS("LAPACKorgtr",LAPACKorgtr_("U",&cs1t2,V,&cs1t2,tau,lapackw,&lw_,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xORGTR %d",info);
  PetscStackCallBLAS("LAPACKsteqr",LAPACKsteqr_("V",&cs1t2,d,e,V,&cs1t2,rwork+nrwu,&info));
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xSTEQR %d",info);
  
  /* d is in ascending order */
  i0 = 0;
  i1 = 2*cs1-1;
  while (i1-i0+1>cs1-1) {
    if (PetscAbsReal(d[i0])>PetscAbsReal(d[i1])) i0++;
    else i1--;
  }
  if (i0>cs1-1) {
    if (i1<2*cs1-1) {
      ierr = PetscMemcpy(V+i0*(2*cs1),V+(i1+1)*(2*cs1),(2*cs1-i1-1)*2*cs1*sizeof(PetscScalar));CHKERRQ(ierr);
      for (i=0;i<2*cs1-i1-1;i++) d[i0+i] = d[i1+1+i];
    }
    off1 = 0;
  } else {
    off1 = i1-i0+1;
    if (i0>0) {
      ierr = PetscMemcpy(V+off1*2*cs1,V,i0*2*cs1*sizeof(PetscScalar));CHKERRQ(ierr);
      for (i=0;i<i0;i++) d[off1+i] = d[i];
    }
  } 
  for (j=0;j<cs1+1;j++) {
    qM[j] = (d[off1+j]>0)?1.0:-1.0;
    d[off1+j] = PetscSqrtReal(PetscAbsReal(d[off1+j]));
  }
  for (j=0;j<cs1+1;j++) {
    rt = 1/d[off1+j];
    PetscStackCall("BLASscal",BLASscal_(&cs1t2,&rt,V+(off1+j)*2*cs1,&ione));
  }
  off2 = off1*2*cs1+cs1;
  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&cs1_,&one,S1,&lds_,V+off1*2*cs1,&cs1t2,&zero,St,&rs1_));
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&cs1_,&one,S2,&lds_,V+off2,&cs1t2,&one,St,&rs1_));
  ierr = SlepcUpdateVectors(rs1,qep->V,0,cs1+1,St,rs1,PETSC_FALSE);CHKERRQ(ierr);
  
  /* Update qK */
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&rs1_,&one,qK,&ld_,St,&rs1_,&zero,qKt,&rs1_));
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&cs1p1,&cs1p1,&rs1_,&one,St,&rs1_,qKt,&rs1_,&zero,qK,&ld_));
 
  /* Update S */
  ierr = PetscMemzero(ctx->S,lds*ctx->ld*sizeof(PetscScalar));CHKERRQ(ierr);
  for (j=0;j<cs1+1;j++) {
    for (i=0;i<cs1;i++) {
      S1[j+i*lds] = d[off1+j]*d[off1+j]*V[(off1+j)*2*cs1+i];
      S2[j+i*lds] = d[off1+j]*d[off1+j]*V[(off1+j)*2*cs1+cs1+i];
    }
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IndefOrthog_CGS"
/*
  compute x = x - y*ss^{-1}*y^T*s*x where ss=y^T*s*y
  s diagonal (signature matrix)
*/
static PetscErrorCode IndefOrthog_CGS(PetscInt n,PetscReal *s,PetscInt nv,PetscScalar *Y,PetscInt ldy,PetscReal *ss,PetscScalar *x,PetscScalar *h,PetscScalar *work,PetscInt lw)
{
  PetscErrorCode ierr;
  PetscInt       i,nwall,nwu=0;
  PetscScalar    *h2,*h1,*t1,*t2,one=1.0,zero=0.0,onen=-1.0;
  PetscBLASInt   n_,nv_,ldy_,inc=1;

  PetscFunctionBegin;
  nwall = 3*n;
  if (!work || lw<nwall) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",11);
  t1 = work+nwu;
  nwu += n;
  t2 = work+nwu;
  nwu += n;
  h2 = work+nwu;
  nwu +=n;
  if (h) h1 = h;
  else h1 = h2;
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(nv,&nv_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldy,&ldy_);CHKERRQ(ierr);
  for (i=0;i<n;i++) t1[i] = s[i]*x[i];
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&nv_,&one,Y,&ldy_,t1,&inc,&zero,t2,&inc));
  for (i=0;i<nv;i++) h1[i] = t2[i]/ss[i];
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&nv_,&onen,Y,&ldy_,h1,&inc,&one,x,&inc));
  /* Repeat */
  for (i=0;i<n;i++) t1[i] = s[i]*x[i];
  PetscStackCall("BLASgemv",BLASgemv_("C",&n_,&nv_,&one,Y,&ldy_,t1,&inc,&zero,t2,&inc));
  for (i=0;i<nv;i++) h2[i] = t2[i]/ss[i];
  PetscStackCall("BLASgemv",BLASgemv_("N",&n_,&nv_,&onen,Y,&ldy_,h2,&inc,&one,x,&inc));
  if (h) {
    for (i=0;i<nv;i++) h[i] += h2[i];
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IndefNorm"
/*
   normalization with a indefinite norm
*/
static PetscErrorCode IndefNorm(PetscInt n,PetscReal *s,PetscScalar *x,PetscReal *norm)
{
  PetscInt     i;
  PetscReal    r=0.0,t,max=0.0;

  PetscFunctionBegin;
  for (i=0;i<n;i++) {
    t = PetscAbsScalar(x[i]);
    if (t > max) max = t;
  }
  for (i=0;i<n;i++) {
    t = PetscRealPart(x[i])/max;
    r += t*t*s[i];
  }
  if (r<0) r = -max*PetscSqrtReal(-r);
  else r = max*PetscSqrtReal(r);
  for (i=0;i<n;i++) {
    x[i] /= r;
  }
  if (norm) *norm = r;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSTOARTrunc"
PetscErrorCode QEPSTOARTrunc(QEP qep,PetscInt rs1,PetscInt cs1,PetscScalar *work,PetscInt nw,PetscReal *rwork,PetscInt nrw)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx=(QEP_STOAR*)qep->data;
  PetscInt       lwa,nwu=0,lrwa,nrwu=0;
  PetscInt       j,i,n,lds=2*ctx->ld;
  PetscScalar    *M,*V,*U,*S=ctx->S,*R,sone=1.0,zero=0.0,t;
  PetscReal      *sg,*qM=ctx->qM,*ss,norm;
  PetscBLASInt   cs1_,rs1_,cs1t2,cs1p1,n_,info,lw_,one=1,lds_,ld_;

  PetscFunctionBegin;
  n = (rs1>2*cs1)?2*cs1:rs1;
  lwa = cs1*rs1*4+n*(rs1+2*cs1)+(cs1+1)*(cs1+2);
  lrwa = n+cs1+1+5*n;
  if (!work||nw<lwa){
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",6);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",5);
  }
  if (!rwork||nrw<lrwa){
    if (nrw<lrwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",8);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",7);
  }
  M = work+nwu;
  nwu += rs1*cs1*2;
  U = work+nwu;
  nwu += rs1*n;
  V = work+nwu;
  nwu += 2*cs1*n;
  sg = rwork+nrwu;
  nrwu += n;
  for (i=0;i<cs1;i++) {
    ierr = PetscMemcpy(M+i*rs1,S+i*lds,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
    ierr = PetscMemcpy(M+(i+cs1)*rs1,S+i*lds+ctx->ld,rs1*sizeof(PetscScalar));CHKERRQ(ierr);  
  }
  ierr = PetscBLASIntCast(n,&n_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1,&cs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(rs1,&rs1_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1*2,&cs1t2);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(cs1+1,&cs1p1);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ctx->ld,&ld_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lwa-nwu,&lw_);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,&info));
#else
  PetscStackCall("LAPACKgesvd",LAPACKgesvd_("S","S",&rs1_,&cs1t2,M,&rs1_,sg,U,&rs1_,V,&n_,work+nwu,&lw_,rwork+nrwu,&info));  
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in Lapack xGESVD %d",info);
/* /// Falta ldl previa /// */
  R = work+nwu;
  ierr = PetscMemzero(R,(cs1+1)*(cs1+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  nwu += (cs1+1)*(cs1+1);
  nwu += cs1+1;
  ss = rwork+nrwu;
  nrwu += cs1+1;
  for (j=0;j<cs1+1;j++) {
    ierr = IndefOrthog_CGS(rs1,qM,j,U,rs1,ss,U+j*rs1,R+j*(cs1+1),work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = IndefNorm(rs1,qM,U+j*rs1,&norm);CHKERRQ(ierr);
    ss[j] = (norm>0)?1.0:-1.0;
    R[j+j*(cs1+1)] = norm;
  }

  /* Update the corresponding vectors V(:,idx) = V*Q(:,idx) */
  ierr = SlepcUpdateVectors(rs1,qep->V,0,cs1+1,U,rs1,PETSC_FALSE);CHKERRQ(ierr);
  
  /* Update S */  
  for (i=0;i<cs1+1;i++) {
    t = sg[i];
    PetscStackCall("BLASscal",BLASscal_(&cs1p1,&t,R+i*(cs1+1),&one));
  }
  ierr = PetscMemzero(S,lds*ctx->ld*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&cs1p1,&cs1_,&cs1p1,&sone,R,&cs1p1,V,&n_,&zero,S,&lds_));
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&cs1p1,&cs1_,&cs1p1,&sone,R,&cs1p1,V+cs1*n,&n_,&zero,S+ctx->ld,&lds_));

  /* Update qM and qK */
  for (j=0;j<cs1+1;j++) qM[j] = ss[j];
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&rs1_,&cs1p1,&rs1_,&sone,ctx->qK,&ld_,U,&rs1_,&zero,work+nwu,&rs1_));
  PetscStackCall("BLASgemm",BLASgemm_("C","N",&cs1p1,&cs1p1,&rs1_,&sone,U,&rs1_,work+nwu,&rs1_,&zero,ctx->qK,&ld_));

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPSTOARSupdate"
/*
  S <- S*Q 
  columns s-s+ncu of S
  rows 0-sr of S
  size(Q) qr x ncu
*/
PetscErrorCode QEPSTOARSupdate(PetscScalar *S,PetscInt ld,PetscInt sr,PetscInt s,PetscInt ncu,PetscInt qr,PetscScalar *Q,PetscInt ldq,PetscScalar *work,PetscInt nw)
{
  PetscErrorCode ierr;
  PetscScalar    a=1.0,b=0.0;
  PetscBLASInt   sr_,ncu_,ldq_,lds_,qr_;
  PetscInt       lwa,j,lds=2*ld;

  PetscFunctionBegin;
  lwa = sr*ncu;
  if (!work||nw<lwa){
    if (nw<lwa) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",10);
    if (!work) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid argument %d",9);
  }
  ierr = PetscBLASIntCast(sr,&sr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(qr,&qr_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ncu,&ncu_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(lds,&lds_);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(ldq,&ldq_);CHKERRQ(ierr);
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S,&lds,Q,&ldq_,&b,work,&sr_));
  for(j=0;j<ncu;j++){
    ierr = PetscMemcpy(S+lds*(s+j),work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscStackCall("BLASgemm",BLASgemm_("N","N",&sr_,&ncu_,&qr_,&a,S+ld,&lds,Q,&ldq_,&b,work,&sr_));
  for(j=0;j<ncu;j++){
    ierr = PetscMemcpy(S+lds*(s+j)+ld,work+j*sr,sr*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "QEPSolve_STOAR"
PetscErrorCode QEPSolve_STOAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx=(QEP_STOAR*)qep->data;
  PetscInt       j,k,l,nv,ld=ctx->ld,lds=ctx->d*ctx->ld,off,ldds,t;
  PetscInt       lwa,lrwa,nwu=0,nrwu=0;
  Vec            w=qep->work[0];
  PetscScalar    *S=ctx->S,*Q,*work;
  PetscReal      beta,norm,*omega,*a,*b,*r,*qM=ctx->qM,*rwork;
  PetscBool      breakdown;

  PetscFunctionBegin;
  lwa = 9*ld*ld+5*ld;
  ierr = PetscMalloc(lwa*sizeof(PetscScalar),&work);CHKERRQ(ierr);
  lrwa = 8*ld;
  ierr = PetscMalloc(lrwa*sizeof(PetscReal),&rwork);CHKERRQ(ierr);

  /* Get the starting Lanczos vector */
  if (qep->nini==0) {  
    ierr = SlepcVecSetRandom(qep->V[0],qep->rand);CHKERRQ(ierr);
  }
  ierr = SlepcVecSetRandom(qep->V[1],qep->rand);CHKERRQ(ierr);
  ierr = IPNorm(qep->ip,qep->V[0],&norm);CHKERRQ(ierr);
  ierr = VecScale(qep->V[0],1/norm);CHKERRQ(ierr);
  qM[0] = (norm>0)?1.0:-1.0;
  ierr = QEPSTOARqKupdate(qep,0,w);CHKERRQ(ierr);
  S[0] = norm;
  ierr = IPPseudoOrthogonalize(qep->ip,1,qep->V,qM,qep->V[1],S+ld,&norm,NULL);CHKERRQ(ierr);
  *(S+ld) *= *(ctx->qM);
  ierr = VecScale(qep->V[1],1/norm);CHKERRQ(ierr);
  qM[1] = (norm>0)?1.0:-1.0;
  ierr = QEPSTOARqKupdate(qep,1,w);CHKERRQ(ierr);
  S[1+ld] = norm;
  if (PetscAbsReal(norm)<PETSC_MACHINE_EPSILON) SETERRQ(PetscObjectComm((PetscObject)qep),1,"Problem with initial vector");
  ierr = QEPSTOARNorm(qep,0,&norm,work+nwu,lwa-nwu);CHKERRQ(ierr);
  ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  omega[0] = (norm > 0)?1.0:-1.0;
  ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
  for (j=0;j<2;j++) {
    S[j] /= norm;
    S[j+ld] /= norm;
  }

  /* Restart loop */
  l = 0;
  ierr = DSGetLeadingDimension(qep->ds,&ldds);CHKERRQ(ierr);
  while (qep->reason == QEP_CONVERGED_ITERATING) {
    qep->its++;
    ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    b = a+ldds;
    r = b+ldds;
    ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    
    /* Compute an nv-step Lanczos factorization */
    nv = PetscMin(qep->nconv+qep->mpd,qep->ncv);
    ierr = QEPSTOARrun(qep,a,b,omega,qep->V,qep->nconv+l,&nv,&breakdown,work+nwu,lwa-nwu,qep->work,3);CHKERRQ(ierr);
    beta = b[nv-1];    ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
    ierr = DSSetDimensions(qep->ds,nv,0,qep->nconv,qep->nconv+l);CHKERRQ(ierr);
    if (l==0) {
      ierr = DSSetState(qep->ds,DS_STATE_INTERMEDIATE);CHKERRQ(ierr);
    } else {
      ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);
    }
    /* Solve projected problem */
    ierr = DSSolve(qep->ds,qep->eigr,qep->eigi);CHKERRQ(ierr);
    ierr = DSSort(qep->ds,qep->eigr,qep->eigi,NULL,NULL,NULL);CHKERRQ(ierr);

    /* Check convergence */
    ierr = DSGetDimensions(qep->ds,NULL,NULL,NULL,NULL,&t);CHKERRQ(ierr);    
    ierr = QEPKrylovConvergence(qep,PETSC_FALSE,qep->nconv,t-qep->nconv,nv,beta,&k);CHKERRQ(ierr);
    if (qep->its >= qep->max_it) qep->reason = QEP_DIVERGED_ITS;
    if (k >= qep->nev) qep->reason = QEP_CONVERGED_TOL;

    /* Update l */
    if (qep->reason != QEP_CONVERGED_ITERATING || breakdown) l = 0;
    else { 
      l = PetscMax(1,(PetscInt)((nv-k)/2));
      l = PetscMin(l,t);
      ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
      if (*(a+ldds+k+l-1)!=0) {
        if (k+l<nv-1) l = l+1;
        else l = l-1;
      }
      ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
    }

    /* Update S */
    off = qep->nconv*ldds;
    ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
    ierr = QEPSTOARSupdate(S,ld,nv+2,qep->nconv,k+l-qep->nconv,nv,Q+off,ldds,work+nwu,lwa-nwu);CHKERRQ(ierr);
    ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);

    /* Copy last column of S */
    ierr = PetscMemcpy(S+lds*(k+l),S+lds*nv,lds*sizeof(PetscScalar));CHKERRQ(ierr);

    if (qep->reason == QEP_CONVERGED_ITERATING) {
      if (breakdown) {

        /* Stop if breakdown */
        ierr = PetscInfo2(qep,"Breakdown STOAR method (it=%D norm=%G)\n",qep->its,beta);CHKERRQ(ierr);
        qep->reason = QEP_DIVERGED_BREAKDOWN;
      } else {

        /* Truncate S */
        ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        ierr = QEPSTOARTrunc(qep,nv+2,k+l+1,work+nwu,lwa-nwu,rwork+nrwu,lrwa-nrwu);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);

        /* Prepare the Rayleigh quotient for restart */
        ierr = DSGetArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSGetArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSGetArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
        r = a + 2*ldds;
        for (j=k;j<k+l;j++) {
          r[j] = PetscRealPart(Q[nv-1+j*ldds]*beta);
        }
        b = a+ldds;
        b[k+l-1] = r[k+l-1];
        omega[k+l] = omega[nv];
        ierr = DSRestoreArray(qep->ds,DS_MAT_Q,&Q);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(qep->ds,DS_MAT_T,&a);CHKERRQ(ierr);
        ierr = DSRestoreArrayReal(qep->ds,DS_MAT_D,&omega);CHKERRQ(ierr);
      }
    }
    qep->nconv = k;
    ierr = QEPMonitor(qep,qep->its,qep->nconv,qep->eigr,qep->eigi,qep->errest,nv);CHKERRQ(ierr);
  }

  /* Update vectors V = V*S */    
  ierr = SlepcUpdateVectors(nv+2,qep->V,0,qep->nconv,S,lds,PETSC_FALSE);CHKERRQ(ierr);
  for (j=0;j<qep->nconv;j++) {
    qep->eigr[j] *= qep->sfactor;
    qep->eigi[j] *= qep->sfactor;
  }

  /* truncate Schur decomposition and change the state to raw so that
     DSVectors() computes eigenvectors from scratch */
  ierr = DSSetDimensions(qep->ds,qep->nconv,0,0,0);CHKERRQ(ierr);
  ierr = DSSetState(qep->ds,DS_STATE_RAW);CHKERRQ(ierr);

  /* Compute eigenvectors */
  if (qep->nconv > 0) {
    ierr = QEPComputeVectors_Schur(qep);CHKERRQ(ierr);
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  ierr = PetscFree(rwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPDestroy_STOAR"
PetscErrorCode QEPDestroy_STOAR(QEP qep)
{
  PetscErrorCode ierr;
  QEP_STOAR      *ctx = (QEP_STOAR*)qep->data;

  PetscFunctionBegin;
  ierr = PetscFree(ctx->S);CHKERRQ(ierr);
  ierr = PetscFree(ctx->qM);CHKERRQ(ierr);
  ierr = PetscFree(ctx->qK);CHKERRQ(ierr);
  ierr = PetscFree(qep->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "QEPCreate_STOAR"
PETSC_EXTERN PetscErrorCode QEPCreate_STOAR(QEP qep)
{
  PetscFunctionBegin;
  qep->ops->solve                = QEPSolve_STOAR;
  qep->ops->setup                = QEPSetUp_STOAR;
  qep->ops->reset                = QEPReset_Default;
  qep->ops->destroy              = QEPDestroy_STOAR;
  PetscFunctionReturn(0);
}