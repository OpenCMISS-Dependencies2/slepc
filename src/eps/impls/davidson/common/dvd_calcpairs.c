/*
  SLEPc eigensolver: "davidson"

  Step: calc the best eigenpairs in the subspace V.

  For that, performs these steps:
    1) Update W <- A * V
    2) Update H <- V' * W
    3) Obtain eigenpairs of H
    4) Select some eigenpairs
    5) Compute the Ritz pairs of the selected ones

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2010, Universidad Politecnica de Valencia, Spain

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

#include "davidson.h"
#include <slepcblaslapack.h>

PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateV(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateW(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateAV(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_updateBV(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_VtAV_gen(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func *sr);
PetscErrorCode dvd_calcpairs_VtBV_gen(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func *sr);
PetscErrorCode dvd_calcpairs_projeig_eig(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_qz_std(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_projeig_qz_gen(dvdDashboard *d);
PetscErrorCode dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n);
PetscErrorCode dvd_calcpairs_selectPairs_eig(dvdDashboard *d, PetscInt n);
PetscErrorCode dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *X);
PetscErrorCode dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *Y);
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                                   Vec *R, PetscScalar *auxS, Vec auxV);
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s,
                                      PetscInt r_e, Vec *R);
PetscErrorCode dvd_calcpairs_updateMatV(Mat A, Vec **AV, PetscInt *size_AV,
                                        PetscBool doUpdate, PetscBool doNew,
                                        dvdDashboard *d);
PetscErrorCode dvd_calcpairs_WtMatV_gen(PetscScalar **H, MatType_t sH,
  PetscInt ldH, PetscInt *size_H, PetscScalar *MTY, PetscInt ldMTY,
  PetscScalar *MTX, PetscInt ldMTX, PetscInt rMT, PetscInt cMT, Vec *W,
  Vec *V, PetscInt size_V, PetscScalar *auxS, DvdReduction *r,
  DvdMult_copy_func *sr, dvdDashboard *d);

/**** Control routines ********************************************************/
#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz"
PetscErrorCode dvd_calcpairs_qz(dvdDashboard *d, dvdBlackboard *b, IP ipI)
{
  PetscErrorCode  ierr;
  PetscInt        i;
  PetscBool       std_probl,her_probl;

  PetscFunctionBegin;

  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

  /* Setting configuration constrains */
#if !defined(PETSC_USE_COMPLEX)
  /* if the last converged eigenvalue is complex its conjugate pair is also
     converged */
  b->max_nev = PetscMax(b->max_nev, d->nev+1);
#else
  b->max_nev = PetscMax(b->max_nev, d->nev);
#endif
  b->own_vecs+= b->size_V*(d->B?2:1) + d->eps->nds*(d->ipV_oneMV?1:0);
                                               /* AV, BV?, BDS? */
  b->own_scalars+= b->max_size_V*b->max_size_V*2*(std_probl?1:2);
                                              /* H, G?, S, T? */
  b->own_scalars+= b->max_size_V*b->max_size_V*(std_probl?1:2);
                                              /* pX, pY? */
  b->own_scalars+= b->max_nev*b->max_nev*(her_probl?0:(std_probl?1:2)); /* cS?, cT?? */
  b->max_size_auxS = PetscMax(PetscMax(
                              b->max_size_auxS,
                              b->max_size_V*b->max_size_V*4
                                                      /* SlepcReduction */ ),
                              std_probl?0:(b->max_size_V*11+16) /* projeig */);
#if defined(PETSC_USE_COMPLEX)
  b->max_size_auxS = PetscMax(b->max_size_auxS, b->max_size_V);
                                           /* dvd_calcpairs_projeig_eig */
#endif

  /* Setup the step */
  if (b->state >= DVD_STATE_CONF) {
    d->real_AV = d->AV = b->free_vecs; b->free_vecs+= b->size_V;
    d->max_size_AV = b->size_V;
    d->max_size_proj = b->max_size_V;
    d->H = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    d->real_H = d->H;
    d->pX = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    d->S = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    if (!her_probl) {
      d->cS = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->max_size_cS = b->max_nev;
    } else {
      d->cS = PETSC_NULL;
      d->max_size_cS = 0;
    }
    d->ldcS = b->max_nev;
    d->ipV = ipI;
    d->ipW = ipI;
    if (d->ipV_oneMV) {
      d->BDS = b->free_vecs; b->free_vecs+= d->eps->nds;
      for (i=0; i<d->eps->nds; i++) {
        ierr = MatMult(d->B, d->eps->DS[i], d->BDS[i]); CHKERRQ(ierr);
      }
    }
    if (d->B) {
      d->real_BV = d->BV = b->free_vecs; b->free_vecs+= b->size_V;
    } else {
      d->real_BV = d->BV = PETSC_NULL;
    }
    if (!std_probl) {
      d->G = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      d->real_G = d->G;
      d->T = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
      d->cT = b->free_scalars; b->free_scalars+= b->max_nev*b->max_nev;
      d->ldcT = b->max_nev;
      d->pY = b->free_scalars; b->free_scalars+= b->max_size_V*b->max_size_V;
    } else {
      d->real_G = d->G = PETSC_NULL;
      d->T = PETSC_NULL;
      d->cT = PETSC_NULL;
      d->ldcT = 0;
      d->pY = PETSC_NULL;
    }

    d->calcPairs = dvd_calcpairs_proj;
    d->calcpairs_residual = dvd_calcpairs_res_0;
    d->calcpairs_proj_res = dvd_calcpairs_proj_res;
    d->calcpairs_selectPairs = PETSC_NULL;
    d->calcpairs_X = dvd_calcpairs_X;
    d->calcpairs_Y = dvd_calcpairs_Y;
    d->ipI = ipI;
    d->doNotUpdateBV = PETSC_FALSE;
    DVD_FL_ADD(d->startList, dvd_calcpairs_qz_start);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "dvd_calcpairs_qz_start"
PetscErrorCode dvd_calcpairs_qz_start(dvdDashboard *d)
{
  PetscBool       std_probl, her_probl;
  PetscInt        i;

  PetscFunctionBegin;

  std_probl = DVD_IS(d->sEP, DVD_EP_STD)?PETSC_TRUE:PETSC_FALSE;
  her_probl = DVD_IS(d->sEP, DVD_EP_HERMITIAN)?PETSC_TRUE:PETSC_FALSE;

  d->size_AV = 0;
  d->AV = d->real_AV;
  d->max_size_AV = d->max_size_V;
  d->size_H = 0;
  d->H = d->real_H;
  d->ldH = d->max_size_proj;
  for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cS[i] = 0.0;
  d->size_cX = d->size_cY = 0;
  d->size_BV = 0;
  if (d->B) {
    d->BV = d->real_BV;
    d->max_size_BV = d->max_size_V;
  } else {
    d->BV = PETSC_NULL;
    d->max_size_BV = 0;
  }
  d->size_G = 0;
  d->G = d->real_G;
  if (!std_probl) {
    for (i=0; i<d->max_size_cS*d->max_size_cS; i++) d->cT[i] = 0.0;
    /* If the problem is GHEP without B-orthonormalization, active BcX */
    if(her_probl) d->BcX = d->AV;

    /* Else, active the left and right converged invariant subspaces */
    else {d->cY = d->AV; d->BcX = PETSC_NULL; }
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj"
PetscErrorCode dvd_calcpairs_proj(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  DvdReduction    r;
  DvdReductionChunk
                  ops[2];
  DvdMult_copy_func
                  sr[2];
  PetscInt        size_in = 2*d->size_V*d->size_V;
  PetscScalar     *in = d->auxS, *out = in+size_in;

  PetscFunctionBegin;

  /* Prepare reductions */
  ierr = SlepcAllReduceSumBegin(ops, 2, in, out, size_in, &r,
                                ((PetscObject)d->V[0])->comm); CHKERRQ(ierr);

  /* Update AV, BV, W and the projected matrices */
  ierr = dvd_calcpairs_updateV(d); CHKERRQ(ierr);
  ierr = dvd_calcpairs_updateAV(d); CHKERRQ(ierr);
  if (!d->W) {
    ierr = dvd_calcpairs_VtAV_gen(d, &r, &sr[0]); CHKERRQ(ierr);
    if (d->BV) { ierr = dvd_calcpairs_updateBV(d); CHKERRQ(ierr); }
  } else {
    if (d->BV) { ierr = dvd_calcpairs_updateBV(d); CHKERRQ(ierr); }
    ierr = dvd_calcpairs_updateW(d); CHKERRQ(ierr);
    ierr = dvd_calcpairs_VtAV_gen(d, &r, &sr[0]); CHKERRQ(ierr);
  }
  if (DVD_ISNOT(d->sEP, DVD_EP_STD)) {
    ierr = dvd_calcpairs_VtBV_gen(d, &r, &sr[1]); CHKERRQ(ierr);
  }

  /* Do reductions */
  ierr = SlepcAllReduceSumEnd(&r); CHKERRQ(ierr);

  /* Perform the transformation on the projected problem */
  if (d->calcpairs_proj_trans) {
    ierr = d->calcpairs_proj_trans(d); CHKERRQ(ierr);
  }

  if (d->MT_type != DVD_MT_IDENTITY) {
    d->MT_type = DVD_MT_IDENTITY;
//    d->pX_type|= DVD_MAT_IDENTITY;
    d->V_tra_s = d->V_tra_e = 0;
  }

  /* Solve the projected problem */
  d->pX_type = 0;
//TODO: uncomment this condition 
//  if(d->V_new_e - d->V_new_s > 0) {
    if (DVD_IS(d->sEP, DVD_EP_STD)) {
      if (DVD_IS(d->sEP, DVD_EP_HERMITIAN)) {
        ierr = dvd_calcpairs_projeig_eig(d); CHKERRQ(ierr);
      } else {
        ierr = dvd_calcpairs_projeig_qz_std(d); CHKERRQ(ierr);
      }
    } else {
      ierr = dvd_calcpairs_projeig_qz_gen(d); CHKERRQ(ierr);
    }
//  }
  d->V_new_s = d->V_new_e;

  /* Check consistency */
  if ((d->size_V != d->V_new_e) || (d->size_V != d->size_H) ||
      (d->size_V != d->size_AV) || (DVD_ISNOT(d->sEP, DVD_EP_STD) && (
      (d->size_V != d->size_G) || (d->BV && d->size_V != d->size_BV) ))) {
    SETERRQ(PETSC_COMM_SELF,1, "Consistency broken!");
  }

  PetscFunctionReturn(0);
}

/**** Basic routines **********************************************************/

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateV"
PetscErrorCode dvd_calcpairs_updateV(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  Vec             *cX = d->BcX? d->BcX : ( (d->cY && !d->W)? d->cY : d->cX );

  PetscFunctionBegin;

  /* V <- gs([cX f.V(0:f.V_new_s-1)], f.V(V_new_s:V_new_e-1)) */
  if (d->ipV_oneMV) {
    ierr = dvd_BorthV(d->ipV, d->eps->DS, d->BDS, d->eps->nds, d->cX, d->real_BV,
                      d->size_cX, d->V, d->BV, d->V_new_s, d->V_new_e,
                      d->auxS, d->auxV[0], d->eps->rand); CHKERRQ(ierr);
  } else {
    ierr = dvd_orthV(d->ipV, d->eps->DS, d->eps->nds, cX, d->size_cX, d->V,
                   d->V_new_s, d->V_new_e, d->auxS, d->auxV[0], d->eps->rand);
    CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateW"
PetscErrorCode dvd_calcpairs_updateW(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* Update W */
  ierr = d->calcpairs_W(d); CHKERRQ(ierr);

  /* W <- gs([cY f.W(0:f.V_new_s-1)], f.W(V_new_s:V_new_e-1)) */
  ierr = dvd_orthV(d->ipW, PETSC_NULL, 0, d->cY, d->size_cY, d->W, d->V_new_s,
                   d->V_new_e, d->auxS, d->auxV[0], d->eps->rand);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateAV"
PetscErrorCode dvd_calcpairs_updateAV(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* f.AV(f.V_tra) = f.AV * f.MT; f.AV(f.V_new) = A*f.V(f.V_new) */
  ierr = dvd_calcpairs_updateMatV(d->A, &d->AV, &d->size_AV, PETSC_TRUE, PETSC_TRUE, d);
  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateBV"
PetscErrorCode dvd_calcpairs_updateBV(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* f.BV(f.V_tra) = f.BV * f.MT; f.BV(f.V_new) = B*f.V(f.V_new) */
  ierr = dvd_calcpairs_updateMatV(d->B, &d->BV, &d->size_BV, !d->doNotUpdateBV?PETSC_TRUE:PETSC_FALSE, !d->ipV_oneMV?PETSC_TRUE:PETSC_FALSE, d);
  CHKERRQ(ierr);
  d->doNotUpdateBV = PETSC_FALSE;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_VtAV_gen"
PetscErrorCode dvd_calcpairs_VtAV_gen(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func *sr)
{
  PetscInt        ldMTY = d->MTY?d->ldMTY:d->ldMTX;
  /* WARNING: auxS uses space assigned to r */
  PetscScalar     *auxS = r->out,
                  *MTY = d->MTY?d->MTY:d->MTX;
  Vec             *W = d->W?d->W:d->V;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* f.H = [f.H(f.V_imm,f.V_imm)        f.V(f.V_imm)'*f.AV(f.V_new);
            f.V(f.V_new)'*f.AV(f.V_imm) f.V(f.V_new)'*f.AV(f.V_new) ] */
  if (DVD_IS(d->sA,DVD_MAT_HERMITIAN))
    d->sH = DVD_MAT_HERMITIAN | DVD_MAT_IMPLICIT | DVD_MAT_UTRIANG;
  if ((d->V_imm_e - d->V_imm_s == 0) && (d->V_tra_e - d->V_tra_s == 0))
    d->size_H = 0;
  ierr = dvd_calcpairs_WtMatV_gen(&d->H, d->sH, d->ldH, &d->size_H,
                                  &MTY[ldMTY*d->V_tra_s], ldMTY,
                                  &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                                  d->size_MT, d->V_tra_e-d->V_tra_s,
                                  W, d->AV, d->size_V,
                                 auxS, r, sr, d); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_VtBV_gen"
PetscErrorCode dvd_calcpairs_VtBV_gen(dvdDashboard *d, DvdReduction *r,
                                      DvdMult_copy_func *sr)
{
  PetscErrorCode  ierr;
  PetscInt        ldMTY = d->MTY?d->ldMTY:d->ldMTX;
  /* WARNING: auxS uses space assigned to r */
  PetscScalar     *auxS = r->out,
                  *MTY = d->MTY?d->MTY:d->MTX;
  Vec             *W = d->W?d->W:d->V;

  PetscFunctionBegin;

  /* f.G = [f.G(f.V_imm,f.V_imm)        f.V(f.V_imm)'*f.BV(f.V_new);
            f.V(f.V_new)'*f.BV(f.V_imm) f.V(f.V_new)'*f.BV(f.V_new) ] */
  if (DVD_IS(d->sB,DVD_MAT_HERMITIAN))
    d->sG = DVD_MAT_HERMITIAN | DVD_MAT_IMPLICIT | DVD_MAT_UTRIANG;
  if ((d->V_imm_e - d->V_imm_s == 0) && (d->V_tra_e - d->V_tra_s == 0))
    d->size_G = 0;
  ierr = dvd_calcpairs_WtMatV_gen(&d->G, d->sG, d->ldH, &d->size_G,
                                  &MTY[ldMTY*d->V_tra_s], ldMTY,
                                  &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                                  d->size_MT, d->V_tra_e-d->V_tra_s,
                                  W, d->BV?d->BV:d->V, d->size_V,
                                  auxS, r, sr, d); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* in complex, d->size_H real auxiliar values are needed */
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_eig"
PetscErrorCode dvd_calcpairs_projeig_eig(dvdDashboard *d)
{
  PetscErrorCode  ierr;
  PetscReal       *w;
#if defined(PETSC_USE_COMPLEX)
  PetscInt        i;
#endif

  PetscFunctionBegin;

  /* S <- H */
  d->ldS = d->ldpX = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, DVD_MAT_LTRIANG, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);

  /* S = pX' * L * pX */
#if !defined(PETSC_USE_COMPLEX)
  w = d->eigr;
#else
  w = (PetscReal*)d->auxS;
  for (i=0; i<d->size_H; i++) w[i] = PetscRealPart(d->eigr[i]);
#endif
  ierr = EPSDenseHEP(d->size_H, d->S, d->ldS, w, d->pX); CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  for (i=0; i<d->size_H; i++) d->eigr[i] = w[i];
#endif

  d->pX_type = (d->pX_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;

  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_eig;

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_std"
PetscErrorCode dvd_calcpairs_projeig_qz_std(dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* S <- H */
  d->ldS = d->ldpX = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);

  /* S = pX' * H * pX */
  ierr = EPSDenseHessenberg(d->size_H, 0, d->S, d->ldS, d->pX); CHKERRQ(ierr);
  ierr = EPSDenseSchur(d->size_H, 0, d->S, d->ldS, d->pX, d->eigr, d->eigi);
  CHKERRQ(ierr);

  d->pX_type = (d->pX_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;

  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_qz;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_projeig_qz_gen"
/*
  auxS(dgges) = size_H (beta) + 8*size_H+16 (work)
  auxS(zgges) = size_H (beta) + 1+2*size_H (work) + 8*size_H (rwork)
*/
PetscErrorCode dvd_calcpairs_projeig_qz_gen(dvdDashboard *d)
{
#if defined(SLEPC_MISSING_LAPACK_GGES)
  PetscFunctionBegin;
  SETERRQ(((PetscObject)(d->eps))->comm,PETSC_ERR_SUP,"GGES - Lapack routine is unavailable.");
#else
  PetscErrorCode  ierr;
  PetscScalar     *beta = d->auxS;
#if !defined(PETSC_USE_COMPLEX)
  PetscScalar     *auxS = beta + d->size_H;
  PetscBLASInt    n_auxS = d->size_auxS - d->size_H;
#else
  PetscReal       *auxR = (PetscReal*)(beta + d->size_H);
  PetscScalar     *auxS = (PetscScalar*)(auxR+8*d->size_H);
  PetscBLASInt    n_auxS = d->size_auxS - 9*d->size_H;
#endif
  PetscInt        i;
  PetscBLASInt    info,n,a;

  PetscFunctionBegin;
  /* S <- H, T <- G */
  d->ldS = d->ldT = d->ldpX = d->ldpY = d->size_H;
  ierr = SlepcDenseCopyTriang(d->S, 0, d->size_H, d->H, d->sH, d->ldH,
                              d->size_H, d->size_H);CHKERRQ(ierr);
  ierr = SlepcDenseCopyTriang(d->T, 0, d->size_H, d->G, d->sG, d->ldH,
                              d->size_H, d->size_H);CHKERRQ(ierr);

  /* S = Z'*H*Q, T = Z'*G*Q */
  n = d->size_H;
#if !defined(PETSC_USE_COMPLEX)
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr, d->eigi, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, PETSC_NULL, &info);
#else
  LAPACKgges_(d->pY?"V":"N", "V", "N", PETSC_NULL, &n, d->S, &n, d->T, &n,
              &a, d->eigr, beta, d->pY, &n, d->pX, &n,
              auxS, &n_auxS, auxR, PETSC_NULL, &info);
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "Error in Lapack GGES %d", info);

  /* eigr[i] <- eigr[i] / beta[i] */
  for (i=0; i<d->size_H; i++)
    d->eigr[i] /= beta[i],
    d->eigi[i] /= beta[i];

  d->pX_type = (d->pX_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;
  d->pY_type = (d->pY_type & !DVD_MAT_IDENTITY) | DVD_MAT_UNITARY;
  d->calcpairs_selectPairs = dvd_calcpairs_selectPairs_qz;

  PetscFunctionReturn(0);
#endif
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs_eig"
PetscErrorCode dvd_calcpairs_selectPairs_eig(dvdDashboard *d, PetscInt n)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  ierr = EPSSortDenseHEP(d->eps, d->size_H, 0, d->eigr, d->pX, d->ldpX);
  CHKERRQ(ierr);

  if (d->calcpairs_eigs_trans) {
    ierr = d->calcpairs_eigs_trans(d); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_selectPairs_qz"
PetscErrorCode dvd_calcpairs_selectPairs_qz(dvdDashboard *d, PetscInt n)
{
  PetscErrorCode  ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar     s;
  PetscInt        i, j;
#endif
  PetscFunctionBegin;

  if ((d->ldpX != d->size_H) ||
      ( d->T &&
        ((d->ldS != d->ldT) || (d->ldpX != d->ldpY) ||
         (d->ldpX != d->size_H)) ) ) {
     SETERRQ(PETSC_COMM_SELF,1, "Error before ordering eigenpairs");
  }

  if (d->T) {
    ierr = EPSSortDenseSchurGeneralized(d->eps, d->size_H, 0, n, d->S, d->T,
                                        d->ldS, d->pY, d->pX, d->eigr,
                                        d->eigi); CHKERRQ(ierr);
  } else {
    ierr = EPSSortDenseSchur(d->eps, d->size_H, 0, d->S, d->ldS, d->pX,
                             d->eigr, d->eigi); CHKERRQ(ierr);
  }

  if (d->calcpairs_eigs_trans) {
    ierr = d->calcpairs_eigs_trans(d); CHKERRQ(ierr);
  }

  /* Some functions need the diagonal elements in T be real */
#if defined(PETSC_USE_COMPLEX)
  if (d->T) for(i=0; i<d->size_H; i++)
    if (PetscImaginaryPart(d->T[d->ldT*i+i]) != 0.0) {
      s = PetscConj(d->T[d->ldT*i+i])/PetscAbsScalar(d->T[d->ldT*i+i]);
      for(j=0; j<=i; j++)
        d->T[d->ldT*i+j] = PetscRealPart(d->T[d->ldT*i+j]*s),
        d->S[d->ldS*i+j]*= s;
      for(j=0; j<d->size_H; j++) d->pX[d->ldpX*i+j]*= s;
    }
#endif

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_X"
PetscErrorCode dvd_calcpairs_X(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *X)
{
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* X = V * U(0:n-1) */
  if (DVD_IS(d->pX_type,DVD_MAT_IDENTITY)) {
    if (d->V != X) for(i=r_s; i<r_e; i++) {
      ierr = VecCopy(d->V[i], X[i]); CHKERRQ(ierr);
    }
  } else {
    ierr = SlepcUpdateVectorsZ(X, 0.0, 1.0, d->V, d->size_H, &d->pX[d->ldpX*r_s],
                               d->ldpX, d->size_H, r_e-r_s); CHKERRQ(ierr);
  }

  /* nX[i] <- ||X[i]|| */
  if (d->correctXnorm) for(i=0; i<r_e-r_s; i++) {
    ierr = VecNorm(X[i], NORM_2, &d->nX[r_s+i]); CHKERRQ(ierr);
  } else for(i=0; i<r_e-r_s; i++) {
    d->nX[r_s+i] = 1.0;
  }

  PetscFunctionReturn(0);
}


#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_Y"
PetscErrorCode dvd_calcpairs_Y(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                               Vec *Y)
{
  PetscInt        i, ldpX = d->pY?d->ldpY:d->ldpX;
  PetscErrorCode  ierr;
  Vec             *V = d->W?d->W:d->V;
  PetscScalar     *pX = d->pY?d->pY:d->pX;

  PetscFunctionBegin;

  /* Y = V * pX(0:n-1) */
  if (DVD_IS(d->pX_type,DVD_MAT_IDENTITY)) {
    if (V != Y) for(i=r_s; i<r_e; i++) {
      ierr = VecCopy(V[i], Y[i]); CHKERRQ(ierr);
    }
  } else {
    ierr = SlepcUpdateVectorsZ(Y, 0.0, 1.0, V, d->size_H, &pX[ldpX*r_s], ldpX,
                               d->size_H, r_e-r_s); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_res_0"
/* Compute the residual vectors R(i) <- (AV - BV*eigr(i))*pX(i), and also
   the norm, where
   i <- r_s..r_e,
   UL, auxiliar scalar matrix of size size_H*(r_e-r_s),
   auxV, auxiliar global vector.
*/
PetscErrorCode dvd_calcpairs_res_0(dvdDashboard *d, PetscInt r_s, PetscInt r_e,
                             Vec *R, PetscScalar *UL, Vec auxV)
{
  PetscInt        i, j;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* If the eigenproblem is not reduced to standard */
  if ((d->B == PETSC_NULL) || DVD_ISNOT(d->sEP, DVD_EP_STD)) {
    /* UL = f.U(0:n-1) * diag(f.pL(0:n-1)) */
    for(i=r_s; i<r_e; i++) for(j=0; j<d->size_H; j++)
      UL[d->size_H*(i-r_s)+j] = d->pX[d->ldpX*i+j]*d->eigr[i];

    if (d->B == PETSC_NULL) {
      /* R <- V * UL */
      ierr = SlepcUpdateVectorsZ(R, 0.0, 1.0, d->V, d->size_V, UL, d->size_H,
                                 d->size_H, r_e-r_s); CHKERRQ(ierr);
    } else {
      /* R <- BV * UL */
      ierr = SlepcUpdateVectorsZ(R, 0.0, 1.0, d->BV, d->size_BV, UL,
                                 d->size_H, d->size_H, r_e-r_s);
      CHKERRQ(ierr);
    }
    /* R <- AV*U - R */
    ierr = SlepcUpdateVectorsZ(R, -1.0, 1.0, d->AV, d->size_AV,
                               &d->pX[d->ldpX*r_s], d->ldpX, d->size_H, r_e-r_s);
    CHKERRQ(ierr);

  /* If the problem was reduced to standard, R[i] = B*X[i] */
  } else {
    /* R[i] <- R[i] * eigr[i] */
    for(i=r_s; i<r_e; i++) {
      ierr = VecScale(R[i-r_s], d->eigr[i]); CHKERRQ(ierr); 
    }
      
    /* R <- AV*U - R */
    ierr = SlepcUpdateVectorsZ(R, -1.0, 1.0, d->AV, d->size_AV,
                               &d->pX[d->ldpX*r_s], d->ldpX, d->size_H, r_e-r_s);
    CHKERRQ(ierr);
  }

  ierr = d->calcpairs_proj_res(d, r_s, r_e, R); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_proj_res"
PetscErrorCode dvd_calcpairs_proj_res(dvdDashboard *d, PetscInt r_s,
                                      PetscInt r_e, Vec *R)
{
  PetscInt        i;
  PetscErrorCode  ierr;
  PetscBool       lindep;
  Vec             *cX;

  PetscFunctionBegin;

  /* If exists the BcX, R <- orth(BcX, R), nR[i] <- ||R[i]|| */
  if (d->BcX)
    cX = d->BcX;

  /* If exists left subspace, R <- orth(cY, R), nR[i] <- ||R[i]|| */
  else if (d->cY)
    cX = d->cY;

  /* If fany configurations, R <- orth(cX, R), nR[i] <- ||R[i]|| */
  else if (!(DVD_IS(d->sEP, DVD_EP_STD) && DVD_IS(d->sEP, DVD_EP_HERMITIAN)))
    cX = d->cX;

  /* Otherwise, nR[i] <- ||R[i]|| */
  else
    cX = PETSC_NULL;

  if (cX) for (i=0; i<r_e-r_s; i++) {
    ierr = IPOrthogonalize(d->ipI, 0, PETSC_NULL, d->size_cX, PETSC_NULL,
                           cX, R[i], PETSC_NULL, &d->nR[r_s+i], &lindep);
    CHKERRQ(ierr);
    if(lindep || (d->nR[r_s+i] < PETSC_MACHINE_EPSILON)) {
      ierr = PetscInfo2(d->eps,"The computed eigenvector residual %D is too low, %G!\n",r_s+i,d->nR[r_s+i]);CHKERRQ(ierr);
    }

  } else for(i=0; i<r_e-r_s; i++) {
    ierr = VecNorm(R[i], NORM_2, &d->nR[r_s+i]); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

/**** Patterns implementation *************************************************/
#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_updateMatV"
PetscErrorCode dvd_calcpairs_updateMatV(Mat A, Vec **AV, PetscInt *size_AV,
                                        PetscBool doUpdate, PetscBool doNew,
                                        dvdDashboard *d)
{
  PetscInt        i;
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* f.AV((0:f.V_tra.size)+f.imm.s) = f.AV * f.U(f.V_tra) */
  if (doUpdate) {
    if (d->MT_type == DVD_MT_pX) {
      ierr = SlepcUpdateVectorsZ(*AV+d->V_imm_e, 0.0, 1.0, *AV, *size_AV,
                                 &d->pX[d->ldpX*d->V_tra_s], d->ldpX,
                                 *size_AV, d->V_tra_e-d->V_tra_s); CHKERRQ(ierr);
    } else if (d->MT_type == DVD_MT_ORTHO) {
      ierr = SlepcUpdateVectorsZ(*AV+d->V_imm_e, 0.0, 1.0, *AV, *size_AV,
                                 &d->MTX[d->ldMTX*d->V_tra_s], d->ldMTX,
                                 *size_AV, d->V_tra_e-d->V_tra_s); CHKERRQ(ierr);
    }
  }
  *AV = *AV+d->V_imm_s;

  /* f.AV(f.V_new) = A*f.V(f.V_new) */
  if (d->V_imm_e-d->V_imm_s + d->V_tra_e-d->V_tra_s != d->V_new_s) {
    SETERRQ(((PetscObject)A)->comm,1, "Incompatible dimensions");
  }

  if (doNew) for (i=d->V_new_s; i<d->V_new_e; i++) {
    ierr = MatMult(A, d->V[i], (*AV)[i]); CHKERRQ(ierr);
  }
  *size_AV = d->V_new_e;

  PetscFunctionReturn(0);
}

#undef __FUNCT__ 
#define __FUNCT__ "dvd_calcpairs_WtMatV_gen"
/*
  Compute f.H = [MTY'*H*MTX     W(tra)'*V(new);
                 W(new)'*V(tra) W(new)'*V(new) ]
  where
  tra = 0:cMT-1,
  new = cMT:size_V-1,
  ldH, the leading dimension of H,
  auxS, auxiliary scalar vector of size ldH*max(tra,size_V),
  */
PetscErrorCode dvd_calcpairs_WtMatV_gen(PetscScalar **H, MatType_t sH,
  PetscInt ldH, PetscInt *size_H, PetscScalar *MTY, PetscInt ldMTY,
  PetscScalar *MTX, PetscInt ldMTX, PetscInt rMT, PetscInt cMT, Vec *W,
  Vec *V, PetscInt size_V, PetscScalar *auxS, DvdReduction *r,
  DvdMult_copy_func *sr, dvdDashboard *d)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;

  /* H <- MTY^T * (H * MTX) */
  if (cMT > 0) {
    ierr = SlepcDenseMatProdTriang(auxS, 0, ldH,
                                   *H, sH, ldH, *size_H, *size_H, PETSC_FALSE,
                                   MTX, 0, ldMTX, rMT, cMT, PETSC_FALSE);
    CHKERRQ(ierr);
    ierr = SlepcDenseMatProdTriang(*H, sH, ldH,
                                   MTY, 0, ldMTY, rMT, cMT, PETSC_TRUE,
                                   auxS, 0, ldH, *size_H, cMT, PETSC_FALSE);
    CHKERRQ(ierr);
    *size_H = cMT;
  }

  /* H = [H              W(tra)'*W(new);
          W(new)'*V(tra) W(new)'*V(new) ] */
  ierr = VecsMultS(*H, sH, ldH, W, *size_H, size_V, V, *size_H, size_V, r, sr);
  CHKERRQ(ierr);
  *size_H = size_V;

  PetscFunctionReturn(0);
}
