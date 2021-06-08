/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2021, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

#if !defined(SLEPCCONTOUR_H)
#define SLEPCCONTOUR_H

#include <slepc/private/slepcimpl.h>
#include <petscksp.h>

/* Data structures and functions for contour integral methods (used in several classes) */
struct _n_SlepcContourData {
  PetscObject  parent;     /* parent object */
  PetscSubcomm subcomm;    /* subcommunicator for top level parallelization */
  PetscInt     npoints;    /* number of integration points assigned to the local subcomm */
  KSP          *ksp;       /* ksp array for storing factorizations at integration points */
  Mat          *pA;        /* redundant copies of the matrices in the local subcomm */
  PetscInt     nmat;       /* number of matrices in pA */
  Vec          xsub;       /* aux vector with parallel layout as redundant Mat */
  Vec          xdup;       /* aux vector with parallel layout as original Mat (with contiguous order) */
  VecScatter   scatterin;  /* to scatter from regular vector to xdup */
};
typedef struct _n_SlepcContourData* SlepcContourData;

SLEPC_EXTERN PetscErrorCode SlepcContourDataCreate(PetscInt,PetscInt,PetscObject,SlepcContourData*);
SLEPC_EXTERN PetscErrorCode SlepcContourDataReset(SlepcContourData);
SLEPC_EXTERN PetscErrorCode SlepcContourDataDestroy(SlepcContourData*);

SLEPC_EXTERN PetscErrorCode SlepcContourRedundantMat(SlepcContourData,PetscInt,Mat*);
SLEPC_EXTERN PetscErrorCode SlepcContourScatterCreate(SlepcContourData,Vec);

#endif
