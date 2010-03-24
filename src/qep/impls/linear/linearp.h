/*                       

   Private header for QEPLINEAR.

   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2009, Universidad Politecnica de Valencia, Spain

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

typedef struct {
  PetscTruth explicitmatrix;
  PetscInt   cform;           /* companion form */
  Mat        A,B;             /* matrices of generalized eigenproblem */
  EPS        eps;             /* linear eigensolver for Az=lBz */
  Mat        M,C,K;           /* copy of QEP coefficient matrices */
  Vec        x1,x2,y1,y2;     /* work vectors */
} QEP_LINEAR;

/* N1 */
EXTERN PetscErrorCode MatMult_QEPLINEAR_N1A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_N1B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N1A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N1B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N1A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N1B(MPI_Comm,QEP_LINEAR*,Mat*);

/* N2 */
EXTERN PetscErrorCode MatMult_QEPLINEAR_N2A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_N2B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N2A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_N2B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N2A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_N2B(MPI_Comm,QEP_LINEAR*,Mat*);

/* S1 */
EXTERN PetscErrorCode MatMult_QEPLINEAR_S1A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_S1B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_S1A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_S1B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_S1A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_S1B(MPI_Comm,QEP_LINEAR*,Mat*);

/* S2 */
EXTERN PetscErrorCode MatMult_QEPLINEAR_S2A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_S2B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_S2A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_S2B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_S2A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_S2B(MPI_Comm,QEP_LINEAR*,Mat*);

/* H1 */
EXTERN PetscErrorCode MatMult_QEPLINEAR_H1A(Mat,Vec,Vec);
EXTERN PetscErrorCode MatMult_QEPLINEAR_H1B(Mat,Vec,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_H1A(Mat,Vec);
EXTERN PetscErrorCode MatGetDiagonal_QEPLINEAR_H1B(Mat,Vec);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_H1A(MPI_Comm,QEP_LINEAR*,Mat*);
EXTERN PetscErrorCode MatCreateExplicit_QEPLINEAR_H1B(MPI_Comm,QEP_LINEAR*,Mat*);

