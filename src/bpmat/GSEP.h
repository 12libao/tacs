#ifndef TACS_GSEP_H
#define TACS_GSEP_H

#include "KSM.h"

/*!  
  The following contains classes required for computation of
  general, symmetric eigenvalue problems (GSEP).

  Copyright (c) 2010 Graeme Kennedy. All rights reserved.
  Not for commercial purposes.

  The Lanczos method is used to construct an orthonormal basis
  of a Krylov subspace. This method produces a tri-diagonal
  Hessenberg matrix. Approximate eigenvalues of the full matrix
  may be found by discarding the last row of the rectangular 
  - equivalent to projecting the matrix onto the orthonormal basis
  - and computing the eigenvalues of the much smaller system. 

  The method is implemented in a block-form such that operations
  are performed with a block-size of P.

  Solve the eigenvalue problem:

  A x = lambda B x
*/
class EPOperator : public TACSObject {
 public:
  virtual ~EPOperator(){}

  // Create a vector associated with this operator
  // ---------------------------------------------
  virtual TACSVec *createVec() = 0;

  // Compute y = A *x
  // -----------------
  virtual void mult( TACSVec *x, TACSVec *y ) = 0;

  // Compute the inner product <x,y>
  // -------------------------------
  virtual TacsScalar dot( TACSVec *x, TACSVec *y ){ return x->dot(y); }

  // Compute || B *x || - this is used to compute the eigenvalue error
  // ------------------------------------------------------------------
  virtual TacsScalar errorNorm( TACSVec *x ){ return 1.0; }

  // Convert the shifted spectrum back to the regular spectrum
  // ---------------------------------------------------------
  virtual TacsScalar convertEigenvalue( TacsScalar value ){ return value; }
};

/*!  
  Solve the regular eigenvalue problem without any shift/invert
  operations 
*/
class EPRegular : public EPOperator {
 public:
  EPRegular( TACSMat *_mat );
  ~EPRegular();

  TACSVec *createVec();
  void mult( TACSVec *x, TACSVec *y );

 private:
  TACSMat *mat;
};


/*!
  Shift and invert the spectrum
*/
class EPShiftInvert : public EPOperator {
 public:
  EPShiftInvert( TacsScalar _sigma, TACSKsm *_ksm );
  ~EPShiftInvert();

  TACSVec *createVec();
  void mult( TACSVec *x, TACSVec *y );

  // The eigenvalues are computed as mu = 1.0/( eig - sigma )
  // eig = 1.0/mu + sigma
  TacsScalar convertEigenvalue( TacsScalar value );

 private:
  TACSKsm *ksm;
  TacsScalar sigma;
};

/*!
  Solve the generalized eigenvalue problem,

  A x = lambda B x

  Here B must be positive definite. The following shifted
  problem is solved,

  ( A - sigma B )^{-1} B x = 1.0/( lambda - sigma ) x
  
  In this case the B-inner products must be used so inner = B.
*/
class EPGeneralizedShiftInvert : public EPOperator {
 public:
  EPGeneralizedShiftInvert( TacsScalar _sigma, TACSKsm *_ksm, 
			    TACSMat *_inner );
  ~EPGeneralizedShiftInvert();
  
  void setSigma( TacsScalar _sigma );
  TACSVec *createVec();
  void mult( TACSVec *x, TACSVec *y ); // Compute y = (A - sigma B)^{-1} * inner * x
  TacsScalar dot( TACSVec *x, TACSVec *y ); // Compute <x,y> = x^{T} * inner * y
  TacsScalar errorNorm( TACSVec *x );
  TacsScalar convertEigenvalue( TacsScalar value );

 private:
  TacsScalar sigma;  
  TACSKsm *ksm;
  TACSMat *inner;
  TACSVec *temp;
};

/*
  Solve a generalized eigenvalue problem for a buckling problem,
  
  A x = - lambda B x

  (Note the sign!) Here A = K, must be positive definite, but B 
  does not need to be. The shifted problem is,

  (A - sigma B)^{-1} A x = lambda/(lambda + sigma) x 

  Here A-inner products are required, so inner = A.
*/
class EPBucklingShiftInvert : public EPOperator {
 public:
  EPBucklingShiftInvert( TacsScalar _sigma, TACSKsm *_ksm, 
			 TACSMat *_inner );
  ~EPBucklingShiftInvert();
  
  void setSigma( TacsScalar _sigma );
  TACSVec *createVec();
  void mult( TACSVec *x, TACSVec *y ); // Compute y = (A - sigma B)^{-1} * inner * x
  TacsScalar dot( TACSVec *x, TACSVec *y ); // Compute <x,y> = x^{T} * inner * y
  TacsScalar errorNorm( TACSVec *x );
  TacsScalar convertEigenvalue( TacsScalar value );

 private:
  TacsScalar sigma;  
  TACSKsm *ksm;
  TACSMat *inner;
  TACSVec *temp;
};

/*
  The symmetric eigenvalue solver
*/
class SEP : public TACSObject {
 public:
  // Set the type of orthogonalization to use
  enum OrthoType { FULL, LOCAL };
  enum EigenSpectrum { SMALLEST, LARGEST, 
		       SMALLEST_MAGNITUDE, LARGEST_MAGNITUDE };

  SEP( EPOperator *_Op, int _max_iters, 
       enum OrthoType _ortho_type=FULL );
  ~SEP();

  void setOrthoType( enum OrthoType _ortho_type );
  void setTolerances( double _tol, 
		      enum EigenSpectrum _spectrum, int _neigvals );
  void setOperator( EPOperator *_Op );
  void solve( KSMPrint *ksm_print = NULL );
  TacsScalar extractEigenvalue( int n, TacsScalar *error );
  TacsScalar extractEigenvector( int n, TACSVec *ans, TacsScalar *error );

  TacsScalar checkOrthogonality();
  void printOrthogonality();

 private:
  void sortEigenvalues( TacsScalar *values, int neigs, int *permutation );

  // Check whether the right eigenvalues have converged
  int checkConverged( TacsScalar *A, TacsScalar *B, int n );			
  
  // Data used to determine which spectrum to use and when
  // enough eigenvalues are converged
  double tol; 
  enum EigenSpectrum spectrum;
  int neigvals;

  // The eigenproblem operator object
  EPOperator *Op;

  // The number of iterations completed thus far
  int niters;
  
  // The coefficients of the symmetric tridiagonal matrix
  TacsScalar *Alpha, *Beta;

  // Eigenvalues/eigenvectors of the tridiagonal matrix
  TacsScalar *eigs, *eigvecs;

  // Permutation yielding an ascending ordering of the eigenvalues after 
  // the shift has been applied
  int *perm; 

  enum OrthoType ortho_type;

  int max_iters;
  TACSVec **Q; // The Vectors for the eigenvalue problem...
};

#endif
