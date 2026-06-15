//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_GEM_harris.cpp
//! \brief Reconnection initialized with harris current sheet and magnetic island in center
//!
//! 2D collapse on center, x distance to center
//! initial conditions with x in cm:
//! rho  = rho0*sech2(z/L) [g/cm^3]
//! u    = v0*rho    [g/cm^2/s]
//! Bz   = B0 tanh(z/L)   [gauss] axial
//! pressure = 1.E-6*B^2   actually zero in the exact solution
//!
//!
//! REFERENCES:
//! 1) GEM Challenge, 1999, Birn

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt(), cosh(), tanh()
#include <cstring>    // strcmp()
#include <sstream>
#include <stdexcept>
#include <string>

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/bvals.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../field/field.hpp"
#include "../hydro/hydro.hpp"
#include "../mesh/mesh.hpp"
#include "../parameter_input.hpp"

namespace {
Real gm1;
Real B0, rho0, lambda, rho1, phi0;
Real xmin, xmax, ymin, ymax, Lx, Ly, P0;
} // namespace

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // convert from CGS to Athena Heaviside units:
  B0 = pin->GetReal("problem", "B0")/std::sqrt(4*M_PI);
  rho0 = pin->GetOrAddReal("problem", "rho0", 1);
  rho1 = pin->GetOrAddReal("problem", "rho1", 0.2);
  lambda = pin->GetOrAddReal("problem", "lambda", 0.5);
  phi0 = pin->GetOrAddReal("problem", "phi0", 0.1);
  // Pulling Lx and Ly from input file
  xmin = pin->GetReal("mesh", "x1min");
  xmax = pin->GetReal("mesh", "x1max");
  ymin = pin->GetReal("mesh", "x2min");
  ymax = pin->GetReal("mesh", "x2max");
  // Calculating Lx and Ly from input file
  Lx = xmax - xmin;
  Ly = ymax - ymin;
  Lx = 12.8;
  Ly = 25.6;
  // Pressure is initially uniform, scaled with flat B outside current sheet
  P0 = B0*B0/(8*M_PI);
  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for zpinch problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gm1 = peos->GetGamma() - 1.0;

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in magnoh.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl
        << "Only Cartesian is supported for this problem" << std::endl;
    ATHENA_ERROR(msg);
  }

  // initialize conserved variables
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // Volume centered coordinates and quantities
        Real x1,x2;
        x1 = pcoord->x1v(i);
        x2 = pcoord->x2v(j);

        Real rho = rho0 / std::pow(std::cosh(x1/lambda), 2) + rho1;
        
        phydro->u(IDN,k,j,i) = rho;

        // x-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // Internal energy is initially kinetic energy plus magnetic energy
        // phydro->u(IEN,k,j,i) = P/gm1 + 0.5*rho*SQR(v0);
        phydro->u(IEN,k,j,i) = P0/gm1;
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x1, x2;
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          pfield->b.x1f(k,j,i) = phi0*(M_PI/Ly)*std::cos(2*M_PI*x1/Lx)*std::sin(M_PI*x2/Ly);
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x1, x2;
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          pfield->b.x2f(k,j,i) = B0 * std::tanh(x1/lambda) 
                  - phi0*(2*M_PI/Lx)*std::sin(2*M_PI*x1/Lx)*std::cos(M_PI*x2/Ly);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = 0.0;
        }
      }
    }
    if (NON_BAROTROPIC_EOS) {
      for (int k=ks; k<=ke; k++) {
        for (int j=js; j<=je; j++) {
          for (int i=is; i<=ie; i++) {
            phydro->u(IEN,k,j,i) +=
                // second-order accurate assumption about volume-averaged field
                0.5*0.25*(SQR(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))
                          + SQR(pfield->b.x2f(k,j,i)  + pfield->b.x2f(k,j+1,i))
                          + SQR(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)));
          }
        }
      }
    }
  }
  return;
}
