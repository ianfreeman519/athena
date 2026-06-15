//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_harris.cpp
//! \brief Basic reconnection initialized with harris current sheet
//!
//! 2D collapse on center, x distance to center
//! initial conditions with x in cm:
//! rho  = rho0*sech2(z/L) [g/cm^3]
//! u    = v0*rho    [g/cm^2/s]
//! Bz   = B0 tanh(z/L)   [gauss] axial
//! pressure = 1.E-6*B^2   actually zero in the exact solution
//!
//! Can apply sine wave perturbation in a form
//!   \f$ (1+perturb*std::cos(mphi*phi)) \f$ to the magnetic potential Az
//!
//! REFERENCES:
//! 1) Treumann and Baumjohann 2013

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
Real rho0, w1x, w2x, b0, L, width, n_wire;
// Real nu_iso, eta_ohm;
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
  // initialize global variables
  // nu_iso = pin->GetOrAddReal("problem", "nu_iso", 0.0);
  // eta_ohm = pin->GetOrAddReal("problem", "eta_ohm", 0.0);

  rho0  = pin->GetReal("problem", "rho0");
  w1x  = pin->GetReal("problem", "w1x");
  w2x  = pin->GetReal("problem", "w2x");
  L = pin->GetReal("problem", "L");
  width = pin->GetReal("problem","wire_width");
  n_wire = pin->GetReal("problem","n_wire");
  // convert from CGS to Athena Heaviside units:
  b0 = pin->GetReal("problem", "b0")/std::sqrt(4*M_PI);  
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

  // initialize vector potential for inflowing B
  // (only initializing 2D array for vec potential)
  AthenaArray<Real> az;
  az.NewAthenaArray(ncells2, ncells1);  // ncells2 is consistent only if 2D or 3D

  // initialize conserved variables
  for (int k=ks; k<=ke; k++) {  // Loop over Z
    for (int j=js; j<=je; j++) {  // Loop over Y
      for (int i=is; i<=ie; i++) {  // Loop over X
        
        Real rho = 0.0;     // Default density
        Real P0 = 4*M_PI*rho0*gm1*(b0*b0);;   // Magnetic default pressure pressure
        
        // Loop over n_wire to check if the cell falls within any of the wire regions
        for (int n=0; n<n_wire; n++) {
          Real x1 = pcoord->x1v(i);     // X position
          Real x2 = pcoord->x2v(j);     // Y position
          Real y_wire = L * n / (n_wire - 1);  // Y position of the nth wire

          // Check if the cell lies within the nth wire region
          if (std::abs(x1 - w1x) <= width && std::abs(x2 - y_wire) <= width) {
            rho = rho0;   // Set the high density
            break;  // Exit the loop since this cell already belongs to a wire
          }
          else if (std::abs(x1 - w2x) <= width && std::abs(x2 - y_wire) <= width) {
            rho = rho0;   // Set the high density for the second wire
            break;  // Exit the loop, as this cell belongs to a wire
          }
        }

        // Assign the computed values to the conserved variables
        phydro->u(IDN, k,j,i) = rho;
        phydro->u(IEN, k,j,i) = P0; // + gm1*rho;
        
        // System starts at rest
        phydro->u(IM1, k, j, i) = 0.0;
        phydro->u(IM2, k, j, i) = 0.0;
        phydro->u(IM3, k, j, i) = 0.0;
        
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          pfield->b.x1f(k,j,i) = 0.0;
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x1;
          x1 = pcoord->x1v(i);
          pfield->b.x2f(k,j,i) = b0 * std::tanh(x1/L);
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
