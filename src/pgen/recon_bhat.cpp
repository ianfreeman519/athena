//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_bhat.cpp
//! \brief Basic reconnection initialized with initial conditions from Bhattacharjee 2009
//!
//! 2D collapse on center, x distance to center
//! initial conditions with x in cm:
//! rho  = 1 [g/cm^3]
//! u    = 0    [g/cm^2/s]
//! Bx   = -1.4cos(1.4pi y)sin(1.4pi x)   [gauss]
//! By   = 1.4cos(1.4pi x)sin(1.4pi y)    [gauss]
//! pressure = 6   actually zero in the exact solution
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
Real gm1, pmcoeff;  // pmcoeff=A0^2 Pi^2/L^2, wm=2pi/L
Real rho0, P0, A0, L;
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

  // convert from CGS to Athena Heaviside units:
  A0 = pin->GetReal("problem", "A0"); // / std::sqrt(4*M_PI);
  rho0  = pin->GetReal("problem", "rho0");
  P0 =  pin->GetReal("problem", "P0");
  L = pin->GetReal("problem", "L");

  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
    << "--- Input parameters of the simulation ---" << std::endl
    << "A0 = " << A0 << std::endl
    << "rho0 = " << rho0 << std::endl
    << "L = " << L << std::endl
    << "P0 = " << P0 << std::endl
    << "eta_ohm = " << pin->GetReal("problem", "eta_ohm") << std::endl; 
  }

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
    msg << "### FATAL ERROR in recon_bhat.cpp ProblemGenerator" << std::endl
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

        Real rho = rho0;

        phydro->u(IDN,k,j,i) = rho;

        // x-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;

        // y-, z-direction momentum is 0
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // Internal energy is initially thermal and magnetic energy
        Real bx = -2*A0*M_PI / L * std::cos(2*M_PI*x2/L) * std::sin(2*M_PI*x1/L);
        Real by = 2*A0*M_PI / L * std::cos(2*M_PI*x1/L) * std::sin(2*M_PI*x2/L);
        Real P = P0/gm1 + 0.5*(bx*bx + by*by);
        phydro->u(IEN,k,j,i) = P;
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x1,x2;
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          pfield->b.x1f(k,j,i) = -2*A0*M_PI / L * std::cos(2*M_PI*x2/L) * std::sin(2*M_PI*x1/L);
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x1,x2;
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          pfield->b.x2f(k,j,i) = 2*A0*M_PI / L * std::cos(2*M_PI*x1/L) * std::sin(2*M_PI*x2/L);
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
  }
  return;
}
