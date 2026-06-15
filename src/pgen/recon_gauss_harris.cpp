//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_fast.cpp
//! \brief Basic reconnection initialized with initial conditions "based on vibes"
//! REFERENCES:
//! 

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
Real rho0, P0, B0, delta, w, powP;
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
  B0 = pin->GetOrAddReal("problem", "B0", 1.0);
  rho0  = pin->GetOrAddReal("problem", "rho0", 1.0);
  P0 =  pin->GetOrAddReal("problem", "P0", 2.0);
  w = pin->GetOrAddReal("problem", "w", 0.5);
  delta = pin->GetOrAddReal("problem", "delta", 0.5);
  powP = pin->GetOrAddReal("problem", "powP", 2.0);

  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
    << "--- Input parameters of the simulation ---" << std::endl
    << "B0 = " << B0 << std::endl
    << "rho0 = " << rho0 << std::endl
    << "w = " << w << std::endl
    << "delta = " << delta << std::endl
    << "P0 = " << P0 << std::endl
    << "powP = " << powP << std::endl
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
        Real x, y;
        Real Bx = 0.0;
        Real By = 0.0;
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);

        Real rho = rho0;

        phydro->u(IDN,k,j,i) = rho;

        // x-, z-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;

        // Internal energy is initially magnetic energy + pressure profile
        for (int xi=-1; xi<=1; xi+=2){
          Bx += 2*M_PI*w*y/delta * std::exp(-std::pow(4*(x-xi)/w, 2) - std::pow(4*y/delta, 2)) + std::tanh(y/delta);
          By += 2*M_PI*delta/w * (xi - x) * std::exp(-std::pow(4*(x-xi)/w, 2) - std::pow(4*y/delta, 2));
        }
        // Real Pvar = delta / powP * std::pow(std::abs(y)/delta, powP);
        Real P   = (P0 + std::pow(std::abs(y), powP))/gm1 + 0.5*(Bx*Bx + By*By);

        phydro->u(IEN,k,j,i) = P;
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x,y;
          Real b = 0.0;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          for (int xi=-1; xi<=1; xi+=2){
            b += 2*M_PI*w*y/delta * std::exp(-std::pow(4*(x-xi)/w, 2) - std::pow(4*y/delta, 2)) + std::tanh(y/delta);
          }
          pfield->b.x1f(k,j,i) = b;
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x,y;
          Real b = 0.0;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          for (int xi=-1; xi<=1; xi+=2){
            b += 2*M_PI*delta/w * (xi - x) * std::exp(-std::pow(4*(x-xi)/w, 2) - std::pow(4*y/delta, 2));
          }          
          pfield->b.x2f(k,j,i) = b;
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
