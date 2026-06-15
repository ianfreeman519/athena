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
//! rho  = 1    [g/cm^3]
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
Real gm1, wm, APiOL;  // APiOL=A0*Pi/L, wm=2pi/L
Real rho0, P0, A0, L, bz, vin0;
Real powRho, powP, powv;
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
  A0 = pin->GetReal("problem", "A0");
  bz = pin->GetOrAddReal("problem", "bz", 2.0);
  rho0  = pin->GetReal("problem", "rho0");
  P0 =  pin->GetReal("problem", "P0");
  L = pin->GetReal("problem", "L");
  vin0 = pin->GetOrAddReal("problem", "vin0", 0.0);
  powRho = pin->GetOrAddReal("problem", "powRho", 2.0);
  powP = pin->GetOrAddReal("problem", "powP", 2.0);
  powv = pin->GetOrAddReal("problem", "powv", 0.0);

  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
    << "--- Input parameters of the simulation ---" << std::endl
    << "A0 = " << A0 << std::endl
    << "rho0 = " << rho0 << std::endl
    << "powRho = " << powRho << std::endl
    << "L = " << L << std::endl
    << "P0 = " << P0 << std::endl
    << "powP = " << powP << std::endl
    << "vin0 = " << vin0 << std::endl
    << "powv = " << powv << std::endl
    << "eta_ohm = " << pin->GetReal("problem", "eta_ohm") << std::endl
    << "bz = " << bz << std::endl; 
  }

  return;
}


//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for zpinch problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gm1 = peos->GetGamma() - 1.0;
  wm = 2*M_PI/L;
  APiOL = A0*M_PI/L;

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
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);

        Real rho = rho0 * std::pow(std::abs(y), powRho);

        phydro->u(IDN,k,j,i) = rho;

        // x-, z-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // y- direction momentum
        Real sgn_y = (y > 0.0) ? 1.0 : (y < 0.0) ? -1.0 : 0.0;
        Real vy = -1.0 * sgn_y * vin0 * std::pow(y, powv);
        phydro->u(IM2,k,j,i) = vy;

        // Internal energy is initially kinetic energy + magnetic energy + pressure profile
        Real bxmtanh, by;
        bxmtanh =  - 20*A0*std::exp(-10*y*y) * y * std::cos(4*wm*x) * std::cos(wm*y)
              - 2*APiOL*std::exp(-10*y*y) * std::cos(4*wm*x) * std::sin(wm*y); // + 3*std::tanh(2*y);
        by = 8*APiOL * std::exp(-10*y*y) * std::cos(wm*y) * std::sin(4*wm*x);
        Real P   = P0 + std::pow(std::abs(y), powP);

        phydro->u(IEN,k,j,i) = P + bxmtanh*bxmtanh + by*by + bz*bz + rho*vy*vy/2;
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x,y;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          pfield->b.x1f(k,j,i) = 3*std::tanh(2*y) - 20*A0*std::exp(-10*y*y) * y * std::cos(4*wm*x) * std::cos(wm*y)
              - 2*APiOL*std::exp(-10*y*y) * std::cos(4*wm*x) * std::sin(wm*y);
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x, y;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          pfield->b.x2f(k,j,i) = 8*APiOL * std::exp(-10*y*y) * std::cos(wm*y) * std::sin(4*wm*x);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          pfield->b.x3f(k,j,i) = bz;
        }
      }
    }
  }
  return;
}
