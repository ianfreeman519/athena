//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file recon_fast.cpp
//! \brief Basic reconnection initialized with initial conditions "based on vibes"
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
Real gm1;
Real rho0, P0, B0, L, delta, bz, vin;
Real powrho, powP, powv;
Real Nx;
} // namespace

#if !MAGNETIC_FIELDS_ENABLED
#error "This problem generator requires magnetic fields"
#endif

Real activation(Real y, Real d) {
  Real out = 0.5 - 0.5*std::tanh(50*(std::abs(y) - 0.8*d));
  return out;
}

//========================================================================================
//! \fn void Mesh::InitUserMeshData(ParameterInput *pin)
//  \brief Function to initialize problem-specific data in mesh class.  Can also be used
//  to initialize variables which are global to (and therefore can be passed to) other
//  functions in this file.  Called in Mesh constructor.
//========================================================================================

void Mesh::InitUserMeshData(ParameterInput *pin) {
  // initialize global variables
  B0 = pin->GetOrAddReal("problem", "B0", 1.0);
  bz = pin->GetOrAddReal("problem", "bz", 0.0);
  rho0  = pin->GetOrAddReal("problem", "rho0", 1.0);
  P0 =  pin->GetOrAddReal("problem", "P0", 2.0);
  L = pin->GetOrAddReal("problem", "L", 1.0);
  Nx = pin->GetOrAddReal("problem", "Nx", 4.0);
  vin = pin->GetOrAddReal("problem", "vin", 0.0);
  powv = pin->GetOrAddReal("problem", "powv", 0.0);
  powrho = pin->GetOrAddReal("problem", "powrho", 0.0);
  powP = pin->GetOrAddReal("problem", "powP", 0.0);
  delta = pin->GetOrAddReal("problem", "delta", L);

  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
    << "--- Input parameters of the simulation ---" << std::endl
    << "B0 = " << B0 << std::endl
    << "vin = " << vin << std::endl
    << "powv = " << powv << std::endl
    << "Nx (loops) = " << Nx << std::endl
    << "rho0 = " << rho0 << std::endl
    << "L = " << L << std::endl
    << "delta = " << delta << std::endl
    << "P0 = " << P0 << std::endl
    << "powP = " << powP << std::endl
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
        Real x, y, Bx, By;
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);

        Real rho = rho0 * std::pow(std::abs(y), powrho);

        phydro->u(IDN,k,j,i) = rho;

        // x-, z-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // y- direction momentum
        Real sgn_y = (y > 0.0) ? 1.0 : (y < 0.0) ? -1.0 : 0.0;
        Real vy = -1.0 * sgn_y * vin * std::pow(std::abs(y), powv);
        phydro->u(IM2,k,j,i) = rho * vy;

        // Internal energy is initially kinetic energy + magnetic energy + pressure profile
        Bx = -1.0 * B0 * std::cos(Nx*M_PI*x / L) * std::sin(2*M_PI*y / delta) * activation(y, delta) + std::tanh(10*y);
        By = 1.0 * B0 * std::cos(2*M_PI*y / delta) * std::sin(Nx*M_PI*x / L) * activation(y, delta);
        Real KE  = 0.5 * rho * vy*vy;
        // Real Pvar = delta / powP * std::pow(std::abs(y)/delta, powP);
        Real P   = KE + (P0 - 0.5*std::exp(-delta*y*y))/gm1 + 0.5*(Bx*Bx + By*By);

        phydro->u(IEN,k,j,i) = P;
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real x,y,b;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          b = -1.0 * B0 * std::cos(Nx*M_PI*x / L) * std::sin(2*M_PI*y / L) * activation(y, delta) + std::tanh(10*y);;
          pfield->b.x1f(k,j,i) = b;
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          Real x, y, b;
          x = pcoord->x1v(i);
          y = pcoord->x2v(j);
          b = 1.0 * B0 * std::cos(2*M_PI*y / L) * std::sin(Nx*M_PI*x / L) * activation(y, delta);
          pfield->b.x2f(k,j,i) = b;
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
