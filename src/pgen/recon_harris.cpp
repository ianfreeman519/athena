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
Real a, rho0, drho, nx, P0, b0, bz, L, d;
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

  a = pin->GetOrAddReal("problem", "a", 0.0);
  rho0 = pin->GetOrAddReal("problem", "rho0", 2.0);
  drho = pin->GetOrAddReal("problem", "drho", 1.0);
  nx = pin->GetOrAddReal("problem", "nx", 10);
  P0 = pin->GetOrAddReal("problem", "P0", 1.0);
  b0 = pin->GetOrAddReal("problem", "b0", 1.0);
  bz = pin->GetOrAddReal("problem", "bz", 0.0);
  L = pin->GetOrAddReal("problem", "L", 1.0);
  d = pin->GetOrAddReal("problem", "d", 0.1);


  if (Globals::my_rank == 0 && ncycle == 0) {
    std::cout << std::endl
    << "--- Input parameters of the simulation ---" << std::endl
    << "a (Bx=tanh(ax)) = " << a << std::endl
    << "rho0 = "    << rho0 << std::endl
    << "drho = "    << drho << std::endl
    << "nx = "      << nx   << std::endl
    << "P0 = "      << P0   << std::endl
    << "b0 = "      << b0   << std::endl
    << "L = "       << L    << std::endl
    << "d = "       << d    << std::endl
    << "bz = "      << bz   << std::endl
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
        Real x, y;
        x = pcoord->x1v(i);
        y = pcoord->x2v(j);

        // Density profile is constant initial density + pertubation(x) * activation(y)
        Real activation = 0.5 - 0.5*std::tanh(a*(std::abs(y) - d));
        Real perturbate = drho * std::cos(nx*M_PI*x);
        Real rho = rho0 + activation * perturbate;
        Real P   = P0;

        phydro->u(IDN,k,j,i) = rho;

        // x-direction momentum
        phydro->u(IM1,k,j,i) = 0.0;
        phydro->u(IM2,k,j,i) = 0.0;
        phydro->u(IM3,k,j,i) = 0.0;

        // Internal energy is initially kinetic energy plus magnetic energy
        // phydro->u(IEN,k,j,i) = P/gm1 + 0.5*rho*SQR(v0) + 0.5*b^2;
        Real bx = b0 * std::tanh(a * y);
        phydro->u(IEN,k,j,i) = P0/gm1 + 0.5*(bx*bx + bz*bz);
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real y;
          y = pcoord->x2v(j);
          pfield->b.x1f(k,j,i) = b0 * std::tanh(a * y);
        }
      }
    }
    // Y-direction face-centered B-field
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {          
          pfield->b.x2f(k,j,i) = 0.0;
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
    // if (NON_BAROTROPIC_EOS) {
    //   for (int k=ks; k<=ke; k++) {
    //     for (int j=js; j<=je; j++) {
    //       for (int i=is; i<=ie; i++) {
    //         phydro->u(IEN,k,j,i) +=
    //             // second-order accurate assumption about volume-averaged field
    //             0.5*0.25*(SQR(pfield->b.x1f(k,j,i) + pfield->b.x1f(k,j,i+1))
    //                       + SQR(pfield->b.x2f(k,j,i)  + pfield->b.x2f(k,j+1,i))
    //                       + SQR(pfield->b.x3f(k,j,i) + pfield->b.x3f(k+1,j,i)));
    //       }
    //     }
    //   }
    // }
  }
  return;
}
