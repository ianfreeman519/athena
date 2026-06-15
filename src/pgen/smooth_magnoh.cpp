//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file magnoh.cpp
//! \brief Magnetized Noh with perturbation in B_phi. Authored by A. Beresnyak
//!
//! 2D collapse on center, r distance to center
//! initial conditions with r in cm:
//! rho  = rho0*r^alpha [g/cm^3]
//! V    = V0    [cm/s]
//! Bphi = Bphi0*r^beta   [gauss] azimuthal
//! Bz   = Bz0*  r^beta   [gauss] axial
//! pressure = 1.E-6*B^2   actually zero in the exact solution
//!
//! Can apply sine wave perturbation in a form
//!   \f$ (1+perturb*std::cos(mphi*phi)) \f$ to the magnetic potential Az
//!
//! REFERENCES:
//! 1) Velikovich, Giuliani, Zalesak, Gardiner, "Exact self-similar solutions for the
//! magnetized Noh Z pinch problem", Phys. of Plasmas, vol.19, p.012707 (2012)
//!
//! 2) Giuliani, Velikovich, Beresnyak, Zalesak, Gianakon, Rousculp, "Self-similar
//! solutions for the magnetized Noh problem with axial and azimuthal field", Phys. of
//! Plasmas, in prep (2018)

// C headers

// C++ headers
#include <algorithm>
#include <cmath>      // sqrt()
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
Real alpha, beta, rho0, P0, pcoeff, vr, perturb, mphi;
Real bphi0, bz;
Real c;
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

  alpha  = pin->GetReal("problem", "alpha");
  beta  = pin->GetReal("problem", "beta");
  pcoeff = pin->GetReal("problem", "pcoeff");
  rho0  = pin->GetReal("problem", "d");
  vr =  pin->GetReal("problem", "vr");
  // convert from CGS to Athena Heaviside units:
  bphi0 = pin->GetReal("problem", "bphi")/std::sqrt(4*M_PI);
  bz = pin->GetReal("problem", "bz")/std::sqrt(4*M_PI);
  P0 = 4*M_PI*pcoeff*(bphi0*bphi0+bz*bz);

  perturb = pin->GetOrAddReal("problem","perturb",0.0);
  mphi = pin->GetOrAddReal("problem","mphi",1.0);

  c = pin->GetOrAddReal("problem", "smooth_cutoff", 1.5);

  return;
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for zpinch problem
//========================================================================================

void MeshBlock::ProblemGenerator(ParameterInput *pin) {
  gm1 = peos->GetGamma() - 1.0;

  std::cout << "Writing Smooth Variables" << std::endl;

  if (std::strcmp(COORDINATE_SYSTEM, "cartesian") != 0 &&
      std::strcmp(COORDINATE_SYSTEM, "cylindrical") != 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in magnoh.cpp ProblemGenerator" << std::endl
        << "Unrecognized COORDINATE_SYSTEM= " << COORDINATE_SYSTEM << std::endl
        << "Only Cartesian and cylindrical are supported for this problem" << std::endl;
    ATHENA_ERROR(msg);
  }

  // initialize vector potential for inflowing B
  // (only initializing 2D array for vec potential)
  AthenaArray<Real> az;
  az.NewAthenaArray(ncells2, ncells1);  // ncells2 is consistent only if 2D or 3D

  for (int j=js; j<=je+1; ++j) {
    for (int i=is; i<=ie+1; ++i) {
      Real rad,phi;
      if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
        rad = pcoord->x1f(i);
        phi = pcoord->x2f(j);
      } else { // cartesian
        Real x1  = pcoord->x1f(i);
        Real x2  = pcoord->x2f(j);
        rad = std::sqrt(SQR(x1) + SQR(x2));
        phi = atan2(x2, x1);
      }

      Real azr;
      if (rad <= c) {azr = (bphi0/(beta+1))*std::pow(rad,beta+1)*(1+perturb*std::cos(mphi*phi));}
      else {azr = std::tanh((beta+1)*pow(c, beta)*(rad-c)) + (bphi0/(beta+1))*std::pow(c,beta+1)*(1+perturb*std::cos(mphi*phi));}
      
      // Populating a smoothed az(j,i)
      az(j,i) = azr;
    }
  }

  // initialize conserved variables
  for (int k=ks; k<=ke; k++) {
    for (int j=js; j<=je; j++) {
      for (int i=is; i<=ie; i++) {
        // Volume centered coordinates and quantities
        Real rad,x1,x2;
        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          rad = pcoord->x1v(i);
        } else { // cartesian
          x1 = pcoord->x1v(i);
          x2 = pcoord->x2v(j);
          rad = std::sqrt(SQR(x1) + SQR(x2));
        }
        // Smoothed Field values at current loop x, y, z
        Real rho, P;
        if (rad <= c) {
          rho = rho0*std::pow(rad, alpha);
          P = P0  *std::pow(rad, 2*beta);
          }
        else {
          rho = std::tanh(alpha*std::pow(c, alpha-1)*(rad-c)) + rho0*std::pow(c, alpha);
          P = std::tanh(2*beta*std::pow(c, 2*beta-1)*(rad-c)) + P0*std::pow(c, 2*beta);
          }

        phydro->u(IDN,k,j,i) = rho;

        if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
          phydro->u(IM1,k,j,i) = rho*vr;
          phydro->u(IM2,k,j,i) = 0.0;
        } else { // cartesian
          phydro->u(IM1,k,j,i) = rho*vr*x1/rad;
          phydro->u(IM2,k,j,i) = rho*vr*x2/rad;
        }

        phydro->u(IM3,k,j,i) = 0.0;
        phydro->u(IEN,k,j,i) = P/gm1 + 0.5*rho*SQR(vr);
      }
    }
  }

  // initialize face-averaged magnetic fields
  if (MAGNETIC_FIELDS_ENABLED) {
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie+1; i++) {
          Real geom_coeff = 1.0;
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            geom_coeff = 1.0/pcoord->x1f(i);
          }
          pfield->b.x1f(k,j,i) = geom_coeff*(az(j+1,i) - az(j,i))/pcoord->dx2f(j);
        }
      }
    }
    for (int k=ks; k<=ke; k++) {
      for (int j=js; j<=je+1; j++) {
        for (int i=is; i<=ie; i++) {
          Real geom_coeff = 1.0;
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            geom_coeff = -1.0; // Left hand system?
          }
          pfield->b.x2f(k,j,i) = geom_coeff*(az(j,i) - az(j,i+1))/pcoord->dx1f(i);
        }
      }
    }
    for (int k=ks; k<=ke+1; k++) {
      for (int j=js; j<=je; j++) {
        for (int i=is; i<=ie; i++) {
          Real rad;
          if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
            rad = pcoord->x1v(i);
          } else {
            rad = std::sqrt(SQR(pcoord->x1v(i)) + SQR(pcoord->x2v(j)));
          }
          // az is smoothed, but bz is constant and uniform:
          Real x1 = pcoord->x1v(i);
          Real x2 = pcoord->x2v(j);
          rad = std::sqrt(SQR(x1) + SQR(x2));

          // Calculating smoothing parameters for rho and P
          Real gr = (std::tanh(10 * (1.0 - rad)) + 1.0) / 2.0;
          Real g1 = (std::tanh(10 * (rad - 1.0)) + 1.0) / 2.0;

          //Field values at r=r and r=1
          Real bzr = bz*std::pow(rad,beta);
          Real bz1 = bz*std::pow(1.0,beta);
          
          // Populating a smoothed bz
          pfield->b.x3f(k,j,i) = gr * bzr + g1 * bz1;
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
