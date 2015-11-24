//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
// distribution.  If not see <http://www.gnu.org/licenses/>.
//======================================================================================
//! \file mesh.cpp
//  \brief implementation of functions in classes Mesh, and MeshBlock
//======================================================================================

// C/C++ headers
#include <cfloat>     // FLT_MAX
#include <cmath>      // std::abs(), pow()
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()
#include <algorithm>  // sort
#include <iomanip>
#include <stdlib.h>
#include <string.h>  // memcpy

// Athena++ classes headers
#include "athena.hpp"                   // enums, macros, Real
#include "globals.hpp"
#include "athena_arrays.hpp"            // AthenaArray
#include "coordinates/coordinates.hpp"  // Coordinates
#include "hydro/hydro.hpp" 
#include "field/field.hpp"              // Field
#include "bvals/bvals.hpp"              // BoundaryValues
#include "hydro/eos/eos.hpp"
#include "hydro/integrators/hydro_integrator.hpp" 
#include "field/integrators/field_integrator.hpp"  // FieldIntegrator
#include "parameter_input.hpp"          // ParameterInput
#include "meshblocktree.hpp"
#include "outputs/wrapper.hpp"
#include "task_list.hpp"
#include "mesh_refinement/mesh_refinement.hpp"

// this class header
#include "mesh.hpp"

// MPI/OpenMP header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

//--------------------------------------------------------------------------------------
// Mesh constructor, builds mesh at start of calculation using parameters in input file

Mesh::Mesh(ParameterInput *pin, int test_flag)
{
  std::stringstream msg;
  RegionSize block_size;
  MeshBlockTree *neibt;
  MeshBlock *pfirst;
  int block_bcs[6];
  int nbmax, dim;

// mesh test
  if(test_flag>0) Globals::nranks=test_flag;

// read time and cycle limits from input file

  start_time = pin->GetOrAddReal("time","start_time",0.0);
  tlim       = pin->GetReal("time","tlim");
  cfl_number = pin->GetReal("time","cfl_number");
  time = start_time;
  dt   = (FLT_MAX*0.4);

  nlim = pin->GetOrAddInteger("time","nlim",-1);
  ncycle = 0;

// read number of OpenMP threads for mesh

  num_mesh_threads_ = pin->GetOrAddInteger("mesh","num_threads",1);
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" 
        << num_mesh_threads_ << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// read number of grid cells in root level of mesh from input file.  

  mesh_size.nx1 = pin->GetInteger("mesh","nx1");
  if (mesh_size.nx1 < 4) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx1 must be >= 4, but nx1=" 
        << mesh_size.nx1 << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  mesh_size.nx2 = pin->GetInteger("mesh","nx2");
  if (mesh_size.nx2 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx2 must be >= 1, but nx2=" 
        << mesh_size.nx2 << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  mesh_size.nx3 = pin->GetInteger("mesh","nx3");
  if (mesh_size.nx3 < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file nx3 must be >= 1, but nx3=" 
        << mesh_size.nx3 << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (mesh_size.nx2 == 1 && mesh_size.nx3 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "In mesh block in input file: nx2=1, nx3=" << mesh_size.nx3 
        << ", 2D problems in x1-x3 plane not supported" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// check cfl_number
  if(cfl_number > 1.0 && mesh_size.nx2==1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The CFL number must be smaller than 1.0 in 1D simulation" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if(cfl_number > 0.5 && mesh_size.nx2 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The CFL number must be smaller than 0.5 in 2D/3D simulation" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  dim=1;
  if(mesh_size.nx2>1) dim=2;
  if(mesh_size.nx3>1) dim=3;

// read physical size of mesh (root level) from input file.  

  mesh_size.x1min = pin->GetReal("mesh","x1min");
  mesh_size.x2min = pin->GetReal("mesh","x2min");
  mesh_size.x3min = pin->GetReal("mesh","x3min");

  mesh_size.x1max = pin->GetReal("mesh","x1max");
  mesh_size.x2max = pin->GetReal("mesh","x2max");
  mesh_size.x3max = pin->GetReal("mesh","x3max");

  if (mesh_size.x1max <= mesh_size.x1min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x1max must be larger than x1min: x1min=" << mesh_size.x1min 
        << " x1max=" << mesh_size.x1max << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (mesh_size.x2max <= mesh_size.x2min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x2max must be larger than x2min: x2min=" << mesh_size.x2min 
        << " x2max=" << mesh_size.x2max << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (mesh_size.x3max <= mesh_size.x3min) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Input x3max must be larger than x3min: x3min=" << mesh_size.x3min 
        << " x3max=" << mesh_size.x3max << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// read ratios of grid cell size in each direction

  block_size.x1rat = mesh_size.x1rat = pin->GetOrAddReal("mesh","x1rat",1.0);
  block_size.x2rat = mesh_size.x2rat = pin->GetOrAddReal("mesh","x2rat",1.0);
  block_size.x3rat = mesh_size.x3rat = pin->GetOrAddReal("mesh","x3rat",1.0);

  if (std::abs(mesh_size.x1rat - 1.0) > 0.1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Ratio of cell sizes must be 0.9 <= x1rat <= 1.1, x1rat=" 
        << mesh_size.x1rat << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (std::abs(mesh_size.x2rat - 1.0) > 0.1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Ratio of cell sizes must be 0.9 <= x2rat <= 1.1, x2rat=" 
        << mesh_size.x2rat << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if (std::abs(mesh_size.x3rat - 1.0) > 0.1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Ratio of cell sizes must be 0.9 <= x3rat <= 1.1, x3rat=" 
        << mesh_size.x3rat << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// read BC flags for each of the 6 boundaries in turn.  Error tests performed in
// BoundaryValues constructor

  mesh_bcs[inner_x1] = pin->GetOrAddInteger("mesh","ix1_bc",0);
  mesh_bcs[outer_x1] = pin->GetOrAddInteger("mesh","ox1_bc",0);
  mesh_bcs[inner_x2] = pin->GetOrAddInteger("mesh","ix2_bc",0);
  mesh_bcs[outer_x2] = pin->GetOrAddInteger("mesh","ox2_bc",0);
  mesh_bcs[inner_x3] = pin->GetOrAddInteger("mesh","ix3_bc",0);
  mesh_bcs[outer_x3] = pin->GetOrAddInteger("mesh","ox3_bc",0);

// read MeshBlock parameters
  block_size.nx1 = pin->GetOrAddInteger("meshblock","nx1",mesh_size.nx1);
  if(dim>=2)
    block_size.nx2 = pin->GetOrAddInteger("meshblock","nx2",mesh_size.nx2);
  else
    block_size.nx2=mesh_size.nx2;
  if(dim==3)
    block_size.nx3 = pin->GetOrAddInteger("meshblock","nx3",mesh_size.nx3);
  else
    block_size.nx3=mesh_size.nx3;

// check consistency of the block and mesh
  if(mesh_size.nx1%block_size.nx1 != 0
  || mesh_size.nx2%block_size.nx2 != 0
  || mesh_size.nx3%block_size.nx3 != 0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "the mesh must be evenly divisible by the meshblock" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if(block_size.nx1 <4 || (block_size.nx2<4 && dim>=2)
     || (block_size.nx3<4 && dim==3)) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "block_size must be larger than or equal to 4 meshes." << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

// calculate the number of the blocks
  nrbx1=mesh_size.nx1/block_size.nx1;
  nrbx2=mesh_size.nx2/block_size.nx2;
  nrbx3=mesh_size.nx3/block_size.nx3;
  nbmax=(nrbx1>nrbx2)?nrbx1:nrbx2;
  nbmax=(nbmax>nrbx3)?nbmax:nrbx3;

  if(Globals::my_rank==0)
    std::cout << "RootGrid = " << nrbx1 << " x " << nrbx2
              << " x " << nrbx3 << std::endl;

// calculate the logical root level and maximum level
  for(root_level=0;(1<<root_level)<nbmax;root_level++);
  current_level=root_level;

// create the root grid
  tree.CreateRootGrid(nrbx1,nrbx2,nrbx3,root_level);

// SMR / AMR: create finer grids here
  multilevel=false;
  adaptive=false;
  if(pin->GetOrAddString("mesh","refinement","static")=="adaptive")
    adaptive=true, multilevel=true;
  if(adaptive==true) {
    max_level = pin->GetOrAddInteger("mesh","maxlevel",1)+root_level-1;
    if(max_level > 63) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "The maximum refinement level must be smaller than "
          << 63-root_level+1 << "." << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }
  else
    max_level = 63;

  InputBlock *pib = pin->pfirst_block;
  while (pib != NULL) {
    if (pib->block_name.compare(0,10,"refinement") == 0) {
      RegionSize ref_size;
      ref_size.x1min=pin->GetReal(pib->block_name,"x1min");
      ref_size.x1max=pin->GetReal(pib->block_name,"x1max");
      if(dim>=2) {
        ref_size.x2min=pin->GetReal(pib->block_name,"x2min");
        ref_size.x2max=pin->GetReal(pib->block_name,"x2max");
      }
      else {
        ref_size.x2min=mesh_size.x2min;
        ref_size.x2max=mesh_size.x2max;
      }
      if(dim>=3) {
        ref_size.x3min=pin->GetReal(pib->block_name,"x3min");
        ref_size.x3max=pin->GetReal(pib->block_name,"x3max");
      }
      else {
        ref_size.x3min=mesh_size.x3min;
        ref_size.x3max=mesh_size.x3max;
      }
      int ref_lev=pin->GetReal(pib->block_name,"level");
      int lrlev=ref_lev+root_level;
      if(lrlev>current_level) current_level=lrlev;
      if(lrlev!=root_level)
        multilevel=true;
      // range check
      if(ref_lev<1) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "Refinement level must be larger than 0 (root level = 0)" << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if(lrlev > max_level) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "Refinement level exceeds the maximum level (specify maxlevel in <mesh> if adaptive)."
            << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if(ref_size.x1min > ref_size.x1max || ref_size.x2min > ref_size.x2max
      || ref_size.x3min > ref_size.x3max)  {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "Invalid refinement region is specified."<<  std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      if(ref_size.x1min < mesh_size.x1min || ref_size.x1max > mesh_size.x1max
      || ref_size.x2min < mesh_size.x2min || ref_size.x2max > mesh_size.x2max
      || ref_size.x3min < mesh_size.x3min || ref_size.x3max > mesh_size.x3max) {
        msg << "### FATAL ERROR in Mesh constructor" << std::endl
            << "Refinement region must be smaller than the whole mesh." << std::endl;
        throw std::runtime_error(msg.str().c_str());
      }
      // find the logical range in the ref_level
      // note: if this is too slow, this should be replaced with bi-section search.
      long int lx1min=0, lx1max=0, lx2min=0, lx2max=0, lx3min=0, lx3max=0;
      long int lxmax=nrbx1*(1L<<ref_lev);
      for(lx1min=0;lx1min<lxmax;lx1min++) {
        if(MeshGeneratorX1((Real)(lx1min+1)/lxmax,mesh_size)>ref_size.x1min)
          break;
      }
      for(lx1max=lx1min;lx1max<lxmax;lx1max++) {
        if(MeshGeneratorX1((Real)(lx1max+1)/lxmax,mesh_size)>=ref_size.x1max)
          break;
      }
      if(lx1min%2==1) lx1min--;
      if(lx1max%2==0) lx1max++;
      if(dim>=2) { // 2D or 3D
        lxmax=nrbx2*(1L<<ref_lev);
        for(lx2min=0;lx2min<lxmax;lx2min++) {
          if(MeshGeneratorX2((Real)(lx2min+1)/lxmax,mesh_size)>ref_size.x2min)
            break;
        }
        for(lx2max=lx2min;lx2max<lxmax;lx2max++) {
          if(MeshGeneratorX2((Real)(lx2max+1)/lxmax,mesh_size)>=ref_size.x2max)
            break;
        }
        if(lx2min%2==1) lx2min--;
        if(lx2max%2==0) lx2max++;
      }
      if(dim==3) { // 3D
        lxmax=nrbx3*(1L<<ref_lev);
        for(lx3min=0;lx3min<lxmax;lx3min++) {
          if(MeshGeneratorX3((Real)(lx3min+1)/lxmax,mesh_size)>ref_size.x3min)
            break;
        }
        for(lx3max=lx3min;lx3max<lxmax;lx3max++) {
          if(MeshGeneratorX3((Real)(lx3max+1)/lxmax,mesh_size)>=ref_size.x3max)
            break;
        }
        if(lx3min%2==1) lx3min--;
        if(lx3max%2==0) lx3max++;
      }
      // create the finest level
      std::cout << "refinenment: logical level = " << lrlev << ", lx1min = "
                << lx1min << ", lx1max = " << lx1max << ", lx2min = " << lx2min
                << ", lx2max = " << lx2max << ", lx3min = " << lx3min << ", lx3max = "
                << lx3max << std::endl;
      if(dim==1) {
        for(long int i=lx1min; i<lx1max; i+=2) {
          LogicalLocation nloc;
          nloc.level=lrlev, nloc.lx1=i, nloc.lx2=0, nloc.lx3=0;
          tree.AddMeshBlock(tree,nloc,dim,mesh_bcs,nrbx1,nrbx2,nrbx3,root_level);
        }
      }
      if(dim==2) {
        for(long int j=lx2min; j<lx2max; j+=2) {
          for(long int i=lx1min; i<lx1max; i+=2) {
            LogicalLocation nloc;
            nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=0;
            tree.AddMeshBlock(tree,nloc,dim,mesh_bcs,nrbx1,nrbx2,nrbx3,root_level);
          }
        }
      }
      if(dim==3) {
        for(long int k=lx3min; k<lx3max; k+=2) {
          for(long int j=lx2min; j<lx2max; j+=2) {
            for(long int i=lx1min; i<lx1max; i+=2) {
              LogicalLocation nloc;
              nloc.level=lrlev, nloc.lx1=i, nloc.lx2=j, nloc.lx3=k;
              tree.AddMeshBlock(tree,nloc,dim,mesh_bcs,nrbx1,nrbx2,nrbx3,root_level);
            }
          }
        }
      }
    }
    pib=pib->pnext;
  }

  if(multilevel==true) {
    if(block_size.nx1%2==1 || (block_size.nx2%2==1 && block_size.nx2>1)
                           || (block_size.nx3%2==1 && block_size.nx3>1)) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
      << "The size of MeshBlock must be divisible by 2 in order to use SMR or AMR."
      << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
  }

  face_only=true;
  if (MAGNETIC_FIELDS_ENABLED || multilevel==true || VISCOSITY)
    face_only=false;

  maxneighbor_=BufferID(dim, multilevel, face_only);

  // initial mesh hierarchy construction is completed here

  tree.CountMeshBlock(nbtotal);
  loclist=new LogicalLocation[nbtotal];
  tree.GetMeshBlockList(loclist,NULL,nbtotal);

// check if there are sufficient blocks
#ifdef MPI_PARALLEL
  if(nbtotal < Globals::nranks) {
    if(test_flag==0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few blocks: nbtotal (" << nbtotal << ") < nranks ("<< Globals::nranks
          << ")" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
          << "Too few blocks: nbtotal (" << nbtotal << ") < nranks ("<< Globals::nranks
          << ")" << std::endl;
    }
  }
#endif

  ranklist=new int[nbtotal];
  nslist=new int[Globals::nranks];
  nblist=new int[Globals::nranks];
  costlist=new Real[nbtotal];

  for(int i=0;i<nbtotal;i++)
    costlist[i]=1.0; // the simplest estimate; all the blocks are equal

  LoadBalancing(costlist, ranklist, nslist, nblist);

  // store my nbstart and nbend
  nbstart=nslist[Globals::my_rank];
  if((Globals::my_rank)+1 == Globals::nranks)
    nbend=nbtotal-1;
  else 
    nbend=nslist[(Globals::my_rank)+1]-1;

  // Mesh test only; do not create meshes
  if(test_flag>0) {
    if(Globals::my_rank==0)
      MeshTest(dim);
    return;
  }

// create MeshBlock list for this process
  for(int i=nbstart;i<=nbend;i++) {
    long int &lx1=loclist[i].lx1;
    long int &lx2=loclist[i].lx2;
    long int &lx3=loclist[i].lx3;
    int &ll=loclist[i].level;
    // calculate physical block size, x1
    if(lx1==0) {
      block_size.x1min=mesh_size.x1min;
      block_bcs[inner_x1]=mesh_bcs[inner_x1];
    }
    else {
      Real rx=(Real)lx1/(Real)(nrbx1<<(ll-root_level));
      block_size.x1min=MeshGeneratorX1(rx,mesh_size);
      block_bcs[inner_x1]=-1;
    }
    if(lx1==(nrbx1<<(ll-root_level))-1) {
      block_size.x1max=mesh_size.x1max;
      block_bcs[outer_x1]=mesh_bcs[outer_x1];
    }
    else {
      Real rx=(Real)(lx1+1)/(Real)(nrbx1<<(ll-root_level));
      block_size.x1max=MeshGeneratorX1(rx,mesh_size);
      block_bcs[outer_x1]=-1;
    }

    // calculate physical block size, x2
    if(dim==1) {
      block_size.x2min=mesh_size.x2min;
      block_size.x2max=mesh_size.x2max;
      block_bcs[inner_x2]=mesh_bcs[inner_x2];
      block_bcs[outer_x2]=mesh_bcs[outer_x2];
    }
    else {
      if(lx2==0) {
        block_size.x2min=mesh_size.x2min;
        block_bcs[inner_x2]=mesh_bcs[inner_x2];
      }
      else {
        Real rx=(Real)lx2/(Real)(nrbx2<<(ll-root_level));
        block_size.x2min=MeshGeneratorX2(rx,mesh_size);
        block_bcs[inner_x2]=-1;
      }
      if(lx2==(nrbx2<<(ll-root_level))-1) {
        block_size.x2max=mesh_size.x2max;
        block_bcs[outer_x2]=mesh_bcs[outer_x2];
      }
      else {
        Real rx=(Real)(lx2+1)/(Real)(nrbx2<<(ll-root_level));
        block_size.x2max=MeshGeneratorX2(rx,mesh_size);
        block_bcs[outer_x2]=-1;
      }
    }

    // calculate physical block size, x3
    if(dim<=2) {
      block_size.x3min=mesh_size.x3min;
      block_size.x3max=mesh_size.x3max;
      block_bcs[inner_x3]=mesh_bcs[inner_x3];
      block_bcs[outer_x3]=mesh_bcs[outer_x3];
    }
    else {
      if(lx3==0) {
        block_size.x3min=mesh_size.x3min;
        block_bcs[inner_x3]=mesh_bcs[inner_x3];
      }
      else {
        Real rx=(Real)lx3/(Real)(nrbx3<<(ll-root_level));
        block_size.x3min=MeshGeneratorX3(rx,mesh_size);
        block_bcs[inner_x3]=-1;
      }
      if(lx3==(nrbx3<<(ll-root_level))-1) {
        block_size.x3max=mesh_size.x3max;
        block_bcs[outer_x3]=mesh_bcs[outer_x3];
      }
      else {
        Real rx=(Real)(lx3+1)/(Real)(nrbx3<<(ll-root_level));
        block_size.x3max=MeshGeneratorX3(rx,mesh_size);
        block_bcs[outer_x3]=-1;
      }
    }

    // create a block and add into the link list
    if(i==nbstart) {
      pblock = new MeshBlock(i, i-nbstart, loclist[i], block_size, block_bcs, this, pin);
      pfirst = pblock;
    }
    else {
      pblock->next = new MeshBlock(i, i-nbstart, loclist[i], block_size, block_bcs, this, pin);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }

    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock=pfirst;

// create new Task List, requires mesh to already be constructed
  ptlist = new TaskList(this);

}


//--------------------------------------------------------------------------------------
// Mesh constructor for restarting. Load the restarting file

Mesh::Mesh(ParameterInput *pin, IOWrapper& resfile, int test_flag)
{
  std::stringstream msg;
  RegionSize block_size;
  MeshBlock *pfirst;
  int i, j, nerr, dim;
  IOWrapperSize_t *offset;
  Real totalcost, targetcost, maxcost, mincost, mycost;

// mesh test
  if(test_flag>0) Globals::nranks=test_flag;

// read time and cycle limits from input file

  start_time = pin->GetOrAddReal("time","start_time",0.0);
  tlim       = pin->GetReal("time","tlim");
  cfl_number = pin->GetReal("time","cfl_number");
  nlim = pin->GetOrAddInteger("time","nlim",-1);

// read number of OpenMP threads for mesh
  num_mesh_threads_ = pin->GetOrAddInteger("mesh","num_threads",1);
  if (num_mesh_threads_ < 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Number of OpenMP threads must be >= 1, but num_threads=" 
        << num_mesh_threads_ << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  // read from the restarting file (everyone)
  // the file is already open and the pointer is set to after <par_end>
  nerr=0;
  if(resfile.Read(&nbtotal, sizeof(int), 1)!=1) nerr++;
  if(resfile.Read(&root_level, sizeof(int), 1)!=1) nerr++;
  current_level=root_level;
  if(resfile.Read(&mesh_size, sizeof(RegionSize), 1)!=1) nerr++;
  if(resfile.Read(mesh_bcs, sizeof(int), 6)!=6) nerr++;
  if(resfile.Read(&time, sizeof(Real), 1)!=1) nerr++;
  if(resfile.Read(&dt, sizeof(Real), 1)!=1) nerr++;
  if(resfile.Read(&ncycle, sizeof(int), 1)!=1) nerr++;
  if(nerr>0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restarting file is broken." << std::endl;
    resfile.Close();
    throw std::runtime_error(msg.str().c_str());
  }

  max_level = pin->GetOrAddInteger("mesh","maxlevel",1)+root_level-1;

  dim=1;
  if(mesh_size.nx2>1) dim=2;
  if(mesh_size.nx3>1) dim=3;

// check cfl_number
  if(cfl_number > 1.0 && mesh_size.nx2==1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The CFL number must be smaller than 1.0 in 1D simulation" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }
  if(cfl_number > 0.5 && mesh_size.nx2 > 1) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The CFL number must be smaller than 0.5 in 2D/3D simulation" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

  //initialize
  loclist=new LogicalLocation[nbtotal];
  offset=new IOWrapperSize_t[nbtotal];
  costlist=new Real[nbtotal];
  ranklist=new int[nbtotal];
  nslist=new int[Globals::nranks];
  nblist=new int[Globals::nranks];

  int nx1 = pin->GetOrAddReal("meshblock","nx1",mesh_size.nx1);
  int nx2 = pin->GetOrAddReal("meshblock","nx2",mesh_size.nx2);
  int nx3 = pin->GetOrAddReal("meshblock","nx3",mesh_size.nx3);

// calculate the number of the blocks
  nrbx1=mesh_size.nx1/nx1;
  nrbx2=mesh_size.nx2/nx2;
  nrbx3=mesh_size.nx3/nx3;

  // read the id list (serial, because we need the costs for load balancing)
  // ... perhaps I should pack them.
  multilevel=false;
  nerr=0;
  maxcost=0.0;
  mincost=(FLT_MAX);
  for(int i=0;i<nbtotal;i++) {
    int bgid;
    if(resfile.Read(&bgid,sizeof(int),1)!=1) nerr++;
    if(resfile.Read(&(loclist[i]),sizeof(LogicalLocation),1)!=1) nerr++;
    if(loclist[i].level!=root_level) multilevel=true;
    if(loclist[i].level>current_level) current_level=loclist[i].level;
    if(resfile.Read(&(costlist[i]),sizeof(Real),1)!=1) nerr++;
    if(resfile.Read(&(offset[i]),sizeof(IOWrapperSize_t),1)!=1) nerr++;
    mincost=std::min(mincost,costlist[i]);
    maxcost=std::max(maxcost,costlist[i]);
  }
  if(nerr>0) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "The restarting file is broken." << std::endl;
    resfile.Close();
    throw std::runtime_error(msg.str().c_str());
  }

  adaptive=false;
  if(pin->GetOrAddString("mesh","refinement","static")=="adaptive")
    adaptive=true, multilevel=true;

  face_only=true;
  if (MAGNETIC_FIELDS_ENABLED || multilevel==true || VISCOSITY)
    face_only=false;

  maxneighbor_=BufferID(dim, multilevel, face_only);

  // rebuild the Block Tree
  for(int i=0;i<nbtotal;i++)
    tree.AddMeshBlockWithoutRefine(loclist[i],nrbx1,nrbx2,nrbx3,root_level);
  int nnb;
  // check the tree structure, and assign GID
  tree.GetMeshBlockList(loclist, NULL, nnb);
  if(nnb!=nbtotal) {
    msg << "### FATAL ERROR in Mesh constructor" << std::endl
        << "Tree reconstruction failed. The total numbers of the blocks do not match. ("
        << nbtotal << " != " << nnb << ")" << std::endl;
    throw std::runtime_error(msg.str().c_str());
  }

#ifdef MPI_PARALLEL
  if(nbtotal < Globals::nranks) {
    if(test_flag==0) {
      msg << "### FATAL ERROR in Mesh constructor" << std::endl
          << "Too few blocks: nbtotal (" << nbtotal << ") < nranks ("<< Globals::nranks
          << ")" << std::endl;
      throw std::runtime_error(msg.str().c_str());
    }
    else { // test
      std::cout << "### Warning in Mesh constructor" << std::endl
          << "Too few blocks: nbtotal (" << nbtotal << ") < nranks ("<< Globals::nranks
          << ")" << std::endl;
      return;
    }
  }
#endif

  LoadBalancing(costlist, ranklist, nslist, nblist);

  // store my nbstart and nbend
  nbstart=nslist[Globals::my_rank];
  if((Globals::my_rank)+1==Globals::nranks)
    nbend=nbtotal-1;
  else 
    nbend=nslist[(Globals::my_rank)+1]-1;

  // Mesh test only; do not create meshes
  if(test_flag>0) {
    if(Globals::my_rank==0)
      MeshTest(dim);
    delete [] offset;
    return;
  }

  // load MeshBlocks (parallel)
  for(i=nbstart;i<=nbend;i++) {
    // create a block and add into the link list
    if(i==nbstart) {
      pblock = new MeshBlock(i, i-nbstart, this, pin, loclist[i], resfile, offset[i],
                             costlist[i], ranklist, nslist);
      pfirst = pblock;
    }
    else {
      pblock->next = new MeshBlock(i, i-nbstart, this, pin, loclist[i], resfile,
                                   offset[i], costlist[i], ranklist, nslist);
      pblock->next->prev = pblock;
      pblock = pblock->next;
    }
    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
  }
  pblock=pfirst;

// create new Task List
  ptlist = new TaskList(this);

// clean up
  delete [] offset;
}


// destructor

Mesh::~Mesh()
{
  while(pblock->prev != NULL) // should not be true
    delete pblock->prev;
  while(pblock->next != NULL)
    delete pblock->next;
  delete pblock;
  delete ptlist;
  delete [] nslist;
  delete [] nblist;
  delete [] ranklist;
  delete [] costlist;
  delete [] loclist;
}


//--------------------------------------------------------------------------------------
//! \fn void Mesh::MeshTest(int dim)
//  \brief print the mesh structure information

void Mesh::MeshTest(int dim)
{
  int i, j, nbt=0;
  long int lx1, lx2, lx3;
  int ll;
  Real mycost=0, mincost=FLT_MAX, maxcost=0.0, totalcost=0.0;
  int *nb=new int [max_level-root_level+1];
  FILE *fp;
  if(dim>=2) {
    if ((fp = fopen("meshtest.dat","wb")) == NULL) {
      std::cout << "### ERROR in function Mesh::MeshTest" << std::endl
                << "Cannot open meshtest.dat" << std::endl;
      return;
    }
  }

  std::cout << "Logical level of the physical root grid = "<< root_level << std::endl;
  std::cout << "Logical level of maximum refinement = "<< current_level << std::endl;
  std::cout << "List of MeshBlocks" << std::endl;
  for(i=root_level;i<=max_level;i++) {
    Real dx=1.0/(Real)(1L<<i);
    nb[i-root_level]=0;
    for(j=0;j<nbtotal;j++) {
      if(loclist[j].level==i) {
        long int &lx1=loclist[j].lx1;
        long int &lx2=loclist[j].lx2;
        long int &lx3=loclist[j].lx3;
        int &ll=loclist[j].level;
        std::cout << "MeshBlock " << j << ", lx1 = "
                  << loclist[j].lx1 << ", lx2 = " << lx2 <<", lx3 = " << lx3
                  << ", logical level = " << ll << ", physical level = "
                  << ll-root_level << ", cost = " << costlist[j]
                  << ", rank = " << ranklist[j] << std::endl;
        mincost=std::min(mincost,costlist[i]);
        maxcost=std::max(maxcost,costlist[i]);
        totalcost+=costlist[i];
        nb[i-root_level]++;
        if(dim==2) {
          fprintf(fp, "#MeshBlock %d at %ld %ld %ld %d\n", j, lx1, lx2, lx3, ll);
          fprintf(fp, "%g %g %d %d\n", lx1*dx, lx2*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %d %d\n", lx1*dx+dx, lx2*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %d %d\n", lx1*dx+dx, lx2*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %d %d\n", lx1*dx, lx2*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %d %d\n\n\n", lx1*dx, lx2*dx, ll, ranklist[j]);
        }
        if(dim==3) {
          fprintf(fp, "#MeshBlock %d at %ld %ld %ld %d\n", j, lx1, lx2, lx3, ll);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx+dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx+dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx+dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx+dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx+dx, lx2*dx+dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx+dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx+dx, lx3*dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx+dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n", lx1*dx, lx2*dx, lx3*dx+dx, ll, ranklist[j]);
          fprintf(fp, "%g %g %g %d %d\n\n\n", lx1*dx, lx2*dx, lx3*dx, ll, ranklist[j]);
        }
      }
    }
  }
  if(dim>=2) fclose(fp);

  std::cout << std::endl;

  for(i=root_level;i<=max_level;i++) {
    if(nb[i-root_level]!=0)
      std::cout << "Level " << i-root_level << " (logical level " << i << ") : "
        << nb[i-root_level] << " MeshBlocks" << std::endl;
  }

  std::cout << "Total : " << nbtotal << " MeshBlocks" << std::endl << std::endl;
  std::cout << "Load Balance :" << std::endl;
  std::cout << "Minimum cost = " << mincost << ", Maximum cost = " << maxcost
            << ", Average cost = " << totalcost/nbtotal << std::endl;
  j=0;
  nbt=0;
  for(i=0;i<nbtotal;i++) {
    if(ranklist[i]==j) {
      mycost+=costlist[i];
      nbt++;
    }
    else if(ranklist[i]!=j) {
      std::cout << "Rank " << j << ": " << nbt <<" MeshBlocks, cost = " << mycost << std::endl;
      mycost=costlist[i];
      nbt=1;
      j++;
    }
  }
  std::cout << "Rank " << j << ": " << nbt <<" MeshBlocks, cost = " << mycost << std::endl;

  delete [] nb;
  return;
}

//--------------------------------------------------------------------------------------
// MeshBlock constructor: builds 1D vectors of cell positions and spacings, and
// constructs coordinate, boundary condition, hydro and field objects.

MeshBlock::MeshBlock(int igid, int ilid, LogicalLocation iloc, RegionSize input_block,
                     int *input_bcs, Mesh *pm, ParameterInput *pin)
{
  std::stringstream msg;
  int root_level;
  pmy_mesh = pm;
  root_level = pm->root_level;
  block_size = input_block;
  for(int i=0; i<6; i++) block_bcs[i] = input_bcs[i];
  prev=NULL;
  next=NULL;
  gid=igid;
  lid=ilid;
  loc=iloc;
  cost=1.0;

// initialize grid indices

  is = NGHOST;
  ie = is + block_size.nx1 - 1;

  if (block_size.nx2 > 1) {
    js = NGHOST;
    je = js + block_size.nx2 - 1;
  } else {
    js = je = 0;
  }

  if (block_size.nx3 > 1) {
    ks = NGHOST;
    ke = ks + block_size.nx3 - 1;
  } else {
    ks = ke = 0;
  }

  if(pm->multilevel==true) {
    cnghost=(NGHOST+1)/2+1;
    cis=cnghost; cie=cis+block_size.nx1/2-1;
    cjs=cje=cks=cke=0;
    if(block_size.nx2>1) // 2D or 3D
      cjs=cnghost, cje=cjs+block_size.nx2/2-1;
    if(block_size.nx3>1) // 3D
      cks=cnghost, cke=cks+block_size.nx3/2-1;
  }

  std::cout << "MeshBlock " << gid << ", rank = " << Globals::my_rank << ", lx1 = "
            << loc.lx1 << ", lx2 = " << loc.lx2 <<", lx3 = " << loc.lx3
            << ", level = " << loc.level << std::endl;
  std::cout << "is=" << is << " ie=" << ie << " x1min=" << block_size.x1min
            << " x1max=" << block_size.x1max << std::endl;
  std::cout << "js=" << js << " je=" << je << " x2min=" << block_size.x2min
            << " x2max=" << block_size.x2max << std::endl;
  std::cout << "ks=" << ks << " ke=" << ke << " x3min=" << block_size.x3min
            << " x3max=" << block_size.x3max << std::endl;

// construct Coordinates and Hydro objects stored in MeshBlock class.  Note that the
// initial conditions for the hydro are set in problem generator called from main, not
// in the Hydro constructor
 
  pcoord = new Coordinates(this, pin);
  if(pm->multilevel==true) {
    pcoarsec = new Coordinates(this, pin, 1);
    pmr = new MeshRefinement(this, pin);
  }
  phydro = new Hydro(this, pin);
  pfield = new Field(this, pin);
  pbval  = new BoundaryValues(this, pin);

  return;
}

//--------------------------------------------------------------------------------------
// MeshBlock constructor for restarting

MeshBlock::MeshBlock(int igid, int ilid, Mesh *pm, ParameterInput *pin,
                     LogicalLocation iloc, IOWrapper& resfile, IOWrapperSize_t offset,
                     Real icost, int *ranklist, int *nslist)
{
  std::stringstream msg;
  pmy_mesh = pm;
  prev=NULL;
  next=NULL;
  gid=igid;
  lid=ilid;
  loc=iloc;
  cost=icost;
  int nerr=0;
//  task=NULL;

  // seek the file
  resfile.Seek(offset);
  // load block structure
  if(resfile.Read(&block_size, sizeof(RegionSize), 1)!=1) nerr++;
  if(resfile.Read(block_bcs, sizeof(int), 6)!=6) nerr++;

  if(nerr>0) {
    msg << "### FATAL ERROR in MeshBlock constructor" << std::endl
        << "The restarting file is broken." << std::endl;
    resfile.Close();
    throw std::runtime_error(msg.str().c_str());
  }

// initialize grid indices

  is = NGHOST;
  ie = is + block_size.nx1 - 1;

  if (block_size.nx2 > 1) {
    js = NGHOST;
    je = js + block_size.nx2 - 1;
  } else {
    js = je = 0;
  }

  if (block_size.nx3 > 1) {
    ks = NGHOST;
    ke = ks + block_size.nx3 - 1;
  } else {
    ks = ke = 0;
  }

  if(pm->multilevel==true) {
    cnghost=(NGHOST+1)/2+1;
    cis=cnghost; cie=cis+block_size.nx1/2-1;
    cjs=cje=cks=cke=0;
    if(block_size.nx2>1) // 2D or 3D
      cjs=cnghost, cje=cjs+block_size.nx2/2-1;
    if(block_size.nx3>1) // 3D
      cks=cnghost, cke=cks+block_size.nx3/2-1;
  }

  std::cout << "MeshBlock " << gid << ", rank = " << Globals::my_rank << ", lx1 = "
            << loc.lx1 << ", lx2 = " << loc.lx2 <<", lx3 = " << loc.lx3
            << ", level = " << loc.level << std::endl;
  std::cout << "is=" << is << " ie=" << ie << " x1min=" << block_size.x1min
            << " x1max=" << block_size.x1max << std::endl;
  std::cout << "js=" << js << " je=" << je << " x2min=" << block_size.x2min
            << " x2max=" << block_size.x2max << std::endl;
  std::cout << "ks=" << ks << " ke=" << ke << " x3min=" << block_size.x3min
            << " x3max=" << block_size.x3max << std::endl;

  // create coordinates, hydro, field, and boundary conditions
  pcoord = new Coordinates(this, pin);
  if(pm->multilevel==true) {
    pcoarsec = new Coordinates(this, pin, 1);
    pmr = new MeshRefinement(this, pin);
  }
  phydro = new Hydro(this, pin);
  pfield = new Field(this, pin);
  pbval  = new BoundaryValues(this, pin);

  // load hydro and field data
  nerr=0;
  if(resfile.Read(phydro->u.GetArrayPointer(),sizeof(Real),
                         phydro->u.GetSize())!=phydro->u.GetSize()) nerr++;
  if (GENERAL_RELATIVITY) {
    if(resfile.Read(phydro->w.GetArrayPointer(),sizeof(Real),
                           phydro->w.GetSize())!=phydro->w.GetSize()) nerr++;
    if(resfile.Read(phydro->w1.GetArrayPointer(),sizeof(Real),
                           phydro->w1.GetSize())!=phydro->w1.GetSize()) nerr++;
  }
  if (MAGNETIC_FIELDS_ENABLED) {
    if(resfile.Read(pfield->b.x1f.GetArrayPointer(),sizeof(Real),
               pfield->b.x1f.GetSize())!=pfield->b.x1f.GetSize()) nerr++;
    if(resfile.Read(pfield->b.x2f.GetArrayPointer(),sizeof(Real),
               pfield->b.x2f.GetSize())!=pfield->b.x2f.GetSize()) nerr++;
    if(resfile.Read(pfield->b.x3f.GetArrayPointer(),sizeof(Real),
               pfield->b.x3f.GetSize())!=pfield->b.x3f.GetSize()) nerr++;
  }
  if(nerr>0) {
    msg << "### FATAL ERROR in MeshBlock constructor" << std::endl
        << "The restarting file is broken." << std::endl;
    resfile.Close();
    throw std::runtime_error(msg.str().c_str());
  }
  return;
}

// destructor

MeshBlock::~MeshBlock()
{
  if(prev!=NULL) prev->next=next;
  if(next!=NULL) next->prev=prev;

  delete pcoord;
  delete phydro;
  delete pfield;
  delete pbval;
//  delete [] task;
}


//--------------------------------------------------------------------------------------
// \!fn void Mesh::NewTimeStep(void)
// \brief function that loops over all MeshBlocks and find new timestep
//        this assumes that phydro->NewBlockTimeStep is already called

void Mesh::NewTimeStep(void)
{
  MeshBlock *pmb = pblock;
  Real min_dt=pmb->new_block_dt;
  pmb=pmb->next;
  while (pmb != NULL)  {
    min_dt=std::min(min_dt,pmb->new_block_dt);
    pmb=pmb->next;
  }
#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,&min_dt,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
#endif
  // set it
  dt=std::min(min_dt*cfl_number,2.0*dt);
  if (time < tlim && tlim-time < dt)  // timestep would take us past desired endpoint
    dt = tlim-time;
  return;
}

//--------------------------------------------------------------------------------------
// \!fn void Mesh::Initialize(int res_flag, ParameterInput *pin)
// \brief  initialization before the main loop

void Mesh::Initialize(int res_flag, ParameterInput *pin)
{
  MeshBlock *pmb;
  Hydro *phydro;
  Field *pfield;
  BoundaryValues *pbval;

  if(res_flag==0) {
    pmb = pblock;
    while (pmb != NULL)  {
      phydro=pmb->phydro;
      pfield=pmb->pfield;
      pbval=pmb->pbval;
      ProblemGenerator(phydro,pfield,pin);
      pbval->CheckBoundary();
      pmb=pmb->next;
    }
  }

  pmb = pblock;
  while (pmb != NULL)  {
    pmb->pbval->Initialize();
    pmb->pbval->StartReceivingForInit();
    pmb=pmb->next;
  }

  pmb = pblock;
  while (pmb != NULL)  {
    phydro=pmb->phydro;
    pfield=pmb->pfield;
    pbval=pmb->pbval;
    pbval->SendHydroBoundaryBuffers(phydro->u,0);
    if (MAGNETIC_FIELDS_ENABLED)
      pbval->SendFieldBoundaryBuffers(pfield->b,0);
    pmb=pmb->next;
  }

  pmb = pblock;
  while (pmb != NULL)  {
    phydro=pmb->phydro;
    pfield=pmb->pfield;
    pbval=pmb->pbval;
    pbval->ReceiveHydroBoundaryBuffersWithWait(phydro->u ,0);
    if (MAGNETIC_FIELDS_ENABLED)
      pbval->ReceiveFieldBoundaryBuffersWithWait(pfield->b ,0);
    pmb->pbval->ClearBoundaryForInit();
    if(multilevel==true)
      pbval->ProlongateBoundaries(phydro->w, phydro->u, pfield->b, pfield->bcc);

    int is=pmb->is, ie=pmb->ie, js=pmb->js, je=pmb->je, ks=pmb->ks, ke=pmb->ke;
    if(pmb->nblevel[1][1][0]!=-1) is-=NGHOST;
    if(pmb->nblevel[1][1][2]!=-1) ie+=NGHOST;
    if(pmb->nblevel[1][0][1]!=-1) js-=NGHOST;
    if(pmb->nblevel[1][2][1]!=-1) je+=NGHOST;
    if(pmb->nblevel[0][1][1]!=-1) ks-=NGHOST;
    if(pmb->nblevel[2][1][1]!=-1) ke+=NGHOST;
    phydro->pf_eos->ConservedToPrimitive(phydro->u, phydro->w1, pfield->b, 
                                         phydro->w, pfield->bcc, pmb->pcoord,
                                         is, ie, js, je, ks, ke);
    pbval->ApplyPhysicalBoundaries(phydro->w, phydro->u, pfield->b, pfield->bcc);

    pmb=pmb->next;
  }

  if(res_flag==0 || res_flag==2) {
    pmb = pblock;
    while (pmb != NULL)  {
      pmb->phydro->NewBlockTimeStep(pmb);
      pmb=pmb->next;
    }
    NewTimeStep();
  }
  return;
}


//--------------------------------------------------------------------------------------
//! \fn int64_t Mesh::GetTotalCells(void)
//  \brief return the total number of cells for performance counting

int64_t Mesh::GetTotalCells(void)
{
  return (int64_t)nbtotal*pblock->block_size.nx1*pblock->block_size.nx2*pblock->block_size.nx3;
}

//--------------------------------------------------------------------------------------
//! \fn long int MeshBlock::GetBlockSizeInBytes(void)
//  \brief Calculate the block data size required for restarting.

size_t MeshBlock::GetBlockSizeInBytes(void)
{
  size_t size;

  size =sizeof(RegionSize)+sizeof(int)*6;
  size+=sizeof(Real)*phydro->u.GetSize();
  if (GENERAL_RELATIVITY) {
    size+=sizeof(Real)*phydro->w.GetSize();
    size+=sizeof(Real)*phydro->w1.GetSize();
  }
  if (MAGNETIC_FIELDS_ENABLED)
    size+=sizeof(Real)*(pfield->b.x1f.GetSize()+pfield->b.x2f.GetSize()
                       +pfield->b.x3f.GetSize());
  // please add the size counter here when new physics is introduced

  return size;
}

//--------------------------------------------------------------------------------------
//! \fn void Mesh::UpdateOneStep(void)
//  \brief process the task list and advance one time step

void Mesh::UpdateOneStep(void)
{
  MeshBlock *pmb = pblock;
  int nb=nbend-nbstart+1;

  // initialize
  while (pmb != NULL)  {
    pmb->first_task=0;
    pmb->num_tasks_todo=ptlist->ntasks;
    for(int i=0; i<4; ++i) pmb->finished_tasks[i]=0; // encodes which tasks are done
    pmb->pbval->StartReceivingAll();
    pmb=pmb->next;
  }

  // main loop
  while(nb>0) {
    pmb = pblock;
    while (pmb != NULL)  {
      if(ptlist->DoOneTask(pmb)==TL_COMPLETE) // task list completed
        nb--;
      pmb=pmb->next;
    }
  }

  pmb = pblock;
  while (pmb != NULL)  {
    pmb->pbval->ClearBoundaryAll();
    pmb=pmb->next;
  }
  return;
}

//--------------------------------------------------------------------------------------
//! \fn MeshBlock* Mesh::FindMeshBlock(int tgid)
//  \brief return the MeshBlock whose gid is tgid

MeshBlock* Mesh::FindMeshBlock(int tgid)
{
  MeshBlock *pbl=pblock;
  while(pbl!=NULL)
  {
    if(pbl->gid==tgid)
      break;
    pbl=pbl->next;
  }
  return pbl;
}

//--------------------------------------------------------------------------------------
// \!fn void NeighborBlock::SetNeighbor(int irank, int ilevel, int igid, int ilid,
//                          int iox1, int iox2, int iox3, enum neighbor_type itype,
//                          int ibid, int itargetid, int ifi1=0, int ifi2=0)
// \brief Set neighbor information

void NeighborBlock::SetNeighbor(int irank, int ilevel, int igid, int ilid,
  int iox1, int iox2, int iox3, enum neighbor_type itype, int ibid, int itargetid,
  int ifi1=0, int ifi2=0)
{
  rank=irank; level=ilevel; gid=igid; lid=ilid; ox1=iox1; ox2=iox2; ox3=iox3; type=itype;
  bufid=ibid; targetid=itargetid; fi1=ifi1; fi2=ifi2;
  if(type==neighbor_face) {
    if(ox1==-1)      fid=inner_x1;
    else if(ox1==1)  fid=outer_x1;
    else if(ox2==-1) fid=inner_x2;
    else if(ox2==1)  fid=outer_x2;
    else if(ox3==-1) fid=inner_x3;
    else if(ox3==1)  fid=outer_x3;
  }
  if(type==neighbor_edge) {
    if(ox3==0)      eid=(edgeid)(   ((ox1+1)>>1) | ((ox2+1)&2));
    else if(ox2==0) eid=(edgeid)(4+(((ox1+1)>>1) | ((ox3+1)&2)));
    else if(ox1==0) eid=(edgeid)(8+(((ox2+1)>>1) | ((ox3+1)&2)));
  }
  return;
}

//--------------------------------------------------------------------------------------
// \!fn void MeshBlock::SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist, int *nslist)
// \brief Search and set all the neighbor blocks

void MeshBlock::SearchAndSetNeighbors(MeshBlockTree &tree, int *ranklist, int *nslist)
{
  MeshBlockTree* neibt;
  int myox1, myox2=0, myox3=0, myfx1, myfx2, myfx3;
  myfx1=(int)(loc.lx1&1L);
  myfx2=(int)(loc.lx2&1L);
  myfx3=(int)(loc.lx3&1L);
  myox1=((int)(loc.lx1&1L))*2-1;
  if(block_size.nx2>1) myox2=((int)(loc.lx2&1L))*2-1;
  if(block_size.nx3>1) myox3=((int)(loc.lx3&1L))*2-1;
  long int nrbx1=pmy_mesh->nrbx1, nrbx2=pmy_mesh->nrbx2, nrbx3=pmy_mesh->nrbx3;

  int nf1=1, nf2=1;
  if(pmy_mesh->multilevel==true) {
    if(block_size.nx2>1) nf1=2;
    if(block_size.nx3>1) nf2=2;
  }
  int bufid=0;
  nneighbor=0;
  for(int k=0; k<=2; k++) {
    for(int j=0; j<=2; j++) {
      for(int i=0; i<=2; i++)
        nblevel[k][j][i]=-1;
    }
  }
  nblevel[1][1][1]=loc.level;

  // x1 face
  for(int n=-1; n<=1; n+=2) {
    neibt=tree.FindNeighbor(loc,n,0,0,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
    if(neibt==NULL) { bufid+=nf1*nf2; continue;}
    if(neibt->flag==false) { // finer
      int fface=1-(n+1)/2; // 0 for outer_x3, 1 for inner_x3
      nblevel[1][1][n+1]=neibt->loc.level+1;
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          MeshBlockTree* nf=neibt->GetLeaf(fface,f1,f2);
          int fid = nf->gid;
          int nlevel=nf->loc.level;
          int tbid=FindBufferID(-n,0,0,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
            fid-nslist[ranklist[fid]], n, 0, 0, neighbor_face, bufid, tbid, f1, f2);
          bufid++; nneighbor++;
        }
      }
    }
    else {
      int nlevel=neibt->loc.level;
      int nid=neibt->gid;
      nblevel[1][1][n+1]=nlevel;
      int tbid;
      if(nlevel==loc.level) tbid=FindBufferID(-n,0,0,0,0,pmy_mesh->maxneighbor_);
      else tbid=FindBufferID(-n,0,0,myfx2,myfx3,pmy_mesh->maxneighbor_);
      neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
        nid-nslist[ranklist[nid]], n, 0, 0, neighbor_face, bufid, tbid);
      bufid+=nf1*nf2; nneighbor++;
    }
  }
  if(block_size.nx2==1) return;
  // x2 face
  for(int n=-1; n<=1; n+=2) {
    neibt=tree.FindNeighbor(loc,0,n,0,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
    if(neibt==NULL) { bufid+=nf1*nf2; continue;}
    if(neibt->flag==false) { // finer
      int fface=1-(n+1)/2; // 0 for outer_x3, 1 for inner_x3
      nblevel[1][n+1][1]=neibt->loc.level+1;
      for(int f2=0;f2<nf2;f2++) {
        for(int f1=0;f1<nf1;f1++) {
          MeshBlockTree* nf=neibt->GetLeaf(f1,fface,f2);
          int fid = nf->gid;
          int nlevel=nf->loc.level;
          int tbid=FindBufferID(0,-n,0,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
            fid-nslist[ranklist[fid]], 0, n, 0, neighbor_face, bufid, tbid, f1, f2);
          bufid++; nneighbor++;
        }
      }
    }
    else {
      int nlevel=neibt->loc.level;
      int nid=neibt->gid;
      nblevel[1][n+1][1]=nlevel;
      int tbid;
      if(nlevel==loc.level) tbid=FindBufferID(0,-n,0,0,0,pmy_mesh->maxneighbor_);
      else tbid=FindBufferID(0,-n,0,myfx1,myfx3,pmy_mesh->maxneighbor_);
      neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
        nid-nslist[ranklist[nid]], 0, n, 0, neighbor_face, bufid, tbid);
      bufid+=nf1*nf2; nneighbor++;
    }
  }
  if(block_size.nx3>1) {
    // x3 face
    for(int n=-1; n<=1; n+=2) {
      neibt=tree.FindNeighbor(loc,0,0,n,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
      if(neibt==NULL) { bufid+=nf1*nf2; continue;}
      if(neibt->flag==false) { // finer
        int fface=1-(n+1)/2; // 0 for outer_x3, 1 for inner_x3
        nblevel[n+1][1][1]=neibt->loc.level+1;
        for(int f2=0;f2<nf2;f2++) {
          for(int f1=0;f1<nf1;f1++) {
            MeshBlockTree* nf=neibt->GetLeaf(f1,f2,fface);
            int fid = nf->gid;
            int nlevel=nf->loc.level;
            int tbid=FindBufferID(0,0,-n,0,0,pmy_mesh->maxneighbor_);
            neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
              fid-nslist[ranklist[fid]], 0, 0, n, neighbor_face, bufid, tbid, f1, f2);
            bufid++; nneighbor++;
          }
        }
      }
      else {
        int nlevel=neibt->loc.level;
        int nid=neibt->gid;
        nblevel[n+1][1][1]=nlevel;
        int tbid;
        if(nlevel==loc.level) tbid=FindBufferID(0,0,-n,0,0,pmy_mesh->maxneighbor_);
        else tbid=FindBufferID(0,0,-n,myfx1,myfx2,pmy_mesh->maxneighbor_);
        neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
          nid-nslist[ranklist[nid]], 0, 0, n, neighbor_face, bufid, tbid);
        bufid+=nf1*nf2; nneighbor++;
      }
    }
  }
  if(pmy_mesh->face_only==true) return;
  // edges
  // x1x2
  for(int m=-1; m<=1; m+=2) {
    for(int n=-1; n<=1; n+=2) {
      neibt=tree.FindNeighbor(loc,n,m,0,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
      if(neibt==NULL) { bufid+=nf2; continue;}
      if(neibt->flag==false) { // finer
        int ff1=1-(n+1)/2; // 0 for outer_x1, 1 for inner_x1
        int ff2=1-(m+1)/2; // 0 for outer_x2, 1 for inner_x2
        nblevel[1][m+1][n+1]=neibt->loc.level+1;
        for(int f1=0;f1<nf2;f1++) {
          MeshBlockTree* nf=neibt->GetLeaf(ff1,ff2,f1);
          int fid = nf->gid;
          int nlevel=nf->loc.level;
          int tbid=FindBufferID(-n,-m,0,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
            fid-nslist[ranklist[fid]], n, m, 0, neighbor_edge, bufid, tbid, f1, 0);
          bufid++; nneighbor++;
        }
      }
      else {
        int nlevel=neibt->loc.level;
        int nid=neibt->gid;
        nblevel[1][m+1][n+1]=nlevel;
        int tbid;
        if(nlevel==loc.level) tbid=FindBufferID(-n,-m,0,0,0,pmy_mesh->maxneighbor_);
        else tbid=FindBufferID(-n,-m,0,myfx3,0,pmy_mesh->maxneighbor_);
        if(nlevel>=loc.level || (myox1==n && myox2==m)) {
          neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
            nid-nslist[ranklist[nid]], n, m, 0, neighbor_edge, bufid, tbid);
          nneighbor++;
        }
        bufid+=nf2;
      }
    }
  }
  if(block_size.nx3==1) return;
  // x1x3
  for(int m=-1; m<=1; m+=2) {
    for(int n=-1; n<=1; n+=2) {
      neibt=tree.FindNeighbor(loc,n,0,m,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
      if(neibt==NULL) { bufid+=nf1; continue;}
      if(neibt->flag==false) { // finer
        int ff1=1-(n+1)/2; // 0 for outer_x1, 1 for inner_x1
        int ff2=1-(m+1)/2; // 0 for outer_x3, 1 for inner_x3
        nblevel[m+1][1][n+1]=neibt->loc.level+1;
        for(int f1=0;f1<nf1;f1++) {
          MeshBlockTree* nf=neibt->GetLeaf(ff1,f1,ff2);
          int fid = nf->gid;
          int nlevel=nf->loc.level;
          int tbid=FindBufferID(-n,0,-m,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
            fid-nslist[ranklist[fid]], n, 0, m, neighbor_edge, bufid, tbid, f1, 0);
          bufid++; nneighbor++;
        }
      }
      else {
        int nlevel=neibt->loc.level;
        int nid=neibt->gid;
        nblevel[m+1][1][n+1]=nlevel;
        int tbid;
        if(nlevel==loc.level) tbid=FindBufferID(-n,0,-m,0,0,pmy_mesh->maxneighbor_);
        else tbid=FindBufferID(-n,0,-m,myfx2,0,pmy_mesh->maxneighbor_);
        if(nlevel>=loc.level || (myox1==n && myox3==m)) {
          neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
            nid-nslist[ranklist[nid]], n, 0, m, neighbor_edge, bufid, tbid);
          nneighbor++;
        }
        bufid+=nf1;
      }
    }
  }
  // x2x3
  for(int m=-1; m<=1; m+=2) {
    for(int n=-1; n<=1; n+=2) {
      neibt=tree.FindNeighbor(loc,0,n,m,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
      if(neibt==NULL) { bufid+=nf1; continue;}
      if(neibt->flag==false) { // finer
        int ff1=1-(n+1)/2; // 0 for outer_x1, 1 for inner_x1
        int ff2=1-(m+1)/2; // 0 for outer_x3, 1 for inner_x3
        nblevel[m+1][n+1][1]=neibt->loc.level+1;
        for(int f1=0;f1<nf1;f1++) {
          MeshBlockTree* nf=neibt->GetLeaf(f1,ff1,ff2);
          int fid = nf->gid;
          int nlevel=nf->loc.level;
          int tbid=FindBufferID(0,-n,-m,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[fid], nlevel, fid,
            fid-nslist[ranklist[fid]], 0, n, m, neighbor_edge, bufid, tbid, f1, 0);
          bufid++; nneighbor++;
        }
      }
      else {
        int nlevel=neibt->loc.level;
        int nid=neibt->gid;
        nblevel[m+1][n+1][1]=nlevel;
        int tbid;
        if(nlevel==loc.level) tbid=FindBufferID(0,-n,-m,0,0,pmy_mesh->maxneighbor_);
        else tbid=FindBufferID(0,-n,-m,myfx1,0,pmy_mesh->maxneighbor_);
        if(nlevel>=loc.level || (myox2==n && myox3==m)) {
          neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
            nid-nslist[ranklist[nid]], 0, n, m, neighbor_edge, bufid, tbid);
          nneighbor++;
        }
        bufid+=nf1;
      }
    }
  }
  // corners
  for(int l=-1; l<=1; l+=2) {
    for(int m=-1; m<=1; m+=2) {
      for(int n=-1; n<=1; n+=2) {
        neibt=tree.FindNeighbor(loc,n,m,l,block_bcs,nrbx1,nrbx2,nrbx3,pmy_mesh->root_level);
        if(neibt==NULL) { bufid++; continue;}
        if(neibt->flag==false) { // finer
          int ff1=1-(n+1)/2; // 0 for outer_x1, 1 for inner_x1
          int ff2=1-(m+1)/2; // 0 for outer_x2, 1 for inner_x2
          int ff3=1-(l+1)/2; // 0 for outer_x3, 1 for inner_x3
          neibt=neibt->GetLeaf(ff1,ff2,ff3);
        }
        int nlevel=neibt->loc.level;
        nblevel[l+1][m+1][n+1]=nlevel;
        if(nlevel>=loc.level || (myox1==n && myox2==m && myox3==l)) {
          int nid=neibt->gid;
          int tbid=FindBufferID(-n,-m,-l,0,0,pmy_mesh->maxneighbor_);
          neighbor[nneighbor].SetNeighbor(ranklist[nid], nlevel, nid,
            nid-nslist[ranklist[nid]], n, m, l, neighbor_corner, bufid, tbid);
          nneighbor++;
        }
        bufid++;
      }
    }
  }
  return;
}

//--------------------------------------------------------------------------------------
// \!fn void Mesh::TestConservation(void)
// \brief Calculate and print the total of conservative variables

void Mesh::TestConservation(void)
{
  MeshBlock *pmb = pblock;
  Real tcons[NHYDRO];
  for(int n=0;n<NHYDRO;n++) tcons[n]=0.0;
  while(pmb!=NULL) {
    pmb->IntegrateConservative(tcons);
    pmb=pmb->next;
  }

#ifdef MPI_PARALLEL
  MPI_Allreduce(MPI_IN_PLACE,tcons,NHYDRO,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
#endif

  if(Globals::my_rank==0) {
    std::cout << "Total Conservative : " ;
    for(int n=0;n<NHYDRO;n++)
      std::cout << tcons[n] << " ";
    std::cout << std::endl;
  }

  return;
}

//--------------------------------------------------------------------------------------
// \!fn void MeshBlock::IntegrateConservative(Real *tcons)
// \brief Calculate and print the total of conservative variables

void MeshBlock::IntegrateConservative(Real *tcons)
{
  for(int n=0;n<NHYDRO;n++) {
    for(int k=ks;k<=ke;k++) {
      for(int j=js;j<=je;j++) {
        for(int i=is;i<=ie;i++)
          tcons[n]+=phydro->u(n,k,j,i)*pcoord->GetCellVolume(k,j,i);
      }
    }
  }
  return;
}


//--------------------------------------------------------------------------------------
// \!fn void Mesh::LoadBalancing(Real *clist, int *rlist, int *slist, int *nlist)
// \brief Calculate distribution of MeshBlocks based on the cost list
void Mesh::LoadBalancing(Real *clist, int *rlist, int *slist, int *nlist)
{
  Real totalcost=0, maxcost=0.0, mincost=(FLT_MAX);

  for(int i=0; i<nbtotal; i++) {
    totalcost+=clist[i];
    mincost=std::min(mincost,clist[i]);
    maxcost=std::max(maxcost,clist[i]);
  }
  int j=(Globals::nranks)-1;
  Real targetcost=totalcost/Globals::nranks;
  Real mycost=0.0;
  // create rank list from the end: the master node should have less load
  for(int i=nbtotal-1;i>=0;i--) {
    mycost+=clist[i];
    rlist[i]=j;
    if(mycost >= targetcost && j>0) {
      j--;
      totalcost-=mycost;
      mycost=0.0;
      targetcost=totalcost/(j+1);
    }
  }
  slist[0]=0;
  j=0;
  for(int i=1;i<nbtotal;i++) { // make the list of nbstart
    if(rlist[i]!=rlist[i-1]) {
      nlist[j]=i-nslist[j];
      slist[++j]=i;
    }
  }
  nlist[j]=nbtotal-slist[j];

#ifdef MPI_PARALLEL
  if(nbtotal % Globals::nranks != 0 && adaptive == false
  && maxcost == mincost && Globals::my_rank==0) {
    std::cout << "### Warning in LoadBalancing" << std::endl
              << "The number of MeshBlocks cannot be divided evenly. "
              << "This will cause a poor load balance." << std::endl;
  }
#endif
  return;
}

//--------------------------------------------------------------------------------------
// \!fn void Mesh::MeshRefinement(ParameterInput *pin)
// \brief Main function for mesh refinement
void Mesh::MeshRefinement(ParameterInput *pin)
{
  MeshBlock *pmb;
#ifdef MPI_PARALLEL
  MPI_Request areq[4];
  // start sharing the cost list
  MPI_Iallgatherv(MPI_IN_PLACE, nblist[Globals::my_rank], MPI_INT,
                  costlist, nblist, nslist, MPI_INT, MPI_COMM_WORLD, &areq[3]);
#endif
  int *nref = new int [Globals::nranks];
  int *nderef = new int [Globals::nranks];


  // collect information of refinement from all the meshblocks
  // collect the number of the blocks to be (de)refined
  nref[Globals::my_rank]=0;
  nderef[Globals::my_rank]=0;
  pmb=pblock;
  while(pmb!=NULL) {
    if(pmb->pmr->refine_flag_== 1) nref[Globals::my_rank]++;
    if(pmb->pmr->refine_flag_==-1) nderef[Globals::my_rank]++;
    pmb=pmb->next;
  }
#ifdef MPI_PARALLEL
  // if this does not work due to a version issue, replace these with blocking AllGather
  MPI_Iallgather(MPI_IN_PLACE, 1, MPI_INT, nref,   1, MPI_INT, MPI_COMM_WORLD, &areq[0]);
  MPI_Iallgather(MPI_IN_PLACE, 1, MPI_INT, nderef, 1, MPI_INT, MPI_COMM_WORLD, &areq[1]);
  MPI_Waitall(2, areq, MPI_STATUS_IGNORE);
#endif

  // count the number of the blocks to be (de)refined and displacement
  int tnref=0, tnderef=0;
  int *rdisp = new int [Globals::nranks];
  int *ddisp = new int [Globals::nranks];
  int *bnref = new int [Globals::nranks];
  int *bnderef = new int [Globals::nranks];
  int *brdisp = new int [Globals::nranks];
  int *bddisp = new int [Globals::nranks];
  for(int n=0; n<Globals::nranks; n++) {
    bnref[n]   = nref[n]*sizeof(LogicalLocation);
    bnderef[n] = nderef[n]*sizeof(LogicalLocation);
    rdisp[n] = tnref;
    ddisp[n] = tnderef;
    brdisp[n] = tnref*sizeof(LogicalLocation);
    bddisp[n] = tnderef*sizeof(LogicalLocation);
    tnref  += nref[n];
    tnderef+= nderef[n];
  }
  if(Globals::my_rank==0) {
    std::cout << tnref << " blocks need to be refined, and " 
              << tnderef << " blocks can be derefined." << std::endl;
  }
  if(tnref==0 && tnderef==0) {
    delete [] nref;
    delete [] nderef;
    delete [] bnref;
    delete [] bnderef;
    delete [] rdisp;
    delete [] ddisp;
    delete [] brdisp;
    delete [] bddisp;
    return;
  }

  // allocate memory for the location arrays
  LogicalLocation *lref, *lderef, *clderef;
  int *fref;
  if(tnref!=0) {
    lref = new LogicalLocation[tnref];
    fref = new int[tnref];
  }

  int minbl=2;
  if(mesh_size.nx2 > 1) minbl=4;
  if(mesh_size.nx3 > 1) minbl=8;
  if(tnderef>minbl) {
    lderef = new LogicalLocation[tnderef];
    clderef = new LogicalLocation[tnderef/minbl];
  }

  // collect the locations and costs
  int iref = rdisp[Globals::my_rank], ideref = ddisp[Globals::my_rank];
  pmb=pblock;
  while(pmb!=NULL) {
    if(pmb->pmr->refine_flag_== 1) {
      lref[iref]=pmb->loc;
      fref[iref]=pmb->pmr->neighbor_rflag_;
      iref++;
    }
    if(pmb->pmr->refine_flag_==-1 && tnderef>minbl) {
      lderef[ideref]=pmb->loc;
      ideref++;
    }
    pmb=pmb->next;
  }
#ifdef MPI_PARALLEL
  if(tnref>0 && tnderef>minbl) {
    MPI_Iallgatherv(MPI_IN_PLACE, bnref[Globals::my_rank],   MPI_BYTE,
                    lref,   bnref,   brdisp, MPI_BYTE, MPI_COMM_WORLD, &areq[0]);
    MPI_Iallgatherv(MPI_IN_PLACE, nref[Globals::my_rank], MPI_INT,
                    fref, nref, rdisp, MPI_INT, MPI_COMM_WORLD, &areq[1]);
    MPI_Iallgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                    lderef, bnderef, bddisp, MPI_BYTE, MPI_COMM_WORLD, &areq[2]);
    MPI_Waitall(3, areq, MPI_STATUS_IGNORE);
  }
  else if(tnref>0) {
    MPI_Iallgatherv(MPI_IN_PLACE, bnref[Globals::my_rank],   MPI_BYTE,
                    lref,   bnref,   brdisp, MPI_BYTE, MPI_COMM_WORLD, &areq[0]);
    MPI_Iallgatherv(MPI_IN_PLACE, nref[Globals::my_rank], MPI_INT,
                    fref, nref, rdisp, MPI_INT, MPI_COMM_WORLD, &areq[1]);
    MPI_Waitall(2, areq, MPI_STATUS_IGNORE);
  }
  else if(tnderef>minbl) {
    MPI_Allgatherv(MPI_IN_PLACE, bnderef[Globals::my_rank], MPI_BYTE,
                    lderef, bnderef, bddisp, MPI_BYTE, MPI_COMM_WORLD);
  }
#endif
  delete [] nref;
  delete [] bnref;
  delete [] ddisp;
  delete [] rdisp;
  delete [] brdisp;

  // calculate the list of the newly derefined blocks
  int ke=0, je=0, ctnd=0;
  if(mesh_size.nx2 > 1) je=1;
  if(mesh_size.nx3 > 1) ke=1;
  for(int n=0; n<tnderef; n++) {
    if((lderef[n].lx1&1L)==0 && (lderef[n].lx2&1L)==0 && (lderef[n].lx3&1L)==0) {
      int r=n+1, rr=0;
      for(long int k=0;k<=ke;k++) {
        for(long int j=0;j<=je;j++) {
          for(long int i=0;i<=1;i++) {
            if((lderef[n].lx1+i)==lderef[r].lx1
            && (lderef[n].lx2+j)==lderef[r].lx2
            && (lderef[n].lx3+k)==lderef[r].lx3
            &&  lderef[n].level ==lderef[r].level)
              rr++;
            r++;
          }
        }
      }
      if(rr==minbl) {
        clderef[ctnd].lx1  =(lderef[n].lx1>>1L);
        clderef[ctnd].lx2  =(lderef[n].lx2>>1L);
        clderef[ctnd].lx3  =(lderef[n].lx3>>1L);
        clderef[ctnd].level=lderef[n].level-1;
        ctnd++;
      }
    }
  }
  // sort the lists by level
  if(ctnd>1)
    std::sort(clderef, &(clderef[ctnd-1]), LogicalLocation::Greater);

  delete [] nderef;
  delete [] bnderef;
  delete [] bddisp;
  if(tnderef>minbl)
    delete [] lderef;

  // Now the lists of the blocks to be refined and derefined are completed
  if(Globals::my_rank==0) {
    for(int n=0; n<tnref; n++) {
      std::cout << "Refine   " << n << " :  Location " << lref[n].lx1 << " "
                << lref[n].lx2 << " " << lref[n].lx3 << " " << lref[n].level
                << " " << fref[n] << std::endl;
    }
    for(int n=0; n<ctnd; n++)
      std::cout << "Derefine " << n << " :  Location " << clderef[n].lx1 << " " << 
           clderef[n].lx2 << " " << clderef[n].lx3 << " " << clderef[n].level << std::endl;
  }

  // start tree manipulation
  // Step 1. perform refinement

  // Step 2. perform derefinement

  // Step 3. assign new costs
#ifdef MPI
  // receive the cost list
  MPI_Wait(&areq[3], MPI_STATUS_IGNORE);
#endif


  // re-initialize the MeshBlocks
  pmb=pblock;
  while(pmb!=NULL) {
    pblock->SearchAndSetNeighbors(tree, ranklist, nslist);
    pmb=pmb->next;
  }
  Initialize(2, pin);
  

  // free temporary arrays
  if(tnref!=0) {
    delete [] lref;
    delete [] fref;
  }
  if(tnderef>minbl)
    delete [] clderef;
  return;
}
