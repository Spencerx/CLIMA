include("DGBalanceLawDiscretizations_kernels.jl")
using Random
using CLIMA.Grids
using StaticArrays
let
  DFloat = Float64

  nelem = 4
  dim = 2
  N = 4

  nstate = 5
  nauxstate = 2

  flux! = (x...) -> nothing
  numerical_flux! = (x...) -> nothing
  source! = nothing
  numerical_boundary_flux! = nothing

  nviscstate = 6
  ngradstate = 3
  states_grad = Tuple(1:4)
  viscous_transform! = (x...) -> nothing
  gradient_transform! = (x...) -> nothing
  viscous_penalty! = (x...) -> nothing
  viscous_boundary_penalty! = nothing

  t = zero(DFloat)

  # Generate a random mapping
  nface = 2dim
  rnd = MersenneTwister(0)
  probability_boundary_face = 0.01

  elemtoelem = zeros(Int, nface, nelem)
  elemtobndy = zeros(Int, nface, nelem)
  elemtoface = zeros(Int, nface, nelem)
  elemtoordr = ones(Int, nface, nelem)

  Faces = Set(1:nelem*nface)
  for gf1 in Faces
    pop!(Faces, gf1)
    e1 = div(gf1-1, nface) + 1
    f1 = ((gf1-1) % nface) + 1

    @assert elemtoelem[f1, e1] == 0
    @assert elemtoface[f1, e1] == 0
    @assert elemtobndy[f1, e1] == 0
    if isempty(Faces) || probability_boundary_face > rand(rnd)
      elemtoelem[f1, e1] = e1
      elemtoface[f1, e1] = f1
      elemtobndy[f1, e1] = 1
    else
      gf2 = rand(rnd, Faces)
      pop!(Faces, gf2)
      e2 = div(gf2-1, nface) + 1
      f2 = ((gf2-1) % nface) + 1
      @assert elemtoelem[f2, e2] == 0
      @assert elemtoface[f2, e2] == 0
      @assert elemtobndy[f2, e2] == 0
      elemtoelem[f1, e1], elemtoelem[f2, e2] = e2, e1
      elemtoface[f1, e1], elemtoface[f2, e2] = f2, f1
    end
  end
  vmapM, vmapP = Grids.mappings(N, elemtoelem, elemtoface, elemtoordr)

  # Generate random geometry terms and solutions
  Nq = N + 1
  Nqk = dim == 3 ? N + 1 : 1
  Q = rand(rnd, DFloat, Nq, Nq, Nqk, nstate, nelem)
  Qvisc = rand(rnd, DFloat, Nq, Nq, Nqk, nviscstate, nelem)
  auxstate = rand(rnd, DFloat, Nq, Nq, Nqk, nauxstate, nelem)
  vgeo = rand(rnd, DFloat, Nq, Nq, Nqk, _nvgeo, nelem)
  sgeo = rand(rnd, DFloat, _nsgeo, Nq^(dim-1), nface, nelem)
  rhs = similar(Q)
  D = rand(rnd, DFloat, Nq, Nq)


  # Make sure the entries of the mass matrix satisfy the inverse relation
  vgeo[:, :, :, _MJ, :] .+= 3
  vgeo[:, :, :, _MJI, :] .= 1 ./ vgeo[:, :, :, _MJ, :]

  # FIXME: Do we need to correct the surface terms in any way?

  # Call the volume kernel
  volumerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate),
             Val(nauxstate), flux!, source!, rhs, Q, Qvisc, auxstate, vgeo, t,
             D, 1:nelem)

  # Call the volume kernel
  facerhs!(Val(dim), Val(N), Val(nstate), Val(nviscstate), Val(nauxstate),
           numerical_flux!, numerical_boundary_flux!, rhs, Q, Qvisc, auxstate,
           vgeo, sgeo, t, vmapM, vmapP, elemtobndy, 1:nelem)

  if nviscstate > 0
    volumeviscterms!(Val(dim), Val(N), Val(nstate), Val(states_grad),
                     Val(ngradstate), Val(nviscstate), Val(nauxstate),
                     viscous_transform!, gradient_transform!, Q, Qvisc,
                     auxstate, vgeo, t, D, 1:nelem)

    faceviscterms!(Val(dim), Val(N), Val(nstate), Val(states_grad),
                   Val(ngradstate), Val(nviscstate), Val(nauxstate),
                   viscous_penalty!, viscous_boundary_penalty!,
                   gradient_transform!, Q, Qvisc, auxstate, vgeo, sgeo, t,
                   vmapM, vmapP, elemtobndy, 1:nelem)
  end

end
