/*
 * Copyright (C) 2020, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact myViewer@inria.fr and/or George.Drettakis@inria.fr
 */



#include "myRay.hpp"

namespace myViewer
{
	/*static*/ myViewer_RAYCASTER_EXPORT const float	RayHit::InfinityDist = std::numeric_limits<float>::infinity();

	Ray::Ray( const myViewer::Vector3f& orig, const myViewer::Vector3f& dir )
		: _orig(orig), _dir(dir)
	{
		if (_dir[0] != 0.f || _dir[1] != 0.f || _dir[2] != 0.f)
			_dir.normalize();
	}

	RayHit::RayHit( const Ray& r, float dist, const BCCoord& coord,
					const myViewer::Vector3f& normal, const Primitive& prim )
					: _ray(r), _dist(dist), _coord(coord), _normal(normal), _prim(prim)
	{
		_dist = std::max(dist, 0.f);

		// normalize '_normal'
		float len = length(_normal);
		if (len > 1e-10)
			_normal = _normal / len;

	}

	myViewer::Vector3f			RayHit::interpolateUV( void ) const
	{
		float ucoord = barycentricCoord().u;
		float vcoord = barycentricCoord().v;
		return myViewer::Vector3f(std::max((1.f-ucoord-vcoord), 0.f), ucoord, vcoord);
	}

} // namespace myViewer
