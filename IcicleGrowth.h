#pragma once

// Reduced, physics-inspired liquid-to-ice bridge. C++17, no graphics dependency.
// Surface flow is a graph-advection estimate, NOT an SPH particle velocity.
// Model length units and seconds are used; temperature maps to a phenomenological freezing rate; no heat equation.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace icicle_growth
{
struct Vec3
{
    float x = 0, y = 0, z = 0;
    Vec3() = default;
    explicit Vec3(float value) : x(value), y(value), z(value) {}
    Vec3(float a, float b, float c) : x(a), y(b), z(c) {}
    Vec3 operator+(Vec3 b) const { return {x+b.x, y+b.y, z+b.z}; }
    Vec3 operator-(Vec3 b) const { return {x-b.x, y-b.y, z-b.z}; }
    Vec3 operator*(float s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(float s) const { return *this * (1.0f/s); }
    Vec3& operator+=(Vec3 b) { *this = *this+b; return *this; }
    Vec3& operator*=(float s) { *this = *this*s; return *this; }
};
inline float Dot(Vec3 a, Vec3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
inline float Length(Vec3 a) { return std::sqrt(Dot(a,a)); }
inline Vec3 Unit(Vec3 a, Vec3 fallback = {0,-1,0})
{
    const float l = Length(a);
    return std::isfinite(l) && l > 1e-8f ? a/l : fallback;
}
inline double Positive(double a)
{
    return std::isfinite(a) ? std::max(0.0, a) : 0.0;
}

struct SurfaceSample
{
    Vec3 position, normal, surfaceVelocity, flowDirection;
    double water = 0, dripMass = 0;
};
using Adjacency = std::vector<std::vector<unsigned int>>;
inline Adjacency BuildAdjacency(std::size_t count,
                               const std::vector<unsigned int>& triangles)
{
    Adjacency result(count);
    for (std::size_t t=0; t+2<triangles.size(); t+=3)
    {
        const unsigned int a=triangles[t], b=triangles[t+1], c=triangles[t+2];
        if (a>=count || b>=count || c>=count) continue;
        result[a].push_back(b); result[a].push_back(c);
        result[b].push_back(a); result[b].push_back(c);
        result[c].push_back(a); result[c].push_back(b);
    }
    for (auto& neighbors : result)
    {
        std::sort(neighbors.begin(),neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(),neighbors.end()),neighbors.end());
    }
    return result;
}

// Expected graph hop / second: kFlow * sum(weight_ij * edge_ij).
// This uses flowDirection, mesh geometry and the existing flow time scale;
// it does not assert that the transport solver tracks physical momentum.
// At sinks, use a mass-flux-weighted incoming estimate when available.
inline std::vector<Vec3> EstimateSurfaceVelocities(
    const std::vector<SurfaceSample>& samples, const Adjacency& adjacency,
    float kFlow, float transitionGamma)
{
    const std::size_t count=samples.size();
    std::vector<Vec3> outgoing(count), incoming(count);
    std::vector<double> incomingFlux(count,0);
    const float rate=static_cast<float>(Positive(kFlow));
    const float gamma=std::max(0.05f,transitionGamma);
    for (std::size_t i=0; i<count && i<adjacency.size(); ++i)
    {
        std::vector<float> weights(adjacency[i].size(),0);
        float sum=0;
        const Vec3 direction=Unit(samples[i].flowDirection,{0,0,0});
        const Vec3 normal=Unit(samples[i].normal,{0,0,0});
        for (std::size_t n=0; n<adjacency[i].size(); ++n)
        {
            const auto j=adjacency[i][n];
            if (j>=count) continue;
            const Vec3 edge=samples[j].position-samples[i].position;
            const Vec3 tangent=edge-normal*Dot(edge,normal);
            weights[n]=std::pow(std::max(0.0f,Dot(direction,Unit(tangent,{0,0,0}))),gamma);
            sum+=weights[n];
        }
        if (sum<=1e-8f) continue;
        for (std::size_t n=0; n<adjacency[i].size(); ++n)
        {
            const auto j=adjacency[i][n];
            if (j>=count || weights[n]<=0) continue;
            const float weight=weights[n]/sum;
            const Vec3 edgeVelocity=(samples[j].position-samples[i].position)*rate;
            outgoing[i]+=edgeVelocity*weight;
            const double flux=Positive(samples[i].water)*rate*weight;
            incoming[j]+=edgeVelocity*static_cast<float>(flux);
            incomingFlux[j]+=flux;
        }
    }
    for (std::size_t i=0; i<count; ++i)
        if (Length(outgoing[i])<1e-8f && incomingFlux[i]>1e-12)
            outgoing[i]=incoming[i]/static_cast<float>(incomingFlux[i]);
    return outgoing;
}

struct RootBinding
{
    bool valid = false;
    std::array<unsigned int,3> indices{{0,0,0}};
    std::array<float,3> weights{{0,0,0}};
};
struct Supply
{
    double mass = 0;                // mass in THIS transport step, not mass/s
    Vec3 velocityWorld;            // surface motion + graph flow estimate
};
inline std::vector<Supply> AllocateDripSupplies(
    const std::vector<SurfaceSample>& samples,
    const std::vector<Vec3>& relativeVelocities,
    const std::vector<RootBinding>& roots)
{
    std::vector<double> requests(samples.size(),0);
    for (const auto& root:roots)
        if (root.valid)
            for (std::size_t k=0;k<3;++k)
                if (root.indices[k]<samples.size())
                    requests[root.indices[k]]+=Positive(root.weights[k]);
    std::vector<Supply> result(roots.size());
    for (std::size_t r=0;r<roots.size();++r)
    {
        if (!roots[r].valid) continue;
        Vec3 momentum;
        for (std::size_t k=0;k<3;++k)
        {
            const auto i=roots[r].indices[k];
            if (i>=samples.size()) continue;
            const double mass=Positive(samples[i].dripMass)*Positive(roots[r].weights[k])/
                              std::max(1.0,requests[i]);
            const Vec3 relative=i<relativeVelocities.size()?relativeVelocities[i]:Vec3{};
            momentum+=(samples[i].surfaceVelocity+relative)*static_cast<float>(mass);
            result[r].mass+=mass;
        }
        if (result[r].mass>1e-12)
            result[r].velocityWorld=momentum/static_cast<float>(result[r].mass);
    }
    return result;
}

// Only dripping vertices need a velocity for inlet allocation. Avoid repeating
// the full-mesh graph pass and allocating a temporary weight vector per vertex.
inline float GraphWeight(const std::vector<SurfaceSample>& s,
                         std::size_t i, std::size_t j, float gamma)
{
    if (i>=s.size() || j>=s.size()) return 0;
    const Vec3 normal=Unit(s[i].normal,{0,0,0});
    const Vec3 edge=s[j].position-s[i].position;
    const Vec3 tangent=edge-normal*Dot(edge,normal);
    return std::pow(std::max(0.0f,Dot(Unit(s[i].flowDirection,{0,0,0}),
                                    Unit(tangent,{0,0,0}))),gamma);
}
struct GraphVelocityEstimate { Vec3 velocity; float weightSum=0; };
inline GraphVelocityEstimate EstimateOneVertex(
    const std::vector<SurfaceSample>& s, const Adjacency& adjacency,
    std::size_t i, float rate, float gamma)
{
    GraphVelocityEstimate result;
    if (i>=s.size() || i>=adjacency.size()) return result;
    for (const auto j:adjacency[i])
    {
        if (j>=s.size()) continue;
        const float weight=GraphWeight(s,i,j,gamma);
        result.weightSum+=weight;
        result.velocity+=(s[j].position-s[i].position)*(rate*weight);
    }
    if (result.weightSum>1e-8f) result.velocity=result.velocity/result.weightSum;
    else result.velocity={};
    return result;
}
inline std::vector<Vec3> EstimateDripVelocities(
    const std::vector<SurfaceSample>& s, const Adjacency& adjacency,
    float kFlow, float transitionGamma)
{
    std::vector<Vec3> result(s.size());
    const float rate=static_cast<float>(Positive(kFlow));
    if (rate<=0) return result;
    const float gamma=std::max(.05f,transitionGamma);
    for (std::size_t i=0;i<s.size() && i<adjacency.size();++i)
    {
        if (Positive(s[i].dripMass)<=0) continue;
        result[i]=EstimateOneVertex(s,adjacency,i,rate,gamma).velocity;
        if (Length(result[i])>=1e-8f) continue;
        Vec3 incoming;
        double incomingFlux=0;
        for (const auto donor:adjacency[i])
        {
            if (donor>=s.size() || Positive(s[donor].water)<=0) continue;
            const auto estimate=EstimateOneVertex(s,adjacency,donor,rate,gamma);
            if (estimate.weightSum<=1e-8f) continue;
            const double flux=Positive(s[donor].water)*rate*
                              GraphWeight(s,donor,i,gamma)/estimate.weightSum;
            incoming+=(s[i].position-s[donor].position)*(rate*static_cast<float>(flux));
            incomingFlux+=flux;
        }
        if (incomingFlux>1e-12) result[i]=incoming/static_cast<float>(incomingFlux);
    }
    return result;
}

// A persistent root captures nearby detached water on a similarly oriented
// surface. Each vertex's drip mass goes to ONE root, including its first step.
// This is a local capture-region approximation, not a solved film trajectory.
inline std::vector<Supply> AllocateNearbyDripSupplies(
    const std::vector<SurfaceSample>& samples,
    const std::vector<Vec3>& relativeVelocities,
    const std::vector<RootBinding>& roots, float captureRadius)
{
    std::vector<Supply> result(roots.size());
    std::vector<Vec3> positions(roots.size()),normals(roots.size());
    std::vector<bool> valid(roots.size(),false);
    for (std::size_t r=0;r<roots.size();++r)
    {
        if (!roots[r].valid) continue;
        double sum=0;
        for (std::size_t k=0;k<3;++k)
        {
            const auto i=roots[r].indices[k];
            if (i>=samples.size()) continue;
            const float w=static_cast<float>(Positive(roots[r].weights[k]));
            positions[r]+=samples[i].position*w;
            normals[r]+=samples[i].normal*w;
            sum+=w;
        }
        if (sum<=1e-12) continue;
        positions[r]=positions[r]/static_cast<float>(sum);
        normals[r]=Unit(normals[r],{0,0,0});
        valid[r]=true;
    }
    const float radius=static_cast<float>(Positive(captureRadius));
    for (std::size_t i=0;i<samples.size();++i)
    {
        const double mass=Positive(samples[i].dripMass);
        if (mass<=0) continue;
        std::size_t nearest=roots.size();
        float best=radius*radius;
        const Vec3 sourceNormal=Unit(samples[i].normal,{0,0,0});
        for (std::size_t r=0;r<roots.size();++r)
        {
            if (!valid[r]) continue;
            bool exact=false;
            for (std::size_t k=0;k<3;++k)
                exact=exact || (roots[r].indices[k]==i && roots[r].weights[k]>0);
            if (!exact && Dot(sourceNormal,normals[r])<.5f) continue;
            const Vec3 delta=samples[i].position-positions[r];
            const float distance=Dot(delta,delta);
            if (distance<=best && (nearest==roots.size() || distance<best))
            {
                best=distance;
                nearest=r;
            }
        }
        if (nearest==roots.size()) continue; // unclaimed water remains shed
        const Vec3 relative=i<relativeVelocities.size()?relativeVelocities[i]:Vec3{};
        result[nearest].velocityWorld+=(samples[i].surfaceVelocity+relative)*static_cast<float>(mass);
        result[nearest].mass+=mass;
    }
    for (auto& supply:result)
        if (supply.mass>1e-12)
            supply.velocityWorld=supply.velocityWorld/static_cast<float>(supply.mass);
    return result;
}

struct Parameters
{
    float gravityAcceleration = 9.81f; // model length units/s^2, calibrate scale
    float airCouplingRate = 0.0f;      // 1/s; zero disables air coupling
    float surfaceCouplingRate = 8.0f;  // 1/s; viscous relaxation to attached ice
    float flowMotionGain = 1.0f; // 0: gravity direction, 1: relative flow, >1: art exaggeration
    float travelSpeed = 0.4f;          // model length units/s, lumped transit
    float minimumTravelLength = 0.01f;
    float tipDripRate = 0.1f;           // 1/s; liquid can leave without freezing
    float maxSubstep = 1.0f/240.0f;
};
// Artist-controlled first-order kinetics, NOT calibrated thermodynamics.
// tauRef is the e-folding time at -referenceUndercooling C, without runoff.
struct FreezingControls
{
    float referenceTimeSeconds = 0.25f;
    float temperatureC = -10.0f;
    float referenceUndercooling = 10.0f;
};
inline float EffectiveFreezeRate(const FreezingControls& p)
{
    const double cold=Positive(-p.temperatureC);
    return static_cast<float>(cold / std::max(1e-4,Positive(p.referenceUndercooling))
        / std::max(1e-4,Positive(p.referenceTimeSeconds)));
}
struct LiquidState
{
    double transitMass=0, tipMass=0;
    Vec3 transitVelocityWorld, tipVelocityWorld;
    double totalInput=0, totalFrozen=0, totalDripped=0;
    Vec3 lastDirection{0,-1,0};
    double MassError() const
    {
        return totalInput-transitMass-tipMass-totalFrozen-totalDripped;
    }
};
struct FrozenSample
{
    double mass=0;
    Vec3 directionWorld;
    float timeOffset=0,duration=0;
    float relativeSpeed=0; // liquid speed relative to the moving ice tip
};
struct StepResult
{
    double frozenMass=0;
    Vec3 frozenDirectionMoment; // sum(delta frozen mass * direction)
    std::vector<FrozenSample> samples; // chronological, before frame averaging
};
inline void MixLiquid(double& mass, Vec3& velocity, double added, Vec3 incoming)
{
    added=Positive(added);
    if (added<=0) return;
    const double total=mass+added;
    velocity=velocity*static_cast<float>(mass/total)+incoming*static_cast<float>(added/total);
    mass=total;
}
inline Vec3 AdvanceVelocity(Vec3 velocity, Vec3 iceVelocity, Vec3 gravity,
                            Vec3 wind, float dt, const Parameters& p)
{
    const float surface=static_cast<float>(Positive(p.surfaceCouplingRate));
    const float air=static_cast<float>(Positive(p.airCouplingRate));
    const float drag=surface+air;
    if (drag<1e-8f) return velocity+gravity*dt;
    const Vec3 target=(gravity+iceVelocity*surface+wind*air)/drag;
    return target+(velocity-target)*std::exp(-drag*dt);
}

// Lumped transit -> tip liquid -> freezing. No temperature or fluid particles.
// Inlet mass is distributed across substeps; freezing is an exact first-order
// removal together with runoff. Gravity/air act on liquid velocity, not ice.
inline StepResult AdvanceLiquid(LiquidState& state, Supply supply, float dt,
                                float pathLength, Vec3 tipVelocityWorld,
                                Vec3 gravityDirection, Vec3 wind,
                                float freezeRate, const Parameters& p)
{
    StepResult result;
    if (!(dt>0) || !std::isfinite(dt)) return result;
    const Vec3 down=Unit(gravityDirection);
    const Vec3 gravity=down*static_cast<float>(Positive(p.gravityAcceleration));
    const float safeSubstep=std::max(1e-4f,p.maxSubstep);
    const int steps=std::max(1,static_cast<int>(std::ceil(dt/safeSubstep)));
    result.samples.reserve(static_cast<std::size_t>(steps));
    const float h=dt/static_cast<float>(steps);
    const double added=Positive(supply.mass)/steps;
    const double freezing=Positive(freezeRate), runoff=Positive(p.tipDripRate);
    const double removal=freezing+runoff;
    const double length=std::max(Positive(pathLength),
                                std::max(1e-5,Positive(p.minimumTravelLength)));
    const double transitRate=Positive(p.travelSpeed)/length;
    for (int step=0;step<steps;++step)
    {
        MixLiquid(state.transitMass,state.transitVelocityWorld,added,supply.velocityWorld);
        state.totalInput+=added;
        if (state.transitMass>1e-15)
        {
            state.transitVelocityWorld=AdvanceVelocity(state.transitVelocityWorld,
                tipVelocityWorld,gravity,wind,h,p);
            const double delivered=state.transitMass*(-std::expm1(-transitRate*h));
            state.transitMass-=delivered;
            MixLiquid(state.tipMass,state.tipVelocityWorld,delivered,state.transitVelocityWorld);
        }
        if (state.tipMass<=1e-15) continue;
        state.tipVelocityWorld=AdvanceVelocity(state.tipVelocityWorld,
            tipVelocityWorld,gravity,wind,h,p);
        const Vec3 relativeVelocity=state.tipVelocityWorld-tipVelocityWorld;
        // Blend in velocity space; gain=1 preserves the reduced relative-flow model.
        // The gravity baseline uses the surface-drag terminal speed only as a scale.
        const float speedScale=static_cast<float>(Positive(p.gravityAcceleration)) /
            std::max(0.01f,static_cast<float>(Positive(p.surfaceCouplingRate)));
        const Vec3 baseline=down*speedScale;
        const Vec3 directed=baseline+(relativeVelocity-baseline)*
            static_cast<float>(Positive(p.flowMotionGain));
        state.lastDirection=Unit(directed,down);
        if (removal<=0) continue;
        const double removed=state.tipMass*(-std::expm1(-removal*h));
        const double frozen=removed*freezing/removal;
        state.tipMass-=removed;
        state.totalFrozen+=frozen;
        state.totalDripped+=removed-frozen;
        result.frozenMass+=frozen;
        result.frozenDirectionMoment+=state.lastDirection*static_cast<float>(frozen);
        if (frozen>0) result.samples.push_back(
            {frozen,state.lastDirection,(step+1)*h,h,Length(relativeVelocity)});
    }
    return result;
}

// Spatial resolution must not set physical length growth. mu=rho*A after unit
// calibration; the default preserves the former .001 mass / .01 length ratio.
struct GrowthResolution
{
    double linearMassDensity=.1; // model mass / model length, NOT calibrated SI
    float maxSegmentLength=.01f;
    float maxWindowSeconds=.05f;
    float maxTurnDegrees=5;
};

// Optional numerical response model for NEW growth only. This is a tunable
// approximation, not an experimentally measured freezing response time.
// Filter after conversion into the attachment's local frame. Filtering world
// directions and then attaching them again would mix carrier rotation twice.
struct DirectionResponseParameters
{
    bool enabled=true;
    float responseTimeSeconds=.03f;
    float minimumRelativeSpeed=1e-4f; // model length units/s, not an SI constant
};
struct DirectionResponseState
{
    bool initialized=false;
    Vec3 directionLocal{0,-1,0};
    void Reset() { initialized=false; directionLocal={0,-1,0}; }
};
inline Vec3 RelaxGrowthDirection(DirectionResponseState& state,Vec3 rawLocal,
    float relativeSpeed,float duration,Vec3 fallbackLocal,
    const DirectionResponseParameters& p)
{
    const Vec3 target=Unit(rawLocal,Unit(fallbackLocal));
    if (!p.enabled) {
        state.directionLocal=target; state.initialized=true;
        return target;
    }
    const bool reliable=std::isfinite(relativeSpeed) &&
        relativeSpeed>std::max(1e-8f,p.minimumRelativeSpeed) &&
        std::isfinite(Length(rawLocal)) && Length(rawLocal)>1e-8f;
    if (!state.initialized) {
        state.directionLocal=reliable?target:Unit(fallbackLocal);
        state.initialized=true;
        return state.directionLocal;
    }
    // A near-zero relative velocity has an ill-defined direction. Keep the
    // last reliable local direction instead of jumping to world-down.
    if (!reliable) return state.directionLocal;
    const float tau=static_cast<float>(Positive(p.responseTimeSeconds));
    if (tau<=1e-8f) { state.directionLocal=target; return target; }
    const float dt=static_cast<float>(Positive(duration));
    if (dt<=0) return state.directionLocal;
    const float alpha=-std::expm1(-dt/tau);
    const Vec3 previous=Unit(state.directionLocal);
    const float cosine=std::clamp(Dot(previous,target),-1.f,1.f);
    if (cosine>.9995f) {
        state.directionLocal=Unit(previous*(1-alpha)+target*alpha,previous);
    } else {
        Vec3 tangent=target-previous*cosine;
        if (Length(tangent)<1e-6f) {
            // At an exact reversal the rotation plane is not unique. Select
            // a deterministic perpendicular axis; never normalize a zero sum.
            const Vec3 axis=std::abs(previous.x)<.8f?Vec3{1,0,0}:Vec3{0,1,0};
            tangent=axis-previous*Dot(axis,previous);
        }
        tangent=Unit(tangent);
        const float angle=std::acos(cosine)*alpha;
        state.directionLocal=Unit(previous*std::cos(angle)+tangent*std::sin(angle),previous);
    }
    return state.directionLocal;
}
struct GrowthChunk
{
    double mass=0;
    Vec3 directionLocal;
    double time=0;
};
struct GrowthBuffer
{
    double mass=0,startTime=0,lastTime=0;
    Vec3 moment;
    void Clear() { mass=0; startTime=lastTime=0; moment={}; }
};
inline void FlushGrowthBuffer(GrowthBuffer& buffer,std::vector<GrowthChunk>& chunks)
{
    if (buffer.mass>1e-15)
        chunks.push_back({buffer.mass,Unit(buffer.moment/static_cast<float>(buffer.mass)),buffer.lastTime});
    buffer.Clear();
}
inline void AccumulateFrozenSample(GrowthBuffer& buffer,double mass,Vec3 directionLocal,
    double time,float duration,const GrowthResolution& resolution,std::vector<GrowthChunk>& chunks)
{
    mass=Positive(mass);
    if (mass<=1e-15 || !(resolution.linearMassDensity>0) ||
        !(resolution.maxSegmentLength>0)) return;
    const Vec3 direction=Unit(directionLocal);
    const double maxMass=resolution.linearMassDensity*resolution.maxSegmentLength;
    if (!(maxMass>1e-15)) return;
    const float angle=std::clamp(resolution.maxTurnDegrees,0.01f,179.f)*.01745329252f;
    const double window=std::max(1e-5f,resolution.maxWindowSeconds);
    const double start=time-std::max(0.f,duration);
    // Flush the old direction BEFORE adding a differently directed sample.
    // Opposite phases are never averaged into a near-zero vector.
    if (buffer.mass>0 && (Dot(Unit(buffer.moment/static_cast<float>(buffer.mass)),direction)<std::cos(angle) ||
                         start-buffer.startTime>=window))
        FlushGrowthBuffer(buffer,chunks);
    double remaining=mass;
    while (remaining>1e-15)
    {
        if (buffer.mass<=0) buffer.startTime=start+(time-start)*(1-remaining/mass);
        const double capacity=maxMass-buffer.mass;
        if (capacity<=1e-15) { FlushGrowthBuffer(buffer,chunks); continue; }
        const double take=std::min(remaining,capacity);
        buffer.mass+=take;
        buffer.moment+=direction*static_cast<float>(take);
        remaining-=take;
        buffer.lastTime=start+(time-start)*(1-remaining/mass);
        if (buffer.mass>=maxMass-1e-15 || buffer.lastTime-buffer.startTime>=window)
            FlushGrowthBuffer(buffer,chunks);
    }
}

struct GroundClip
{
    Vec3 endpoint;
    float fraction=1;
    bool hit=false, startsBelow=false;
};
inline GroundClip ClipGrowthToGround(Vec3 from, Vec3 to, float height, float radius)
{
    const float limit=height+std::max(0.0f,radius);
    GroundClip r; r.endpoint=to;
    if (from.y<limit-1e-6f)
    {
        r.endpoint=from; r.fraction=0; r.hit=true; r.startsBelow=true;
    }
    else if (to.y<=limit && to.y<=from.y)
    {
        const float denominator=from.y-to.y;
        r.fraction=denominator>1e-8f?std::clamp((from.y-limit)/denominator,0.0f,1.0f):0;
        r.endpoint=from+(to-from)*r.fraction;
        r.endpoint.y=limit;
        r.hit=true;
    }
    return r;
}
// Contact geometry is independent of skeleton names, animation frames or FBX.
inline Vec3 Cross(Vec3 a,Vec3 b)
{ return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x}; }
struct Quaternion { float x=0,y=0,z=0,w=1; };
inline Quaternion Normalize(Quaternion q)
{
    const float n=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w);
    if (!(n>1e-8f)) return {};
    return {q.x/n,q.y/n,q.z/n,q.w/n};
}
inline Quaternion Multiply(Quaternion a,Quaternion b)
{
    return Normalize({a.w*b.x+a.x*b.w+a.y*b.z-a.z*b.y,
        a.w*b.y-a.x*b.z+a.y*b.w+a.z*b.x,
        a.w*b.z+a.x*b.y-a.y*b.x+a.z*b.w,
        a.w*b.w-a.x*b.x-a.y*b.y-a.z*b.z});
}
inline Quaternion Conjugate(Quaternion q) { return {-q.x,-q.y,-q.z,q.w}; }
inline Quaternion RotationVector(Vec3 r)
{
    const float angle=Length(r);
    if (angle<1e-8f) return {};
    const float factor=std::sin(angle*.5f)/angle;
    return {r.x*factor,r.y*factor,r.z*factor,std::cos(angle*.5f)};
}
inline Vec3 Rotate(Quaternion q,Vec3 p)
{
    const Vec3 v{q.x,q.y,q.z};
    const Vec3 t=Cross(v,p)*2;
    return p+t*q.w+Cross(v,t);
}
inline Quaternion OrientationFromAxes(Vec3 x,Vec3 y,Vec3 z)
{
    Quaternion q;
    const float trace=x.x+y.y+z.z;
    if (trace>0) {
        const float s=std::sqrt(trace+1)*2;
        q={(y.z-z.y)/s,(z.x-x.z)/s,(x.y-y.x)/s,s*.25f};
    } else if (x.x>y.y && x.x>z.z) {
        const float s=std::sqrt(std::max(0.f,1+x.x-y.y-z.z))*2;
        if (s<1e-8f) return {};
        q={s*.25f,(y.x+x.y)/s,(z.x+x.z)/s,(y.z-z.y)/s};
    } else if (y.y>z.z) {
        const float s=std::sqrt(std::max(0.f,1+y.y-x.x-z.z))*2;
        if (s<1e-8f) return {};
        q={(y.x+x.y)/s,s*.25f,(z.y+y.z)/s,(z.x-x.z)/s};
    } else {
        const float s=std::sqrt(std::max(0.f,1+z.z-x.x-y.y))*2;
        if (s<1e-8f) return {};
        q={(z.x+x.z)/s,(z.y+y.z)/s,s*.25f,(x.y-y.x)/s};
    }
    return Normalize(q);
}
inline Quaternion Interpolate(Quaternion a,Quaternion b,float t)
{
    float d=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
    if (d<0) { b={-b.x,-b.y,-b.z,-b.w}; d=-d; }
    float wa=1-t,wb=t;
    if (d<.9995f) {
        const float angle=std::acos(std::clamp(d,-1.f,1.f));
        const float s=std::sin(angle);
        wa=std::sin((1-t)*angle)/s; wb=std::sin(t*angle)/s;
    }
    return Normalize({a.x*wa+b.x*wb,a.y*wa+b.y*wb,a.z*wa+b.z*wb,a.w*wa+b.w*wb});
}
inline Vec3 AngularVelocity(Quaternion from,Quaternion to,float dt)
{
    if (!(dt>1e-6f)) return {};
    auto q=Multiply(to,Conjugate(from));
    if (q.w<0) q={-q.x,-q.y,-q.z,-q.w};
    const float s=std::sqrt(q.x*q.x+q.y*q.y+q.z*q.z);
    if (s<1e-8f) return {};
    return Vec3{q.x,q.y,q.z}*(2*std::atan2(s,q.w)/(s*dt));
}
// Transport the persistent liquid velocity relative to the moving attachment.
// This is a coordinate-seam policy, not a force or a validated fluid model.
inline Vec3 RebaseCarrierVelocity(Vec3 velocity, Vec3 oldCarrier, Vec3 newCarrier,
                                 Quaternion oldOrientation, Quaternion newOrientation)
{
    return newCarrier+Rotate(newOrientation,
        Rotate(Conjugate(oldOrientation),velocity-oldCarrier));
}
struct RigidPose { Vec3 position; Quaternion orientation; };
inline RigidPose Interpolate(RigidPose a,RigidPose b,float t)
{ return {a.position+(b.position-a.position)*t,Interpolate(a.orientation,b.orientation,t)}; }
inline Vec3 Transform(RigidPose pose,Vec3 p)
{ return pose.position+Rotate(pose.orientation,p); }
inline float MinimumHeight(const std::vector<Vec3>& points,RigidPose pose)
{
    float minimum=0;
    bool first=true;
    for (auto p:points) {
        const float y=Transform(pose,p).y;
        if (first || y<minimum) minimum=y;
        first=false;
    }
    return first?pose.position.y:minimum;
}
struct SweptGroundContact
{
    bool hit=false,startsBelow=false;
    float fraction=1;
    RigidPose pose;
};
// Sample the shortest rigid orientation arc at <=5 degree increments, then
// bisect the first detected entry. This is bounded swept sampling, not exact
// CCD for arbitrary nonlinear bone motion between observed frames.
inline SweptGroundContact SweepRigidPath(const std::vector<Vec3>& points,
    RigidPose from,RigidPose to,float height,float radius)
{
    SweptGroundContact result; result.pose=to;
    if (points.empty()) return result;
    const float limit=height+std::max(0.f,radius);
    float previousHeight=MinimumHeight(points,from);
    if (previousHeight<limit-1e-6f) {
        result.hit=result.startsBelow=true; result.fraction=0; result.pose=from; return result;
    }
    float d=std::abs(from.orientation.x*to.orientation.x+from.orientation.y*to.orientation.y+
                     from.orientation.z*to.orientation.z+from.orientation.w*to.orientation.w);
    const float angle=2*std::acos(std::clamp(d,0.f,1.f));
    const int steps=std::clamp(static_cast<int>(std::ceil(angle/.087266463f)),1,64);
    float previous=0;
    for (int step=1;step<=steps;++step)
    {
        const float t=static_cast<float>(step)/steps;
        const float y=MinimumHeight(points,Interpolate(from,to,t));
        if (y<limit-1e-6f || (previousHeight>limit+1e-6f && y<=limit+1e-6f))
        {
            float lo=previous,hi=t;
            for (int iteration=0;iteration<14;++iteration) {
                const float mid=(lo+hi)*.5f;
                if (MinimumHeight(points,Interpolate(from,to,mid))<limit) hi=mid;
                else lo=mid;
            }
            result.hit=true; result.fraction=lo; result.pose=Interpolate(from,to,lo);
            return result;
        }
        previous=t; previousHeight=y;
    }
    return result;
}
} // namespace icicle_growth
