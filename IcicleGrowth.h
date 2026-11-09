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
inline bool Finite(Vec3 v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}
// Use double mass fractions before converting to float.  Never cast 1/mass
// to float: tiny nonzero masses can otherwise produce INF and poison velocity.
inline Vec3 MassWeightedMean(Vec3 oldMean, double oldMass, Vec3 incoming, double addedMass)
{
    oldMass=std::max(0.0,oldMass);
    addedMass=std::max(0.0,addedMass);
    if (!Finite(incoming)) incoming=Finite(oldMean)?oldMean:Vec3{};
    if (!Finite(oldMean) || oldMass<=0.0) return incoming;
    const double total=oldMass+addedMass;
    if (!(total>0.0) || !std::isfinite(total)) return oldMean;
    const float fraction=static_cast<float>(addedMass/total);
    return oldMean*(1.f-fraction)+incoming*fraction;
}
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
        for (std::size_t k=0;k<3;++k)
        {
            const auto i=roots[r].indices[k];
            if (i>=samples.size()) continue;
            const double mass=Positive(samples[i].dripMass)*Positive(roots[r].weights[k])/
                              std::max(1.0,requests[i]);
            const Vec3 relative=i<relativeVelocities.size()?relativeVelocities[i]:Vec3{};
            result[r].velocityWorld=MassWeightedMean(result[r].velocityWorld,result[r].mass,
                samples[i].surfaceVelocity+relative,mass);
            result[r].mass+=mass;
        }
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
            const Vec3 edgeVelocity=(s[i].position-s[donor].position)*rate;
            incoming=MassWeightedMean(incoming,incomingFlux,edgeVelocity,flux);
            incomingFlux+=flux;
        }
        if (incomingFlux>0) result[i]=incoming;
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
        result[nearest].velocityWorld=MassWeightedMean(
            result[nearest].velocityWorld,result[nearest].mass,
            samples[i].surfaceVelocity+relative,mass);
        result[nearest].mass+=mass;
    }
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
    double detachedMass=0;           // liquid leaving the tip area in this update
    float minimumDirectionCoherence=1.f; // |mass-weighted unit-direction mean|, 0..1
    unsigned unstableDirectionSteps=0;   // low-coherence direction held rather than normalized
};
inline void MixLiquid(double& mass, Vec3& velocity, double added, Vec3 incoming)
{
    added=Positive(added);
    if (added<=0) return;
    velocity=MassWeightedMean(velocity,mass,incoming,added);
    mass+=added;
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

// Lumped transit -> tip liquid -> freezing. The temperature control maps to a rate; no heat equation or fluid particles.
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
    // Do not render a turn/window as a microscopic zigzag.  Keep frozen
    // mass in GrowthBuffer until a resolvable segment can be emitted.
    float minimumEmittedLength=.0025f; // model units (25% of max by default)
    // Artistic curvature constraint for NEW segments, degrees per max length.
    // This does not modify older frozen segments or the mass-length relation.
    float maxTurnDegreesPerMaxLength=40.f;
};

// Optional numerical response model for NEW growth only. This is a tunable
// approximation, not an experimentally measured freezing response time.
// Filter after conversion into the attachment's local frame. Filtering world
// directions and then attaching them again would mix carrier rotation twice.
struct DirectionResponseParameters
{
    bool enabled=false; // minimal model: liquid relaxation (1/surfaceCouplingRate) is the only direction lag
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
    const bool reliable=std::isfinite(relativeSpeed) &&
        relativeSpeed>std::max(1e-8f,p.minimumRelativeSpeed) &&
        Finite(rawLocal) && Length(rawLocal)>1e-8f;
    // OFF means no temporal smoothing, NOT permission to use a nearly-zero
    // or nonfinite velocity as a fresh direction.
    if (!p.enabled) {
        if (!state.initialized) {
            state.directionLocal=reliable?target:Unit(fallbackLocal);
            state.initialized=true;
        } else if (reliable) state.directionLocal=target;
        return state.directionLocal;
    }
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

// Rigid-carrier velocity is measured by differencing a pose that the animation
// system interpolates between keys. That makes velocity piecewise constant and
// the acceleration (g_eff = g - a) a spike train at the key rate. A moving
// average over exactly ONE key interval has a spectral null at the key rate and
// its harmonics, so it removes this sampling artefact without a tuned time
// constant. Input must already be the growth-free per-tick rigid velocity.
struct CarrierVelocityFilter
{
    std::vector<Vec3> ring;
    std::size_t head=0, count=0;
    void Reset() { ring.clear(); head=count=0; }
    // At a looping animation seam the attachment frame can rotate abruptly.
    // Reorient the world-space velocity history without discarding it; the seam
    // itself is not a measured acceleration or a physical growth reset.
    template<class Transform>
    void TransformSamples(Transform transform)
    {
        for (auto& v:ring) v=transform(v);
    }
    Vec3 Push(Vec3 velocity, std::size_t window)
    {
        window=std::max<std::size_t>(1,std::min<std::size_t>(window,64));
        if (ring.size()!=window) { ring.assign(window,Vec3{}); head=count=0; }
        ring[head]=velocity; head=(head+1)%window; count=std::min(count+1,window);
        Vec3 sum; // recomputed every call: no accumulated drift
        for (std::size_t i=0;i<count;++i) sum+=ring[i];
        return sum/static_cast<float>(count);
    }
};
// Number of physics ticks in one animation key interval (>=1). If the tick is
// longer than the key interval the stair cannot be removed; the caller should
// then lower the physics step.
inline std::size_t CarrierWindowTicks(float ticksPerSecond,float motionSpeed,float dt)
{
    const double interval=1.0/(std::max(1e-3,static_cast<double>(ticksPerSecond))*
                               std::max(1e-3,static_cast<double>(motionSpeed)));
    return static_cast<std::size_t>(std::clamp(std::llround(interval/std::max(1e-6,static_cast<double>(dt))),1LL,64LL));
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
    Vec3 pendingReference{0,-1,0};
    Vec3 lastEmittedDirection{0,-1,0};
    bool hasEmitted=false;
    void ClearPending() { mass=0; startTime=lastTime=0; moment={}; pendingReference={0,-1,0}; }
    void Clear() { ClearPending(); lastEmittedDirection={0,-1,0}; hasEmitted=false; }
};
inline void FlushGrowthBuffer(GrowthBuffer& buffer,std::vector<GrowthChunk>& chunks)
{
    if (buffer.mass>1e-15) {
        const Vec3 mean=buffer.moment/static_cast<float>(buffer.mass);
        // Opposing tiny samples can cancel. The normalized sum then has no
        // physically meaningful heading; keep the last emitted direction.
        const Vec3 fallback=buffer.hasEmitted?buffer.lastEmittedDirection:buffer.pendingReference;
        const Vec3 heading=Finite(mean) && Length(mean)>=0.15f
            ? Unit(mean,fallback) : fallback;
        chunks.push_back({buffer.mass,heading,buffer.lastTime});
        buffer.lastEmittedDirection=heading;
        buffer.hasEmitted=true;
    }
    buffer.ClearPending();
}
// Turn a new segment toward the requested heading by at most maxAngle.
// Unlike a time filter this bounds local curvature per UNIT OF ICE GROWTH.
inline Vec3 LimitDirectionTurn(Vec3 from,Vec3 desired,float maxAngle)
{
    from=Unit(from); desired=Unit(desired,from);
    maxAngle=std::clamp(maxAngle,0.f,3.14159265f);
    const float cosine=std::clamp(Dot(from,desired),-1.f,1.f);
    const float angle=std::acos(cosine);
    if (angle<=maxAngle+1e-6f) return desired;
    Vec3 tangent=desired-from*cosine;
    if (Length(tangent)<1e-6f) {
        const Vec3 axis=std::abs(from.x)<.8f?Vec3{1,0,0}:Vec3{0,1,0};
        tangent=axis-from*Dot(axis,from);
    }
    tangent=Unit(tangent);
    return Unit(from*std::cos(maxAngle)+tangent*std::sin(maxAngle),from);
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
    const double minMass=std::min(maxMass, resolution.linearMassDensity*
        std::max(0.f,resolution.minimumEmittedLength));
    const float angle=std::clamp(resolution.maxTurnDegrees,0.01f,179.f)*.01745329252f;
    const double window=std::max(1e-5f,resolution.maxWindowSeconds);
    const double start=time-std::max(0.f,duration);
    // Flush the old direction BEFORE adding a differently directed sample.
    // Opposite phases are never averaged into a near-zero vector.
    if (buffer.mass>=minMass && buffer.mass>1e-15 &&
        (Dot(Unit(buffer.moment/static_cast<float>(buffer.mass),
                  buffer.hasEmitted?buffer.lastEmittedDirection:direction),direction)<std::cos(angle) ||
         start-buffer.startTime>=window))
        FlushGrowthBuffer(buffer,chunks);
    double remaining=mass;
    while (remaining>1e-15)
    {
        if (buffer.mass<=0) {
            buffer.startTime=start+(time-start)*(1-remaining/mass);
            buffer.pendingReference=direction;
        }
        const double capacity=maxMass-buffer.mass;
        if (capacity<=1e-15) { FlushGrowthBuffer(buffer,chunks); continue; }
        const double take=std::min(remaining,capacity);
        buffer.mass+=take;
        buffer.moment+=direction*static_cast<float>(take);
        remaining-=take;
        buffer.lastTime=start+(time-start)*(1-remaining/mass);
        if (buffer.mass>=maxMass-1e-15 ||
            (buffer.mass>=minMass && buffer.lastTime-buffer.startTime>=window))
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
// Preserve a vector's attachment-local components across an animation seam.
inline Vec3 RotateBetweenFrames(Vec3 worldVector,Quaternion from,Quaternion to)
{
    return Rotate(to,Rotate(Conjugate(from),worldVector));
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
    return newCarrier+RotateBetweenFrames(
        velocity-oldCarrier,oldOrientation,newOrientation);
}
// Motion-driven growth direction: integrate liquid velocity in the attachment's
// rotating frame. This is a controllable reduced-order model, not SPH/CFD.
// For an attachment-local point r, its non-inertial acceleration includes
// carrier translation, angular acceleration, centrifugal and Coriolis terms.
// Integrating a persistent velocity (rather than filtering normalized per-frame
// vectors) carries the same growth history through repeated spinning kicks.
struct MotionGrowthParameters
{
    float turnScale = 0.20f; // art control: growth turns / carrier turns
    // Swirl/gravity ratio follows the rotating-frame balance tan(tilt)=gain*w^2*R/g:
    // R is the distance from the carrier's vertical axis to the growing tip, in the
    // same length unit as gravityAcceleration (0.25 = 25 cm if g=9.81 is in m/s^2).
    float leverArm = 0.15f;
    float maximumSwirlRatio = 0.8f; // cap on horizontal/vertical; 0.8 -> 39 deg from down
    float relaxationSeconds = 0.35f;       // liquid motion memory; NOT freeze time
    float accelerationFilterSeconds = 0.10f; // suppress FBX finite-difference spikes
    float inletFlowBlend = 0.20f;         // supplied film velocity influences tip liquid
    float maximumCarrierAcceleration = 40.f; // model units/s^2; stability bound
    float maximumAngularSpeed = 18.f;       // rad/s, diagnostic clamp
    float maximumAngularAcceleration = 80.f; // rad/s^2
};
struct MotionGrowthState
{
    bool steeringInitialized=false;
    Vec3 steeringAxis{0,1,0}, steeringTangent{1,0,0};
    float filteredSpin=0;
    double accumulatedTurn=0;
    bool initialized = false;
    Vec3 localVelocity;
    Vec3 previousCarrierVelocityWorld;
    Vec3 previousAngularVelocityLocal;
    Vec3 filteredCarrierAccelerationWorld;
    Vec3 filteredAngularAccelerationLocal;
    void Reset() { *this = MotionGrowthState{}; }
    // At an FBX loop boundary, keep the already accumulated LIQUID velocity
    // in attachment-local coordinates but ignore unobserved jump kinematics.
    void ResetKinematics()
    {
        initialized=false;
        previousCarrierVelocityWorld={};
        previousAngularVelocityLocal={};
        filteredCarrierAccelerationWorld={};
        filteredAngularAccelerationLocal={};
    }
};
inline Vec3 LimitVector(Vec3 v,float maximum)
{
    const float magnitude=Length(v);
    return magnitude>maximum && magnitude>1e-8f ? v*(maximum/magnitude) : v;
}
// Returns the direction for NEW ice, in the attachment's local coordinates.
// Carrier velocity is the velocity of the ROOT, not of the growing tip;
// rotational terms account for the local offset between root and tip.
inline Vec3 AdvanceInertialGrowthDirection(
    MotionGrowthState& state, Vec3 gravityWorld, Vec3 carrierVelocityWorld,
    Quaternion previousOrientation, Quaternion currentOrientation, Vec3 tipLocal,
    Vec3 incomingVelocityWorld, bool hasInlet, float dt, float motionGain,
    const MotionGrowthParameters& p)
{
    const Vec3 gLocal=Rotate(Conjugate(currentOrientation),gravityWorld);
    const Vec3 downLocal=Unit(gLocal);
    if (!(dt>0.f) || !std::isfinite(dt)) return Unit(state.localVelocity,downLocal);
    const float tau=std::max(0.01f,p.relaxationSeconds);
    const float inverseTau=1.f/tau;
    const Vec3 omegaWorld=AngularVelocity(previousOrientation,currentOrientation,dt);
    const Vec3 omegaLocal=LimitVector(Rotate(Conjugate(currentOrientation),omegaWorld),
                                      std::max(0.01f,p.maximumAngularSpeed));
    if (!state.initialized)
    {
        // Retain localVelocity across a seam; only initialize it at a new root.
        if (Length(state.localVelocity)<1e-8f) state.localVelocity=gLocal*tau;
        state.previousCarrierVelocityWorld=carrierVelocityWorld;
        state.previousAngularVelocityLocal=omegaLocal;
        state.initialized=true;
    }
    const Vec3 rootAccel=LimitVector(
        (carrierVelocityWorld-state.previousCarrierVelocityWorld)/dt,
        std::max(0.01f,p.maximumCarrierAcceleration));
    const Vec3 angularAccel=LimitVector(
        (omegaLocal-state.previousAngularVelocityLocal)/dt,
        std::max(0.01f,p.maximumAngularAcceleration));
    state.previousCarrierVelocityWorld=carrierVelocityWorld;
    state.previousAngularVelocityLocal=omegaLocal;
    const float filterTau=std::max(1e-4f,p.accelerationFilterSeconds);
    const float a=-std::expm1(-dt/filterTau);
    state.filteredCarrierAccelerationWorld+=
        (rootAccel-state.filteredCarrierAccelerationWorld)*a;
    state.filteredAngularAccelerationLocal+=
        (angularAccel-state.filteredAngularAccelerationLocal)*a;
    const Vec3 rootAccelLocal=Rotate(Conjugate(currentOrientation),
                                      state.filteredCarrierAccelerationWorld);
    const Vec3 rotationalAcceleration=
        Cross(state.filteredAngularAccelerationLocal,tipLocal)+
        Cross(omegaLocal,Cross(omegaLocal,tipLocal));
    const Vec3 coriolis=Cross(omegaLocal,state.localVelocity)*2.f;
    // Motion gain is an artist-facing multiplier, not a physical force unit.
    // With gain=0 only gravity is integrated (still expressed in the local frame).
    const Vec3 effectiveAcceleration=gLocal-
        (rootAccelLocal+rotationalAcceleration+coriolis)*std::max(0.f,motionGain);
    Vec3 targetVelocity=effectiveAcceleration*tau;
    if (hasInlet)
    {
        // Surface-flow inlet velocity is computed in world space by the
        // existing water transport graph; subtract root motion once.
        const Vec3 relativeInlet=Rotate(Conjugate(currentOrientation),
                                        incomingVelocityWorld-carrierVelocityWorld);
        targetVelocity+=relativeInlet*std::max(0.f,p.inletFlowBlend)*
                        std::max(0.f,motionGain);
    }
    const float follow=-std::expm1(-inverseTau*dt);
    state.localVelocity+=(targetVelocity-state.localVelocity)*follow;
    return Unit(state.localVelocity,downLocal);
}

// PROCEDURAL, motion-driven steering (reduced-order, NOT a fluid force law).
// Two ingredients, both explicit:
//  * gravity: "down" is the CURRENT world gravity expressed in the attachment
//    frame every tick (it is never frozen at spawn), so a tilting limb still
//    grows toward world-down;
//  * swirl: carrier spin about world-up is integrated into a horizontal tangent
//    (signed, accumulated, persistent). Its weight relative to gravity is the
//    rotating-frame ratio gain*w^2*R/g, so a body that is not spinning grows
//    straight down and the curve appears only while it spins.
inline Vec3 AdvanceMotionGrowthDirection(
    MotionGrowthState& state, Vec3 gravityWorld, Vec3 carrierVelocityWorld,
    Quaternion previousOrientation, Quaternion currentOrientation, Vec3 tipLocal,
    Vec3 incomingVelocityWorld, bool hasInlet, float dt, float motionGain,
    const MotionGrowthParameters& p)
{
    (void)carrierVelocityWorld; (void)tipLocal; (void)incomingVelocityWorld; (void)hasInlet;
    const float gravityMagnitude=std::max(1e-4f,Length(gravityWorld));
    const Vec3 up=Unit(gravityWorld)*-1.f;
    const Vec3 axis=Unit(Rotate(Conjugate(currentOrientation),up),Vec3{0,1,0});
    const auto perpendicular=[&](Vec3 a) {
        Vec3 r=Vec3{0,0,1}-a*Dot(Vec3{0,0,1},a);
        if (Length(r)<1e-5f) r=Vec3{1,0,0}-a*Dot(Vec3{1,0,0},a);
        return Unit(Cross(a,Unit(r)),Vec3{1,0,0});
    };
    if (!state.steeringInitialized) {
        state.steeringTangent=perpendicular(axis);
        state.steeringInitialized=true;
    } else {
        // Keep the swirl horizontal in the WORLD when the attachment tilts.
        const Vec3 t=state.steeringTangent-axis*Dot(state.steeringTangent,axis);
        state.steeringTangent=Length(t)>1e-5f ? Unit(t) : perpendicular(axis);
    }
    state.steeringAxis=axis;
    if (dt>0 && std::isfinite(dt)) {
        const float raw=std::clamp(Dot(AngularVelocity(previousOrientation,currentOrientation,dt),up),
                                   -p.maximumAngularSpeed,p.maximumAngularSpeed);
        const float tau=std::max(.001f,p.relaxationSeconds);
        const float decay=std::exp(-dt/tau);
        // Exact integral of a first-order spin response for constant input.
        const float integrated=raw*dt+(state.filteredSpin-raw)*tau*(1-decay);
        state.filteredSpin=raw+(state.filteredSpin-raw)*decay;
        const float angle=integrated*std::max(0.f,motionGain)*p.turnScale;
        state.accumulatedTurn+=angle;
        state.steeringTangent=Unit(Rotate(RotationVector(axis*angle),state.steeringTangent),
                                   state.steeringTangent);
    }
    const float spin=state.filteredSpin;
    const float swirl=std::min(std::max(0.f,p.maximumSwirlRatio),
        std::max(0.f,motionGain)*spin*spin*std::max(0.f,p.leverArm)/gravityMagnitude);
    state.localVelocity=Unit(axis*-1.f+state.steeringTangent*swirl,axis*-1.f);
    return state.localVelocity;
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

// Integrated mesh collision support; no separate SurfaceClearance.h required.
namespace icicle_growth {

struct SurfaceHit
{
    bool found=false;
    float distance=0, signedDistance=0;
    Vec3 point, normal;
};

inline Vec3 ClosestOnTriangle(Vec3 p,Vec3 a,Vec3 b,Vec3 c,float& u,float& v,float& w)
{
    const Vec3 ab=b-a, ac=c-a, ap=p-a;
    const float d1=Dot(ab,ap), d2=Dot(ac,ap);
    if (d1<=0&&d2<=0) { u=1;v=0;w=0; return a; }
    const Vec3 bp=p-b; const float d3=Dot(ab,bp), d4=Dot(ac,bp);
    if (d3>=0&&d4<=d3) { u=0;v=1;w=0; return b; }
    const float vc=d1*d4-d3*d2;
    if (vc<=0&&d1>=0&&d3<=0) { const float t=d1/(d1-d3); u=1-t;v=t;w=0; return a+ab*t; }
    const Vec3 cp=p-c; const float d5=Dot(ab,cp), d6=Dot(ac,cp);
    if (d6>=0&&d5<=d6) { u=0;v=0;w=1; return c; }
    const float vb=d5*d2-d1*d6;
    if (vb<=0&&d2>=0&&d6<=0) { const float t=d2/(d2-d6); u=1-t;v=0;w=t; return a+ac*t; }
    const float va=d3*d6-d5*d4;
    if (va<=0&&(d4-d3)>=0&&(d5-d6)>=0)
    { const float t=(d4-d3)/((d4-d3)+(d5-d6)); u=0;v=1-t;w=t; return b+(c-b)*t; }
    const float inv=1.f/(va+vb+vc); const float vv=vb*inv, ww=vc*inv;
    u=1-vv-ww; v=vv; w=ww; return a+ab*vv+ac*ww;
}

class SurfaceGrid
{
public:
    bool Valid() const { return valid_; }
    // Positions/normals are per vertex (world space); indices are triangle triples.
    void Build(const std::vector<Vec3>& positions,const std::vector<Vec3>& normals,
               const std::vector<unsigned int>& indices,float cellSize)
    {
        valid_=false; pos_=&positions; nrm_=&normals; idx_=&indices;
        if (positions.empty()||normals.size()!=positions.size()||indices.size()<3) return;
        Vec3 lo=positions[0], hi=positions[0];
        for (const auto& p:positions) {
            lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
            hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
        }
        cell_=std::max(1e-5f,cellSize); lo_=lo-Vec3(cell_);
        for (;;) {
            const Vec3 e=hi-lo_+Vec3(cell_);
            nx_=std::max(1,int(e.x/cell_)+1); ny_=std::max(1,int(e.y/cell_)+1); nz_=std::max(1,int(e.z/cell_)+1);
            if (std::size_t(nx_)*ny_*nz_<=(1u<<21)) break;
            cell_*=1.5f;
        }
        const std::size_t cells=std::size_t(nx_)*ny_*nz_, tris=indices.size()/3;
        start_.assign(cells+1,0); list_.clear();
        const auto range=[&](std::size_t t,int r[6]) {
            const Vec3 a=positions[indices[3*t]], b=positions[indices[3*t+1]], c=positions[indices[3*t+2]];
            const Vec3 mn={std::min({a.x,b.x,c.x}),std::min({a.y,b.y,c.y}),std::min({a.z,b.z,c.z})};
            const Vec3 mx={std::max({a.x,b.x,c.x}),std::max({a.y,b.y,c.y}),std::max({a.z,b.z,c.z})};
            r[0]=Clamp((int)std::floor((mn.x-lo_.x)/cell_),nx_); r[1]=Clamp((int)std::floor((mn.y-lo_.y)/cell_),ny_);
            r[2]=Clamp((int)std::floor((mn.z-lo_.z)/cell_),nz_); r[3]=Clamp((int)std::floor((mx.x-lo_.x)/cell_),nx_);
            r[4]=Clamp((int)std::floor((mx.y-lo_.y)/cell_),ny_); r[5]=Clamp((int)std::floor((mx.z-lo_.z)/cell_),nz_);
        };
        int r[6];
        for (std::size_t t=0;t<tris;++t) { range(t,r);
            for (int z=r[2];z<=r[5];++z) for (int y=r[1];y<=r[4];++y) for (int x=r[0];x<=r[3];++x) ++start_[Cell(x,y,z)+1]; }
        for (std::size_t i=0;i<cells;++i) start_[i+1]+=start_[i];
        list_.assign(start_[cells],0); std::vector<unsigned> fill(start_.begin(),start_.end()-1);
        for (std::size_t t=0;t<tris;++t) { range(t,r);
            for (int z=r[2];z<=r[5];++z) for (int y=r[1];y<=r[4];++y) for (int x=r[0];x<=r[3];++x) list_[fill[Cell(x,y,z)]++]=(unsigned)t; }
        valid_=true;
    }
    SurfaceHit Nearest(Vec3 p,float maxDistance) const
    {
        SurfaceHit best; if (!valid_) return best;
        float bestSq=maxDistance*maxDistance;
        const int x0=Clamp((int)std::floor((p.x-maxDistance-lo_.x)/cell_),nx_), x1=Clamp((int)std::floor((p.x+maxDistance-lo_.x)/cell_),nx_);
        const int y0=Clamp((int)std::floor((p.y-maxDistance-lo_.y)/cell_),ny_), y1=Clamp((int)std::floor((p.y+maxDistance-lo_.y)/cell_),ny_);
        const int z0=Clamp((int)std::floor((p.z-maxDistance-lo_.z)/cell_),nz_), z1=Clamp((int)std::floor((p.z+maxDistance-lo_.z)/cell_),nz_);
        float bu=0,bv=0,bw=0; std::size_t bt=0;
        for (int z=z0;z<=z1;++z) for (int y=y0;y<=y1;++y) for (int x=x0;x<=x1;++x) {
            const std::size_t c=Cell(x,y,z);
            for (unsigned k=start_[c];k<start_[c+1];++k) {
                const std::size_t t=list_[k]; float u,v,w;
                const Vec3 q=ClosestOnTriangle(p,(*pos_)[(*idx_)[3*t]],(*pos_)[(*idx_)[3*t+1]],(*pos_)[(*idx_)[3*t+2]],u,v,w);
                const Vec3 d=p-q; const float sq=Dot(d,d);
                if (sq<bestSq) { bestSq=sq; best.found=true; best.point=q; bu=u;bv=v;bw=w;bt=t; }
            }
        }
        if (!best.found) return best;
        const Vec3 n=(*nrm_)[(*idx_)[3*bt]]*bu+(*nrm_)[(*idx_)[3*bt+1]]*bv+(*nrm_)[(*idx_)[3*bt+2]]*bw;
        best.normal=Unit(n,Vec3{0,1,0}); best.distance=std::sqrt(bestSq);
        best.signedDistance=Dot(p-best.point,best.normal);
        return best;
    }

    // Exact centreline/triangle intersection over grid candidates, two-sided.
    // Clearance is a small backoff along the segment, not a tube-radius test.
    GroundClip ClipSegment(Vec3 from, Vec3 to, float backoff=1e-5f) const
    {
        GroundClip result; result.endpoint=to;
        if (!valid_) { result.hit=true; result.fraction=0; result.endpoint=from; return result; }
        const Vec3 d=to-from; const float len=Length(d);
        if (!(len>1e-9f)) return result;
        const auto near=Nearest(from, std::max(backoff*4,1e-4f));
        if (near.found && near.signedDistance < backoff && Dot(d,near.normal)<-1e-9f) {
            result.hit=true; result.fraction=0; result.endpoint=from; return result;
        }
        const Vec3 lo{std::min(from.x,to.x),std::min(from.y,to.y),std::min(from.z,to.z)};
        const Vec3 hi{std::max(from.x,to.x),std::max(from.y,to.y),std::max(from.z,to.z)};
        const int x0=Clamp(int(std::floor((lo.x-lo_.x)/cell_)),nx_), x1=Clamp(int(std::floor((hi.x-lo_.x)/cell_)),nx_);
        const int y0=Clamp(int(std::floor((lo.y-lo_.y)/cell_)),ny_), y1=Clamp(int(std::floor((hi.y-lo_.y)/cell_)),ny_);
        const int z0=Clamp(int(std::floor((lo.z-lo_.z)/cell_)),nz_), z1=Clamp(int(std::floor((hi.z-lo_.z)/cell_)),nz_);
        float first=1.f;
        for(int z=z0;z<=z1;++z) for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
            const auto cell=Cell(x,y,z);
            for(unsigned k=start_[cell];k<start_[cell+1];++k) {
                const auto t=list_[k];
                const Vec3 a=(*pos_)[(*idx_)[3*t]], b=(*pos_)[(*idx_)[3*t+1]], c=(*pos_)[(*idx_)[3*t+2]];
                const Vec3 e1=b-a,e2=c-a,p=Cross(d,e2);
                const float det=Dot(e1,p);
                if(std::abs(det)<1e-8f*std::max(1e-12f,len*Length(e1)*Length(e2))) continue;
                const float inv=1.f/det; const Vec3 v=from-a;
                const float u=Dot(v,p)*inv; if(u<0 || u>1) continue;
                const Vec3 q=Cross(v,e1); const float w=Dot(d,q)*inv;
                if(w<0 || u+w>1) continue;
                const float f=Dot(e2,q)*inv;
                // Ignore the attachment's own surface only at the start.
                if(f>1e-6f && f<=first) { first=f; result.hit=true; }
            }
        }
        if(result.hit) { result.fraction=std::max(0.f,first-backoff/len); result.endpoint=from+d*result.fraction; }
        return result;
    }
private:
    static int Clamp(int v,int n) { return std::max(0,std::min(v,n-1)); }
    std::size_t Cell(int x,int y,int z) const { return (std::size_t(z)*ny_+y)*nx_+x; }
    bool valid_=false; float cell_=1; Vec3 lo_;
    int nx_=1,ny_=1,nz_=1; std::vector<unsigned> start_,list_;
    const std::vector<Vec3>* pos_=nullptr; const std::vector<Vec3>* nrm_=nullptr; const std::vector<unsigned>* idx_=nullptr;
};


} // namespace icicle_growth

namespace icicle_growth {

// Reduced tip-liquid cohorts. offsetWorld is relative to the moving tip;
// integrating v - carrier avoids treating newly emitted solid length as motion.
struct LiquidParcel { Vec3 offsetWorld, velocityWorld; double mass=0; float age=0; };
struct ParcelState {
    std::vector<LiquidParcel> parcels;
    void Reset() { parcels.clear(); }
    void Rebase(Vec3 oldCarrier,Vec3 newCarrier,Quaternion oldQ,Quaternion newQ) {
        for(auto& p:parcels) {
            p.offsetWorld=RotateBetweenFrames(p.offsetWorld,oldQ,newQ);
            p.velocityWorld=RebaseCarrierVelocity(p.velocityWorld,oldCarrier,newCarrier,oldQ,newQ);
        }
    }
};
struct ParcelParameters {
    float couplingSeconds=.35f; // velocity relaxation toward the moving tip
    float detachDistance=.15f; // model length units; uncalibrated artist setting
    float minimumDirectionOffset=1e-4f; // below this, heading of a parcel is unreliable
    float minimumDirectionCoherence=.15f; // prevent normalization of cancelling cohorts
    unsigned maximumParcels=128;
};
inline StepResult AdvanceParcels(ParcelState& cloud,LiquidState& state,const Supply& supply,
    float dt,float pathLength,Vec3 carrier,Vec3 gravityDirection,float freezeRate,
    const Parameters& p,const ParcelParameters& settings)
{
    StepResult result; if(!(dt>0) || !std::isfinite(dt)) return result;
    const Vec3 down=Unit(gravityDirection), gravity=down*float(Positive(p.gravityAcceleration));
    const int count=std::max(1,int(std::ceil(dt/std::max(1e-4f,p.maxSubstep))));
    const float h=dt/count, k=1.f/std::max(.001f,settings.couplingSeconds);
    const float decay=std::exp(-k*h), integral=(1-decay)/k;
    const double rate=Positive(freezeRate), runoff=Positive(p.tipDripRate);
    const double travel=Positive(p.travelSpeed)/std::max(1e-5,std::max(Positive(pathLength),Positive(p.minimumTravelLength)));
    for(int i=0;i<count;++i) {
        const double input=Positive(supply.mass)/count;
        MixLiquid(state.transitMass,state.transitVelocityWorld,input,supply.velocityWorld);
        state.totalInput+=input;
        state.transitVelocityWorld=carrier+(state.transitVelocityWorld-carrier)*decay+gravity*integral;
        const double delivered=state.transitMass*(-std::expm1(-travel*h));
        state.transitMass-=delivered;
        if(delivered>0) cloud.parcels.push_back({{},state.transitVelocityWorld,delivered,0});
        // Merge oldest cohorts only when bounded storage is exhausted; conserve mass/momentum.
        while(cloud.parcels.size()>std::max(2u,settings.maximumParcels)) {
            auto& a=cloud.parcels[0]; const auto b=cloud.parcels[1]; const double total=a.mass+b.mass;
            if(total>0) { const float f=float(b.mass/total); a.offsetWorld=a.offsetWorld*(1-f)+b.offsetWorld*f;
                a.velocityWorld=a.velocityWorld*(1-f)+b.velocityWorld*f; a.age=a.age*(1-f)+b.age*f; }
            a.mass=total; cloud.parcels.erase(cloud.parcels.begin()+1);
        }
        // Form one MASS-NORMALIZED direction average per substep. This is
        // numerically stable even for 1e-39 mass, unlike float(mass) moments.
        double substepMass=0;
        Vec3 substepMean{};
        float substepMeanSpeed=0;
        for(auto& parcel:cloud.parcels) {
            const Vec3 relative=parcel.velocityWorld-carrier, terminal=gravity*(1/k);
            parcel.offsetWorld+=(relative-terminal)*integral+terminal*h;
            parcel.velocityWorld=carrier+relative*decay+gravity*integral;
            parcel.age+=h;
            if(!Finite(parcel.offsetWorld) || !Finite(parcel.velocityWorld) ||
               Length(parcel.offsetWorld)>std::max(1e-5f,settings.detachDistance)) {
                result.detachedMass+=parcel.mass;
                state.totalDripped+=parcel.mass; parcel.mass=0; continue;
            }
            const double removed=parcel.mass*(-std::expm1(-(rate+runoff)*h));
            const double frozen=(rate+runoff)>0?removed*rate/(rate+runoff):0;
            parcel.mass-=removed; state.totalFrozen+=frozen; state.totalDripped+=removed-frozen;
            if(frozen<=0) continue;
            const float magnitude=Length(parcel.offsetWorld);
            const Vec3 base=down*magnitude;
            const Vec3 mixed=base+(parcel.offsetWorld-base)*float(Positive(p.flowMotionGain));
            // A parcel has no usable steering direction until its offset
            // from the moving tip is long enough to resolve numerically.
            const Vec3 direction=magnitude>=settings.minimumDirectionOffset &&
                Finite(mixed) && Length(mixed)>=settings.minimumDirectionOffset*.1f
                ? Unit(mixed,state.lastDirection) : state.lastDirection;
            substepMean=MassWeightedMean(substepMean,substepMass,direction,frozen);
            substepMeanSpeed=static_cast<float>(
                (static_cast<double>(substepMeanSpeed)*substepMass+
                 static_cast<double>(Length(parcel.velocityWorld-carrier))*frozen)/
                 (substepMass+frozen));
            substepMass+=frozen;
        }
        if(substepMass>0) {
            const float coherence=Finite(substepMean)?Length(substepMean):0.f;
            result.minimumDirectionCoherence=std::min(result.minimumDirectionCoherence,coherence);
            const bool reliable=coherence>=std::max(0.f,settings.minimumDirectionCoherence);
            if(!reliable) ++result.unstableDirectionSteps;
            const Vec3 direction=reliable?Unit(substepMean,state.lastDirection):
                Unit(state.lastDirection,down);
            state.lastDirection=direction;
            result.frozenMass+=substepMass;
            result.frozenDirectionMoment+=direction*static_cast<float>(substepMass);
            result.samples.push_back({substepMass,direction,(i+1)*h,h,substepMeanSpeed});
        }
        cloud.parcels.erase(std::remove_if(cloud.parcels.begin(),cloud.parcels.end(),[](const LiquidParcel& q){return q.mass<=0;}),cloud.parcels.end());
    }
    state.tipMass=0;
    Vec3 meanVelocity=carrier;
    for(const auto& q:cloud.parcels) {
        if(q.mass<=0) continue;
        meanVelocity=MassWeightedMean(meanVelocity,state.tipMass,q.velocityWorld,q.mass);
        state.tipMass+=q.mass;
    }
    // Keeping an almost empty reservoir does not justify dividing by 1e-39.
    state.tipVelocityWorld=Finite(meanVelocity)?meanVelocity:carrier;
    return result;
}

}
