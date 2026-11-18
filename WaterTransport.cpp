#include "WaterTransport.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

using namespace std;
using namespace glm;

namespace
{
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kEps = 1e-7f;
}

WaterTransportSimulation::WaterTransportSimulation(
    size_t vertexCount,
    const vector<unsigned int>& triangleIndices)
    : vertexCount_(vertexCount),
    triangleIndices_(triangleIndices),
    adjacency_(vertexCount),
    transitionWeights_(vertexCount),
    vertices_(vertexCount),
    previousPositions_(vertexCount, vec3(0.0f)),
    rainInput_(vertexCount, 0.0f),
    nextWater_(vertexCount, 0.0f),
    nextIce_(vertexCount, 0.0f)
{
    BuildAdjacency();
}

vec3 WaterTransportSimulation::NormalizeOrZero(const vec3& v)
{
    const float len2 = dot(v, v);

    if (len2 <= 1e-12f)
        return vec3(0.0f);

    return v / std::sqrt(len2);
}

vec3 WaterTransportSimulation::ProjectToTangent(
    const vec3& v,
    const vec3& n)
{
    return v - dot(v, n) * n;
}

void WaterTransportSimulation::BuildAdjacency()
{
    vector<unordered_set<unsigned int>> neighborSets(vertexCount_);

    for (size_t t = 0; t + 2 < triangleIndices_.size(); t += 3)
    {
        const unsigned int a = triangleIndices_[t + 0];
        const unsigned int b = triangleIndices_[t + 1];
        const unsigned int c = triangleIndices_[t + 2];

        if (a >= vertexCount_ ||
            b >= vertexCount_ ||
            c >= vertexCount_)
        {
            continue;
        }

        neighborSets[a].insert(b);
        neighborSets[a].insert(c);

        neighborSets[b].insert(a);
        neighborSets[b].insert(c);

        neighborSets[c].insert(a);
        neighborSets[c].insert(b);
    }

    for (size_t i = 0; i < vertexCount_; ++i)
    {
        adjacency_[i].assign(
            neighborSets[i].begin(),
            neighborSets[i].end());

        std::sort(
            adjacency_[i].begin(),
            adjacency_[i].end());

        transitionWeights_[i].assign(
            adjacency_[i].size(),
            0.0f);
    }
}

void WaterTransportSimulation::ResetWater()
{
    for (auto& v : vertices_)
    {
        v.water = 0.0f;
        v.ice = 0.0f;
        v.freezeDelta = 0.0f;
        v.dripMass = 0.0f;

        v.dripCandidate = false;
        v.strongestNeighbor = -1;
        v.strongestWeight = 0.0f;
    }

    std::fill(
        rainInput_.begin(),
        rainInput_.end(),
        0.0f);

    std::fill(
        nextWater_.begin(),
        nextWater_.end(),
        0.0f);

    std::fill(
        nextIce_.begin(),
        nextIce_.end(),
        0.0f);

    lastStepDripMass_ = 0.0f;
    cumulativeDripMass_ = 0.0f;
}

void WaterTransportSimulation::ResetGeometryHistory(
    const vector<vec3>& worldPositions)
{
    if (worldPositions.size() != vertexCount_)
        return;

    previousPositions_ = worldPositions;
    hasGeometryHistory_ = true;

    for (auto& v : vertices_)
        v.surfaceVelocity = vec3(0.0f);
}

void WaterTransportSimulation::RecomputeNormalsAndAreas(
    const vector<vec3>& positions)
{
    for (auto& v : vertices_)
    {
        v.normal = vec3(0.0f);
        v.area = 0.0f;
    }

    for (size_t t = 0; t + 2 < triangleIndices_.size(); t += 3)
    {
        const unsigned int ia = triangleIndices_[t + 0];
        const unsigned int ib = triangleIndices_[t + 1];
        const unsigned int ic = triangleIndices_[t + 2];

        if (ia >= vertexCount_ ||
            ib >= vertexCount_ ||
            ic >= vertexCount_)
        {
            continue;
        }

        const vec3& a = positions[ia];
        const vec3& b = positions[ib];
        const vec3& c = positions[ic];

        const vec3 crossValue = cross(b - a, c - a);
        const float doubleArea = length(crossValue);

        if (doubleArea <= 1e-10f)
            continue;

        const vec3 faceNormal = crossValue / doubleArea;
        const float area = 0.5f * doubleArea;

        vertices_[ia].normal += faceNormal;
        vertices_[ib].normal += faceNormal;
        vertices_[ic].normal += faceNormal;

        vertices_[ia].area += area / 3.0f;
        vertices_[ib].area += area / 3.0f;
        vertices_[ic].area += area / 3.0f;
    }

    for (auto& v : vertices_)
        v.normal = NormalizeOrZero(v.normal);
}

void WaterTransportSimulation::UpdateSurfaceVelocity(
    const vector<vec3>& positions,
    float dt,
    bool resetHistory)
{
    if (!hasGeometryHistory_ || resetHistory)
    {
        ResetGeometryHistory(positions);
        return;
    }

    const float safeDt = std::max(dt, 1e-5f);

    for (size_t i = 0; i < vertexCount_; ++i)
    {
        vertices_[i].surfaceVelocity =
            (positions[i] - previousPositions_[i]) / safeDt;
    }

    previousPositions_ = positions;
}

void WaterTransportSimulation::ComputeFlowAndTransitionWeights(
    const vector<vec3>& positions)
{
    const vec3 gDir = NormalizeOrZero(params_.gravity);

    const float gamma =
        std::max(params_.transitionGamma, 0.05f);

    const float windRef =
        std::max(params_.windReferenceSpeed, 1e-5f);

    for (size_t i = 0; i < vertexCount_; ++i)
    {
        WaterVertexState& vi = vertices_[i];

        vi.relativeWind =
            params_.wind - vi.surfaceVelocity;

        const vec3 gTan =
            NormalizeOrZero(
                ProjectToTangent(gDir, vi.normal));

        const vec3 wTan =
            NormalizeOrZero(
                ProjectToTangent(
                    vi.relativeWind,
                    vi.normal));

        const float windMagnitude01 =
            std::clamp(
                length(vi.relativeWind) / windRef,
                0.0f,
                1.0f);

        const float windWeight =
            windInfluenceEnabled_
            ? params_.alphaWind * windMagnitude01
            : 0.0f;

        const vec3 combined =
            params_.alphaGravity * gTan +
            windWeight * wTan;

        vi.flowDirection =
            NormalizeOrZero(combined);

        auto& weights =
            transitionWeights_[i];

        std::fill(
            weights.begin(),
            weights.end(),
            0.0f);

        vi.strongestNeighbor = -1;
        vi.strongestWeight = 0.0f;

        float sum = 0.0f;

        for (size_t n = 0;
            n < adjacency_[i].size();
            ++n)
        {
            const unsigned int j =
                adjacency_[i][n];

            const vec3 edge =
                NormalizeOrZero(
                    positions[j] - positions[i]);

            const vec3 edgeTan =
                NormalizeOrZero(
                    ProjectToTangent(
                        edge,
                        vi.normal));

            const float score =
                std::max(
                    0.0f,
                    dot(
                        vi.flowDirection,
                        edgeTan));

            const float weightedScore =
                std::pow(score, gamma);

            weights[n] = weightedScore;
            sum += weightedScore;
        }

        if (sum > kEps)
        {
            for (size_t n = 0;
                n < weights.size();
                ++n)
            {
                weights[n] /= sum;

                if (weights[n] >
                    vi.strongestWeight)
                {
                    vi.strongestWeight =
                        weights[n];

                    vi.strongestNeighbor =
                        static_cast<int>(
                            adjacency_[i][n]);
                }
            }
        }
        else
        {
            std::fill(
                weights.begin(),
                weights.end(),
                0.0f);
        }
    }
}

void WaterTransportSimulation::TransportWater(float dt)
{
    const float safeDt =
        std::max(dt, 0.0f);

    const float beta =
        1.0f -
        std::exp(
            -std::max(
                params_.kFlow,
                0.0f) *
            safeDt);

    const vec3 rainDir =
        NormalizeOrZero(
            params_.rainDirection);

    const vec3 gDir =
        NormalizeOrZero(
            params_.gravity);

    const float cosLimit =
        std::cos(
            params_.dripAngleDeg *
            kPi / 180.0f);

    std::fill(
        nextWater_.begin(),
        nextWater_.end(),
        0.0f);

    std::fill(
        nextIce_.begin(),
        nextIce_.end(),
        0.0f);

    lastStepDripMass_ = 0.0f;

    // 1. Rain input
    for (size_t i = 0;
        i < vertexCount_;
        ++i)
    {
        const float exposure =
            rainEnabled_
            ? std::max(
                0.0f,
                dot(
                    vertices_[i].normal,
                    -rainDir))
            : 0.0f;

        rainInput_[i] =
            params_.rainRate *
            vertices_[i].area *
            exposure *
            safeDt;
    }

    // 2. Freeze / Flow / Drip / Remain
    for (size_t i = 0;
        i < vertexCount_;
        ++i)
    {
        WaterVertexState& vi =
            vertices_[i];

        const float availableWater =
            vi.water +
            rainInput_[i];

        if (availableWater <= 0.0f)
        {
            nextIce_[i] = vi.ice;
            vi.freezeDelta = 0.0f;
            vi.dripMass = 0.0f;
            vi.dripCandidate = false; 
            continue;
        }

        const float freezeRate =
            std::max(
                params_.freezeRate,
                0.0f);

        const float freezeMass =
            std::min(
                availableWater,
                freezeRate *
                availableWater *
                safeDt);

        const float liquidAfterFreeze =
            std::max(
                0.0f,
                availableWater -
                freezeMass);


        // 결빙 후 남은 액체를 기준으로 Drip 판단
        const bool waterEnough =
            liquidAfterFreeze >=
            params_.dripThreshold;

        const bool dripGeometry =
            dot(
                vi.normal,
                gDir) >= cosLimit;

        bool localMaximum = true;

        for (unsigned int j : adjacency_[i])
        {
            const float neighborAvailable =
                vertices_[j].water +
                rainInput_[j];

            if (neighborAvailable >
                availableWater + 1e-7f)
            {
                localMaximum = false;
                break;
            }
        }

        const bool dripEligible =
            waterEnough &&
            dripGeometry &&
            localMaximum;

        vi.dripCandidate = dripEligible; 
       

        const auto& weights =
            transitionWeights_[i];

        float weightSum = 0.0f;

        for (float weight : weights)
        {
            weightSum += weight;
        }

        const float wantedFlow =
            weightSum > kEps
            ? beta * liquidAfterFreeze
            : 0.0f;

        const float dripFraction =
            std::clamp(
                params_.dripFraction,
                0.0f,
                1.0f);

        // Interpret dripFraction over a fixed 1/60-second reference interval.
        // p(dt) = 1 - (1 - p(referenceDt))^(dt/referenceDt).
        // Handle zero time and p==1 before log1p(-p).
        constexpr float kDripReferenceDt = 1.0f / 60.0f;
        const float dripStepFraction =
            safeDt <= 0.0f ? 0.0f :
            dripFraction >= 1.0f ? 1.0f :
            -std::expm1(
                std::log1p(-dripFraction) *
                (safeDt / kDripReferenceDt));

        const float wantedDrip =
            dripEligible
            ? dripStepFraction *
            liquidAfterFreeze
            : 0.0f;

        // Flow + Drip이 남은 액체보다 많아지지 않도록 제한
        const float wantedLiquidOut =
            wantedFlow +
            wantedDrip;

        float liquidScale = 1.0f;

        if (wantedLiquidOut >
            liquidAfterFreeze &&
            wantedLiquidOut > kEps)
        {
            liquidScale =
                liquidAfterFreeze /
                wantedLiquidOut;
        }

        const float flowMass =
            wantedFlow *
            liquidScale;

        const float dripMass =
            wantedDrip *
            liquidScale;

        const float remaining =
            std::max(
                0.0f,
                liquidAfterFreeze -
                flowMass -
                dripMass);

        vi.freezeDelta = freezeMass;
        vi.dripMass = dripMass;


        nextWater_[i] += remaining;

        nextIce_[i] =
            vi.ice +
            freezeMass;

        if (flowMass > 0.0f &&
            weightSum > kEps)
        {
            for (size_t n = 0;
                n < adjacency_[i].size();
                ++n)
            {
                const unsigned int j =
                    adjacency_[i][n];

                const float transfer =
                    flowMass *
                    weights[n];

                nextWater_[j] += transfer;
            }
        }

        lastStepDripMass_ +=
            dripMass;
    }

    // 3. Apply new states
    for (size_t i = 0;
        i < vertexCount_;
        ++i)
    {
        vertices_[i].water =
            std::max(
                nextWater_[i],
                0.0f);

        vertices_[i].ice =
            std::max(
                nextIce_[i],
                0.0f);
    }

    cumulativeDripMass_ +=
        lastStepDripMass_;
}

void WaterTransportSimulation::DetectDripCandidates()
{
    const vec3 gDir =
        NormalizeOrZero(
            params_.gravity);

    const float cosLimit =
        std::cos(
            params_.dripAngleDeg *
            kPi / 180.0f);

    for (size_t i = 0;
        i < vertexCount_;
        ++i)
    {
        const float Wi =
            vertices_[i].water;

        const bool waterEnough =
            Wi >=
            params_.dripThreshold;

        const bool dripGeometry =
            dot(
                vertices_[i].normal,
                gDir) >= cosLimit;

        bool localMaximum = true;

        for (const unsigned int j :
        adjacency_[i])
        {
            if (vertices_[j].water >
                Wi + 1e-7f)
            {
                localMaximum = false;
                break;
            }
        }

        vertices_[i].dripCandidate =
            waterEnough &&
            dripGeometry &&
            localMaximum;
    }
}

void WaterTransportSimulation::Update(
    const vector<vec3>& worldPositions,
    float deltaTime,
    bool geometryDiscontinuity)
{
    if (worldPositions.size() != vertexCount_)
        return;

    const float dt =
        std::clamp(
            deltaTime,
            0.0f,
            0.05f);

    UpdateSurfaceVelocity(
        worldPositions,
        dt,
        geometryDiscontinuity);

    RecomputeNormalsAndAreas(
        worldPositions);

    ComputeFlowAndTransitionWeights(
        worldPositions);

    TransportWater(dt);
}

float WaterTransportSimulation::TotalWater() const
{
    float total = 0.0f;

    for (const auto& v : vertices_)
        total += v.water;

    return total;
}

float WaterTransportSimulation::MaxWater() const
{
    float maxWater = 0.0f;

    for (const auto& v : vertices_)
    {
        maxWater =
            std::max(
                maxWater,
                v.water);
    }

    return maxWater;
}

float WaterTransportSimulation::TotalIce() const
{
    float total = 0.0f;

    for (const auto& v : vertices_)
        total += v.ice;

    return total;
}

size_t WaterTransportSimulation::DripCandidateCount() const
{
    size_t count = 0;

    for (const auto& v : vertices_)
    {
        if (v.dripCandidate)
            ++count;
    }

    return count;
}
