#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <vector>
using namespace glm; 
using namespace std; 
struct WaterTransportParams
{
    float alphaGravity = 1.0f;
    float alphaWind = 0.85f;
    float windReferenceSpeed = 2.0f;
    float transitionGamma = 2.0f;

    float kFlow = 2.2f;
    float rainRate = 1.0f;

    float dripThreshold = 0.005f;
    float dripAngleDeg = 120.0f;

    float freezeRate = 0.02f;
    float dripFraction = 0.20f; 

    vec3 gravity{ 0.0f, -1.0f, 0.0f };
    vec3 wind{ 0.0f };
    vec3 rainDirection{ -0.15f, -1.0f, 0.05f };
};

struct WaterVertexState
{
    glm::vec3 normal{ 0.0f, 1.0f, 0.0f };
    glm::vec3 surfaceVelocity{ 0.0f };
    glm::vec3 relativeWind{ 0.0f };
    glm::vec3 flowDirection{ 0.0f };

    float area = 0.0f;
    float water = 0.0f;

    // 47ÁÖÂ÷
    float ice = 0.0f;         
    float freezeDelta = 0.0f; 
    float dripMass = 0.0f;  

    bool dripCandidate = false;

    int strongestNeighbor = -1;
    float strongestWeight = 0.0f;
};

class WaterTransportSimulation
{
public:
    WaterTransportSimulation(
        std::size_t vertexCount,
        const std::vector<unsigned int>& triangleIndices);

    void Update(
        const std::vector<glm::vec3>& worldPositions,
        float deltaTime,
        bool geometryDiscontinuity = false);

    void ResetWater();
    void ResetGeometryHistory(
        const std::vector<glm::vec3>& worldPositions);

    void SetRainEnabled(bool enabled) { rainEnabled_ = enabled; }
    bool IsRainEnabled() const { return rainEnabled_; }

    void SetWindInfluenceEnabled(bool enabled)
    {
        windInfluenceEnabled_ = enabled;
    }

    bool IsWindInfluenceEnabled() const
    {
        return windInfluenceEnabled_;
    }

    WaterTransportParams& Params() { return params_; }
    const WaterTransportParams& Params() const { return params_; }

    const std::vector<WaterVertexState>& Vertices() const
    {
        return vertices_;
    }

    const std::vector<std::vector<unsigned int>>& Adjacency() const
    {
        return adjacency_;
    }

    const std::vector<std::vector<float>>& TransitionWeights() const
    {
        return transitionWeights_;
    }

    float TotalWater() const;
    float MaxWater() const;
    float TotalIce() const;

    float LastStepDripMass() const
    {
        return lastStepDripMass_;
    }

    float CumulativeDripMass() const
    {
        return cumulativeDripMass_;
    }

    std::size_t DripCandidateCount() const;

private:
    void BuildAdjacency();

    void RecomputeNormalsAndAreas(
        const std::vector<glm::vec3>& positions);

    void UpdateSurfaceVelocity(
        const std::vector<glm::vec3>& positions,
        float dt,
        bool resetHistory);

    void ComputeFlowAndTransitionWeights(
        const std::vector<glm::vec3>& positions);

    void TransportWater(float dt);
    void DetectDripCandidates();

    static glm::vec3 NormalizeOrZero(const glm::vec3& v);
    static glm::vec3 ProjectToTangent(
        const glm::vec3& v,
        const glm::vec3& n);

    std::size_t vertexCount_ = 0;

    std::vector<unsigned int> triangleIndices_;
    std::vector<std::vector<unsigned int>> adjacency_;
    std::vector<std::vector<float>> transitionWeights_;
    std::vector<WaterVertexState> vertices_;

    std::vector<glm::vec3> previousPositions_;
    bool hasGeometryHistory_ = false;

    std::vector<float> rainInput_;
    std::vector<float> nextWater_;
    std::vector<float> nextIce_;

    float lastStepDripMass_ = 0.0f;
    float cumulativeDripMass_ = 0.0f;

    WaterTransportParams params_{};

    bool rainEnabled_ = true;
    bool windInfluenceEnabled_ = true;
};
