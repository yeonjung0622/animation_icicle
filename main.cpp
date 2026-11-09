#include "Animation.h"
#include "Animator.h"
#include "Model.h"
#include "Shader.h"
#include "WaterTransport.h"
#include "IcicleGrowth.h"
// Surface collision is integrated in IcicleGrowth.h

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

using namespace std;
using namespace glm;


namespace {

    template <typename T>
    constexpr T MinValue(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }

    template <typename T>
    constexpr T MaxValue(const T& a, const T& b)
    {
        return (a < b) ? b : a;
    }

    template <typename T>
    constexpr T ClampValue(const T& value, const T& low, const T& high)
    {
        return value < low ? low : (high < value ? high : value);
    }

    constexpr int kWindowWidth = 1280;
    constexpr int kWindowHeight = 720;
    constexpr int kMaxShaderBones = 256;
    constexpr float kDisplayFps = 30.0f;
    // 기존 Root와 새로운 Root 사이의 최소 거리
    constexpr float kMinRootSpacing = 0.08f;
    constexpr size_t kMaxIcicleRoots = 128;
    constexpr int kMaxNewRootsPerStep = 4;
    constexpr size_t kMaxDebugRootVectors = 12;
    constexpr size_t kMaxGroundedIcicles = 128;

    // 해당 스텝에서 발생한 최소 Drip Mass
    constexpr float kMinIcicleDripMass = 1e-8f;
    //기존 누적 방식 vs segment 생성 시점 궤적 방향 이용 
    constexpr bool kUseInstantGrowthDirection = true;


    float gYaw = -90.0f;
    float gPitch = 10.0f;
    float gDistance = 4.0f;

    bool gDragging = false;
    bool gWireframe = false;
    bool gDebug = false;
    bool gPickRequested = false;

    bool gShowWater = true;
    bool gShowIce = true;
    bool gShowDrips = true;
    bool gShowFlow = false;
    bool gWaterPaused = false;

    // Week 51: graph-flow -> liquid reservoirs -> new ice segments.
    enum class GrowthModel { LiquidFlow, LegacyRootWind };
    GrowthModel gGrowthModel = GrowthModel::LiquidFlow;
    icicle_growth::Parameters gLiquidParameters;
    icicle_growth::GrowthResolution gGrowthResolution;
    icicle_growth::DirectionResponseParameters gDirectionResponse;
    constexpr float kSlowFreezeRateScale = 0.1f;
    float gFreezeRateScale = 1.0f;
    icicle_growth::FreezingControls gFreezing;
    float gMotionSpeed=1.0f;
    // New growth segments are kept outside the deformed character mesh. Existing ice
    // is not pushed by later limb motion. Toggle with K (resets the experiment).
    bool gSurfaceClearanceEnabled=true;
    bool gClearanceToggleWasDown=false;
    icicle_growth::SurfaceGrid gSurfaceGrid;
    std::vector<icicle_growth::Vec3> gGridPositions,gGridNormals;
    float gAnimationTicksPerSecond=30.0f; // clip time units, NOT measured key frequency
    bool gCarrierFilterEnabled=false;
    bool gCarrierToggleWasDown=false;
    // Primary 53-week experiment: loop the spinning kick while the same ice
    // continuously grows. Single-run and pre-growth are diagnostics only.
    bool gPlayKickOnce=false;
    bool gPreWetBeforeKick=false; // OFF by default: freeze during the actual motion
    bool gUseAccumulatedSteeringDirection=true;
    bool gF9OneShotWasDown=false;
    bool gF10MotionModelWasDown=false;
    bool gF11MemoryWasDown=false;
    bool gF12PreWetWasDown=false;
    icicle_growth::MotionGrowthParameters gMotionGrowthParameters;
    // Diagnostic window: dominant observed FBX interval, not all channels.
    constexpr float kCarrierWindowSeconds=1.0f/30.0f;
    float gPhysicsStep=1.0f/120.0f;
    bool gExperimentKeys[6]={};
    bool gF8ResponseWasDown=false;
    bool gSingleRootExperiment=false;
    struct GroundState
    {
        bool enabled = false; // trajectory diagnostic: no ground locking
        bool initialized = false;
        float height = 0.0f;
        float radius = 0.002f;  // conservative radius for line trajectories
        float bodyClearance = 0.004f;
        float characterLift = 0.0f;
        float halfSize = 5.0f;
    };
    GroundState gGround;
    bool g5WasDown = false, g6WasDown = false;
    bool g7WasDown = false, g8WasDown = false, g9WasDown = false;
    // Animated surface flow is an estimate; the original SPH paper is NOT
    // reproduced by this reduced model. All rates use seconds/model units.
    icicle_growth::Vec3 GrowthVector(const vec3& v) { return {v.x,v.y,v.z}; }
    vec3 ViewerVector(const icicle_growth::Vec3& v) { return {v.x,v.y,v.z}; }

    double gLastMouseX = 0.0;
    double gLastMouseY = 0.0;
    double gPickX = 0.0;
    double gPickY = 0.0;

    bool gSpaceWasDown = false;
    bool gRWasDown = false;
    bool gWWasDown = false;
    bool gFWasDown = false;
    bool gCWasDown = false;
    bool gPWasDown = false;
    bool gDWasDown = false;
    bool gPeriodWasDown = false;
    bool gLBWasDown = false;
    bool gRBWasDown = false;

    bool g1WasDown = false;
    bool g2WasDown = false;
    bool g3WasDown = false;
    bool g4WasDown = false;

    bool gTWasDown = false;
    bool gXWasDown = false;
    bool gHWasDown = false;
    bool gGWasDown = false;
    bool gBWasDown = false;
    bool gVWasDown = false;
    bool gYWasDown = false;
    bool gZWasDown = false;

    void FramebufferSizeCallback(GLFWwindow*, int width, int height)
    {
        glViewport(0, 0, width, height);
    }

    void MouseButtonCallback(GLFWwindow* window, int button, int action, int)
    {
        if (button == GLFW_MOUSE_BUTTON_LEFT)
        {
            gDragging = (action == GLFW_PRESS);
            glfwGetCursorPos(window, &gLastMouseX, &gLastMouseY);
        }

        if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS)
        {
            glfwGetCursorPos(window, &gPickX, &gPickY);
            gPickRequested = true;
        }
    }

    void CursorPositionCallback(GLFWwindow*, double x, double y)
    {
        if (!gDragging)
            return;

        const float dx = static_cast<float>(x - gLastMouseX);
        const float dy = static_cast<float>(y - gLastMouseY);

        gLastMouseX = x;
        gLastMouseY = y;

        gYaw += dx * 0.25f;
        gPitch = ClampValue(gPitch - dy * 0.25f, -89.0f, 89.0f);
    }

    void ScrollCallback(GLFWwindow*, double, double y)
    {
        gDistance = ClampValue(
            gDistance - static_cast<float>(y) * 0.3f,
            0.5f,
            20.0f);
    }

    vec3 GetCameraPosition()
    {
        const float yaw = radians(gYaw);
        const float pitch = radians(gPitch);

        return gDistance * vec3(
            cos(pitch) * cos(yaw),
            sin(pitch),
            cos(pitch) * sin(yaw));
    }

    bool Rising(GLFWwindow* window, int key, bool& previous)
    {
        const bool down = glfwGetKey(window, key) == GLFW_PRESS;
        const bool rising = down && !previous;
        previous = down;
        return rising;
    }

    bool NumberRising(GLFWwindow* window,int number,bool& previous)
    {
        const bool down=glfwGetKey(window,GLFW_KEY_0+number)==GLFW_PRESS ||
            glfwGetKey(window,GLFW_KEY_KP_0+number)==GLFW_PRESS;
        const bool rising=down && !previous;
        previous=down;
        return rising;
    }

    vec3 SafeNormalize(
        const vec3& v,
        const vec3& fallback = vec3(0.0f, -1.0f, 0.0f))
    {
        const float lengthSquared = dot(v, v);
        if (lengthSquared < 1e-10f)
            return fallback;

        return v / sqrt(lengthSquared);
    }

    string V3(const vec3& v)
    {
        ostringstream stream;
        stream << fixed << setprecision(3)
            << "(" << v.x << ", " << v.y << ", " << v.z << ")";
        return stream.str();
    }

    const char* kVertexShader = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPosition;
layout(location=1) in vec3 aNormal;
layout(location=2) in vec2 aTexCoords;
layout(location=3) in ivec4 aBoneIDs;
layout(location=4) in vec4 aWeights;

const int MAX_BONES = 256;

layout(std140) uniform BoneMatrices {
    mat4 uBoneMatrices[MAX_BONES];
};

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vWorldPosition;
out vec3 vNormal;

void main()
{
    mat4 skin = mat4(0.0);
    float total = 0.0;

    for (int i = 0; i < 4; ++i)
    {
        int id = aBoneIDs[i];
        float w = aWeights[i];

        if (id >= 0 && id < MAX_BONES && w > 0.0)
        {
            skin += uBoneMatrices[id] * w;
            total += w;
        }
    }

    vec4 localPosition;
    vec3 localNormal;

    if (total > 0.00001)
    {
        localPosition = skin * vec4(aPosition, 1.0);
        localNormal = mat3(skin) * aNormal;
    }
    else
    {
        localPosition = vec4(aPosition, 1.0);
        localNormal = aNormal;
    }

    vec4 world = uModel * localPosition;
    vWorldPosition = world.xyz;
    vNormal = normalize(mat3(transpose(inverse(uModel))) * localNormal);

    gl_Position = uProjection * uView * world;
}
)GLSL";

    const char* kFragmentShader = R"GLSL(
#version 330 core

in vec3 vWorldPosition;
in vec3 vNormal;
out vec4 FragColor;

uniform vec3 uCameraPosition;
uniform vec3 uBaseColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 l = normalize(vec3(0.5, 1.0, 0.35));
    float diffuse = max(dot(n, l), 0.0);

    vec3 v = normalize(uCameraPosition - vWorldPosition);
    vec3 h = normalize(l + v);
    float specular = pow(max(dot(n, h), 0.0), 32.0);

    FragColor = vec4(
        uBaseColor * (0.22 + 0.78 * diffuse) + vec3(0.25) * specular,
        1.0);
}
)GLSL";

    GLuint Compile(GLenum type, const char* source)
    {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        GLint ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

        if (!ok)
        {
            GLint length = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

            string log(MaxValue(length, 1), '\0');
            glGetShaderInfoLog(shader, length, nullptr, log.data());
            glDeleteShader(shader);
            throw runtime_error(log);
        }

        return shader;
    }

    GLuint Program(const char* vertexSource, const char* fragmentSource)
    {
        const GLuint vertexShader = Compile(GL_VERTEX_SHADER, vertexSource);
        const GLuint fragmentShader = Compile(GL_FRAGMENT_SHADER, fragmentSource);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        GLint ok = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &ok);

        if (!ok)
        {
            GLint length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

            string log(MaxValue(length, 1), '\0');
            glGetProgramInfoLog(program, length, nullptr, log.data());
            throw runtime_error(log);
        }

        return program;
    }

    class DebugRenderer
    {
    public:
        DebugRenderer()
        {
            const char* vertexShader = R"GLSL(
#version 330 core
layout(location=0) in vec3 aPosition;
uniform mat4 uVP;
void main()
{
    gl_Position = uVP * vec4(aPosition, 1.0);
}
)GLSL";

            const char* fragmentShader = R"GLSL(
#version 330 core
out vec4 FragColor;
uniform vec3 uColor;
void main()
{
    FragColor = vec4(uColor, 1.0);
}
)GLSL";

            program_ = Program(vertexShader, fragmentShader);

            glGenVertexArrays(1, &vao_);
            glGenBuffers(1, &vbo_);

            glBindVertexArray(vao_);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_);
            glBufferData(
                GL_ARRAY_BUFFER,
                sizeof(vec3) * 64,
                nullptr,
                GL_DYNAMIC_DRAW);

            glEnableVertexAttribArray(0);
            glVertexAttribPointer(
                0,
                3,
                GL_FLOAT,
                GL_FALSE,
                sizeof(vec3),
                reinterpret_cast<void*>(0));

            glBindVertexArray(0);

            vpLocation_ = glGetUniformLocation(program_, "uVP");
            colorLocation_ = glGetUniformLocation(program_, "uColor");
        }

        ~DebugRenderer()
        {
            if (vbo_)
                glDeleteBuffers(1, &vbo_);
            if (vao_)
                glDeleteVertexArrays(1, &vao_);
            if (program_)
                glDeleteProgram(program_);
        }

        void Draw(
            GLenum primitive,
            const vec3* positions,
            GLsizei count,
            const vec3& color,
            const mat4&,
            float lineWidth = 2.0f,
            float pointSize = 1.0f)
        {
            if (count <= 0)
                return;

            // All scene overlays use the same VP. Group by render state and
            // upload once per group rather than once per arrow shaft/head.
            const GLenum type=primitive==GL_LINE_STRIP?GL_LINES:primitive;
            Batch* batch=nullptr;
            for (auto& candidate:batches_)
                if (candidate.primitive==type && candidate.color.x==color.x &&
                    candidate.color.y==color.y && candidate.color.z==color.z &&
                    candidate.lineWidth==lineWidth && candidate.pointSize==pointSize)
                { batch=&candidate; break; }
            if (!batch)
            {
                batches_.push_back({type,color,lineWidth,pointSize,{}});
                batch=&batches_.back();
            }
            if (primitive==GL_LINE_STRIP)
                for (GLsizei i=1;i<count;++i)
                {
                    batch->positions.push_back(positions[i-1]);
                    batch->positions.push_back(positions[i]);
                }
            else batch->positions.insert(batch->positions.end(),positions,positions+count);
        }

        void Flush(const mat4& vp)
        {
            glUseProgram(program_);
            glUniformMatrix4fv(vpLocation_,1,GL_FALSE,&vp[0][0]);
            glBindVertexArray(vao_);
            glBindBuffer(GL_ARRAY_BUFFER,vbo_);
            for (auto& batch:batches_)
            {
                if (batch.positions.empty()) continue;
                glUniform3fv(colorLocation_,1,&batch.color[0]);
                glBufferData(GL_ARRAY_BUFFER,sizeof(vec3)*batch.positions.size(),
                             batch.positions.data(),GL_DYNAMIC_DRAW);
                glLineWidth(batch.lineWidth);
                glPointSize(batch.pointSize);
                glDrawArrays(batch.primitive,0,static_cast<GLsizei>(batch.positions.size()));
                batch.positions.clear(); // retain capacity for the next frame
            }
            glBindVertexArray(0);
            glUseProgram(0);
        }

        void Line(
            const vec3& a,
            const vec3& b,
            const vec3& color,
            const mat4& vp,
            float width = 2.0f)
        {
            const vec3 points[2] = { a, b };
            Draw(GL_LINES, points, 2, color, vp, width);
        }

        void Point(
            const vec3& position,
            const vec3& color,
            const mat4& vp,
            float size = 10.0f)
        {
            Draw(GL_POINTS, &position, 1, color, vp, 1.0f, size);
        }

        void Tri(
            const vec3& a,
            const vec3& b,
            const vec3& c,
            const vec3& color,
            const mat4& vp)
        {
            const vec3 points[6] = { a, b, b, c, c, a };
            Draw(GL_LINES, points, 6, color, vp, 4.0f);
        }

        void Polyline(
            const vector<vec3>& points,
            const vec3& color,
            const mat4& vp)
        {
            if (points.size() > 1)
            {
                Draw(
                    GL_LINE_STRIP,
                    points.data(),
                    static_cast<GLsizei>(points.size()),
                    color,
                    vp,
                    5.0f);
            }
        }

        void Arrow(
            const vec3& origin,
            const vec3& vector,
            const vec3& color,
            const mat4& vp,
            const vec3& cameraPosition)
        {
            const float vectorLength = length(vector);
            if (vectorLength < 1e-6f)
                return;

            const vec3 direction = vector / vectorLength;
            const vec3 tip = origin + vector;

            Line(origin, tip, color, vp, 3.0f);

            vec3 side = cross(
                direction,
                SafeNormalize(cameraPosition - tip, vec3(0.0f, 0.0f, 1.0f)));

            if (dot(side, side) < 1e-8f)
                side = cross(direction, vec3(0.0f, 1.0f, 0.0f));

            side = SafeNormalize(side, vec3(1.0f, 0.0f, 0.0f));

            const float headLength = MaxValue(0.025f, vectorLength * 0.18f);
            const float headWidth = headLength * 0.55f;
            const vec3 back = tip - direction * headLength;

            Line(tip, back + side * headWidth, color, vp, 3.0f);
            Line(tip, back - side * headWidth, color, vp, 3.0f);
        }

    private:
        struct Batch
        {
            GLenum primitive;
            vec3 color;
            float lineWidth,pointSize;
            vector<vec3> positions;
        };
        vector<Batch> batches_;
        GLuint program_ = 0;
        GLuint vao_ = 0;
        GLuint vbo_ = 0;
        GLint vpLocation_ = -1;
        GLint colorLocation_ = -1;
    };

    struct Ray
    {
        vec3 origin;
        vec3 direction;
    };

    Ray MouseRay(
        double mouseX,
        double mouseY,
        int width,
        int height,
        const mat4& view,
        const mat4& projection,
        const vec3& cameraPosition)
    {
        const float x = 2.0f * static_cast<float>(mouseX) / static_cast<float>(width) - 1.0f;
        const float y = 1.0f - 2.0f * static_cast<float>(mouseY) / static_cast<float>(height);

        const vec4 clip(x, y, -1.0f, 1.0f);
        vec4 eye = inverse(projection) * clip;
        eye = vec4(eye.x, eye.y, -1.0f, 0.0f);

        return {
            cameraPosition,
            SafeNormalize(
                vec3(inverse(view) * eye),
                vec3(0.0f, 0.0f, -1.0f)) };
    }

    bool RayTri(
        const Ray& ray,
        const vec3& a,
        const vec3& b,
        const vec3& c,
        float& t,
        float& u,
        float& v)
    {
        constexpr float epsilon = 1e-7f;

        const vec3 edge1 = b - a;
        const vec3 edge2 = c - a;
        const vec3 p = cross(ray.direction, edge2);
        const float determinant = dot(edge1, p);

        if (abs(determinant) < epsilon)
            return false;

        const float inverseDeterminant = 1.0f / determinant;
        const vec3 tv = ray.origin - a;

        u = dot(tv, p) * inverseDeterminant;
        if (u < 0.0f || u > 1.0f)
            return false;

        const vec3 q = cross(tv, edge1);
        v = dot(ray.direction, q) * inverseDeterminant;
        if (v < 0.0f || u + v > 1.0f)
            return false;

        t = dot(edge2, q) * inverseDeterminant;
        return t > epsilon;
    }

    mat3 TriangleFrame(
        const vec3& a,
        const vec3& b,
        const vec3& c,
        const vec3& normal)
    {
        vec3 tangent = SafeNormalize(b - a, vec3(1.0f, 0.0f, 0.0f));
        vec3 bitangent = cross(normal, tangent);

        if (dot(bitangent, bitangent) < 1e-8f)
        {
            tangent = SafeNormalize(c - a, vec3(1.0f, 0.0f, 0.0f));
            bitangent = cross(normal, tangent);
        }

        bitangent = SafeNormalize(bitangent, vec3(0.0f, 0.0f, 1.0f));
        tangent = SafeNormalize(cross(bitangent, normal), tangent);

        return mat3(tangent, bitangent, normal);
    }

    struct SelectedSurface
    {
        bool valid = false;
        bool hasPrev = false;

        size_t tri = 0;
        array<unsigned int, 3> idx{ 0, 0, 0 };
        vec3 bary{ 1.0f / 3.0f };

        vec3 p0{};
        vec3 p1{};
        vec3 p2{};
        vec3 pos{};
        vec3 prevPos{};
        vec3 normal{ 0.0f, 1.0f, 0.0f };
        vec3 prevNormal{ 0.0f, 1.0f, 0.0f };
        vec3 velocity{};
        vec3 rawVelocity{};

        float speed = 0.0f;
        float normalDegSec = 0.0f;

        mat3 frame{ 1.0f };
    };

    struct FlowState
    {
        vec3 wind{ 0.0f };
        vec3 gravity{ 0.0f, -1.0f, 0.0f };
        // Vertical rain removes a fixed lateral precipitation bias, leaving
        // character motion as the experimental source of direction changes.
        vec3 rain{ 0.0f, -1.0f, 0.0f };
        vec3 relative{};
        vec3 growth{ 0.0f, -1.0f, 0.0f };

        float wg = 1.0f;
        float wr = 0.35f;
        float wm = 0.35f;
    };

    // Existing trajectory/debug freeze state.
    // This is separate from the vertex Ice State used by WaterTransportSimulation.
    struct FreezeState
    {
        bool active = false;
        float startTime = 0.0f;
        float duration = 1.0f;
        float elapsed = 0.0f;
        double weight = 0.0;
        float segmentLength = 0.01f;

        float tipWater = 0.0f;
        double frozenAtTip = 0.0;
        icicle_growth::LiquidState liquid;
        icicle_growth::GrowthBuffer growthBuffer;
        icicle_growth::DirectionResponseState directionResponse;
        icicle_growth::CarrierVelocityFilter carrierFilter;
        icicle_growth::MotionGrowthState motionGrowth;
        icicle_growth::ParcelState parcelCloud;
        vec3 tipVelocityRawDebug{};
        vec3 tipVelocityDebug{},tipAccelerationDebug{},previousTipVelocity{};
        bool hasPreviousTipVelocity=false;
        double lastFrozenMass=0;
        double lastDetachedMass=0;
        float lastDirectionCoherence=1.f;
        unsigned unstableDirectionSteps=0;
        unsigned turnLimitedSegments=0;
        vec3 rawGrowthLocal{}, respondedGrowthLocal{}, emittedGrowthLocal{};
        float attachmentTurnDegrees=0;
        double addedPathLength=0;
        double axialFrozenMass = 0.0;
        double nonAxialFrozenMass = 0.0;
        float pathLength = 0.0f;
        vec3 previousTipWorld{};
        vec3 inletVelocityWorld{};
        vec3 contactWorld{};
        bool hasPreviousTip = false;
        bool groundContact = false;
        bool existingIceContact = false;
        bool groundLocked = false;
        icicle_growth::RigidPose previousAttachmentPose;
        bool hasAttachmentPose = false;

        float segmentFreezeAmount = 0.001f; 

        double totalGrowthTime = 0.0;

        vector<float> freezeTimes{ 0.0f }; 

        int startFrame = 0;

        vec3 sumLocal{};
        vector<vec3> localPath{ vec3(0.0f) };

        void ResetPath()
        {
            tipWater = 0.0f;
            frozenAtTip = 0.0;
            liquid = icicle_growth::LiquidState{};
            growthBuffer.Clear();
            directionResponse.Reset();
            motionGrowth.Reset();
            parcelCloud.Reset();
            carrierFilter.Reset(); tipVelocityRawDebug=vec3(0);
            tipVelocityDebug=tipAccelerationDebug=previousTipVelocity=vec3(0);
            hasPreviousTipVelocity=false;
            lastFrozenMass=lastDetachedMass=0;
            lastDirectionCoherence=1.f;
            unstableDirectionSteps=turnLimitedSegments=0;
            axialFrozenMass = nonAxialFrozenMass = 0.0;
            pathLength = 0.0f;
            previousTipWorld = inletVelocityWorld = contactWorld = vec3(0.0f);
            hasPreviousTip = groundContact = existingIceContact = false;
            groundLocked = false;
            hasAttachmentPose = false;
            totalGrowthTime = 0.0f; 
            freezeTimes = { 0.0f }; 

            elapsed = 0.0f;
            weight = 0.0f;
            sumLocal = vec3(0.0f);
            localPath = { vec3(0.0f) };
        }

        void Stop()
        {
            active = false;
            ResetPath();
        }
    };

    struct IcicleState
    {
        SelectedSurface root;
        FlowState flow;
        FreezeState freeze;
    };

    struct Commands
    {
        bool freeze = false;
        bool clear = false;
        bool print = false;
        bool step = false;
        bool reset = false;
    };

    Commands ProcessKeyboard(
        GLFWwindow* window,
        Animator* animator,
        float dt,
        FlowState& flow)
    {
        (void)dt; // currently unused in motion-only keyboard handling
        Commands commands;

        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GLFW_TRUE);

        if (Rising(window, GLFW_KEY_SPACE, gSpaceWasDown) && animator)
            animator->SetPlaying(!animator->IsPlaying());

        if (Rising(window, GLFW_KEY_R, gRWasDown) && animator)
        {
            animator->Reset();
            commands.reset = true;
        }

        if (Rising(window, GLFW_KEY_W, gWWasDown))
        {
            gWireframe = !gWireframe;
            glPolygonMode(GL_FRONT_AND_BACK, gWireframe ? GL_LINE : GL_FILL);
        }

        if (Rising(window, GLFW_KEY_F, gFWasDown))
            commands.freeze = true;

        if (Rising(window, GLFW_KEY_C, gCWasDown))
            commands.clear = true;

        if (Rising(window, GLFW_KEY_P, gPWasDown))
            commands.print = true;

        if (Rising(window, GLFW_KEY_D, gDWasDown))
            gDebug = !gDebug;

        if (Rising(window, GLFW_KEY_PERIOD, gPeriodWasDown))
            commands.step = true;


        flow.wind=vec3(0); // motion-only study

        if (Rising(window, GLFW_KEY_Y, gYWasDown))
        {
            gWaterPaused = !gWaterPaused;
            cout << "Water Simulation "
                << (gWaterPaused ? "PAUSED" : "RUNNING")
                << "\n";
        }

        return commands;
    }

    bool Pick(
        const Ray& ray,
        const vector<vec3>& vertices,
        const vector<unsigned int>& indices,
        SelectedSurface& selected)
    {
        float best = numeric_limits<float>::max();
        float bestU = 0.0f;
        float bestV = 0.0f;
        size_t bestTriangle = numeric_limits<size_t>::max();

        for (size_t triangle = 0; triangle < indices.size() / 3; ++triangle)
        {
            const unsigned int i0 = indices[triangle * 3 + 0];
            const unsigned int i1 = indices[triangle * 3 + 1];
            const unsigned int i2 = indices[triangle * 3 + 2];

            if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size())
                continue;

            float distance = 0.0f;
            float u = 0.0f;
            float v = 0.0f;

            if (RayTri(ray, vertices[i0], vertices[i1], vertices[i2], distance, u, v) &&
                distance < best)
            {
                best = distance;
                bestU = u;
                bestV = v;
                bestTriangle = triangle;
            }
        }

        if (bestTriangle == numeric_limits<size_t>::max())
            return false;

        const size_t base = bestTriangle * 3;

        selected = SelectedSurface{};
        selected.valid = true;
        selected.tri = bestTriangle;
        selected.idx = {
            indices[base],
            indices[base + 1],
            indices[base + 2] };
        selected.bary = vec3(1.0f - bestU - bestV, bestU, bestV);

        cout << "Selected triangle " << selected.tri
            << " | indices " << selected.idx[0]
            << ", " << selected.idx[1]
            << ", " << selected.idx[2]
            << " | bary " << V3(selected.bary)
            << "\n";

        return true;
    }

    bool SelectBestDripCandidateRoot(
        SelectedSurface& selected,
        const WaterTransportSimulation& water,
        const vector<vec3>& worldPositions,
        const vector<unsigned int>& triangleIndices)
    {
        const auto& states = water.Vertices();

        if (states.empty() ||
            states.size() != worldPositions.size())
        {
            return false;
        }

        size_t bestVertex =
            numeric_limits<size_t>::max();

        float bestDripMass = -1.0f;
        float bestWater = -1.0f;

        // 1. 현재 Drip Candidate 중 하나 선택
        for (size_t i = 0; i < states.size(); ++i)
        {
            const WaterVertexState& state = states[i];

            if (!state.dripCandidate)
                continue;

            // 실제 떨어진 물이 많은 지점을 우선
            // 동일하면 현재 water가 많은 지점을 선택
            if (state.dripMass > bestDripMass + 1e-7f ||
                (abs(state.dripMass - bestDripMass) <= 1e-7f &&
                    state.water > bestWater))
            {
                bestVertex = i;
                bestDripMass = state.dripMass;
                bestWater = state.water;
            }
        }

        if (bestVertex ==
            numeric_limits<size_t>::max())
        {
            return false;
        }

        // 2. 해당 vertex가 포함된 Triangle 탐색
        for (size_t t = 0;
            t + 2 < triangleIndices.size();
            t += 3)
        {
            const unsigned int i0 =
                triangleIndices[t + 0];

            const unsigned int i1 =
                triangleIndices[t + 1];

            const unsigned int i2 =
                triangleIndices[t + 2];

            int corner = -1;

            if (i0 == bestVertex)
                corner = 0;
            else if (i1 == bestVertex)
                corner = 1;
            else if (i2 == bestVertex)
                corner = 2;

            if (corner < 0)
                continue;

            // 기존 우클릭 선택 정보를
            // Drip Candidate 기반으로 자동 생성
            selected = SelectedSurface{};

            selected.valid = true;
            selected.tri = t / 3;

            selected.idx = {
                i0,
                i1,
                i2
            };

            // Candidate가 vertex이므로
            // Barycentric coordinate는 one-hot
            if (corner == 0)
                selected.bary =
                vec3(1.0f, 0.0f, 0.0f);

            else if (corner == 1)
                selected.bary =
                vec3(0.0f, 1.0f, 0.0f);

            else
                selected.bary =
                vec3(0.0f, 0.0f, 1.0f);

            cout
                << "Auto Drip Root"
                << " | vertex " << bestVertex
                << " | triangle " << selected.tri
                << " | dripMass " << bestDripMass
                << " | water " << bestWater
                << "\n";

            return true;
        }

        return false;
    }

    float SelectedDripSupply(
        const SelectedSurface& selected,
        const WaterTransportSimulation& water)
    {
        if (!selected.valid)
            return 0.0f;

        const auto& states =
            water.Vertices();

        if (selected.idx[0] >= states.size() ||
            selected.idx[1] >= states.size() ||
            selected.idx[2] >= states.size())
        {
            return 0.0f;
        }

        return std::max(
            0.0f,
            selected.bary.x *
            states[selected.idx[0]].dripMass +
            selected.bary.y *
            states[selected.idx[1]].dripMass +
            selected.bary.z *
            states[selected.idx[2]].dripMass);
    }
    void UpdateSelected(
        SelectedSurface& selected,
        const vector<vec3>& vertices,
        float dt,
        bool discontinuity, bool loopBoundary=false)
    {
        const mat3 oldFrame=selected.frame;
        const vec3 oldVelocity=selected.velocity;
        const vec3 oldRawVelocity=selected.rawVelocity;
        if (!selected.valid)
            return;

        if (selected.idx[0] >= vertices.size() ||
            selected.idx[1] >= vertices.size() ||
            selected.idx[2] >= vertices.size())
        {
            selected.valid = false;
            return;
        }

        selected.p0 = vertices[selected.idx[0]];
        selected.p1 = vertices[selected.idx[1]];
        selected.p2 = vertices[selected.idx[2]];

        selected.normal = SafeNormalize(
            cross(selected.p1 - selected.p0, selected.p2 - selected.p0),
            vec3(0.0f, 1.0f, 0.0f));

        selected.pos =
            selected.bary.x * selected.p0 +
            selected.bary.y * selected.p1 +
            selected.bary.z * selected.p2;

        selected.frame = TriangleFrame(
            selected.p0,
            selected.p1,
            selected.p2,
            selected.normal);

        if (discontinuity)
            selected.hasPrev = false;

        if (loopBoundary && !discontinuity && selected.hasPrev)
        {
            // Do not differentiate across the clip seam. Carry the derivative
            // in the persistent attachment frame; normal filtering resumes next step.
            selected.velocity=selected.frame*transpose(oldFrame)*oldVelocity;
            selected.rawVelocity=selected.frame*transpose(oldFrame)*oldRawVelocity;
            selected.speed=length(selected.velocity);
            selected.normalDegSec=0;
        }
        else if (selected.hasPrev && dt > 1e-6f)
        {
            selected.rawVelocity = (selected.pos - selected.prevPos) / dt;

            const float alpha = 1.0f - exp(-12.0f * dt);
            selected.velocity = mix(selected.velocity, selected.rawVelocity, alpha);
            selected.speed = length(selected.velocity);

            const float cosine = ClampValue(
                dot(selected.prevNormal, selected.normal),
                -1.0f,
                1.0f);

            selected.normalDegSec = degrees(acos(cosine)) / dt;
        }
        else
        {
            selected.rawVelocity = vec3(0.0f);
            selected.velocity = vec3(0.0f);
            selected.speed = 0.0f;
            selected.normalDegSec = 0.0f;
        }

        selected.prevPos = selected.pos;
        selected.prevNormal = selected.normal;
        selected.hasPrev = true;
    }

    float RequiredBodyGroundLift(const vector<vec3>& positions)
    {
        if (!gGround.enabled || positions.empty()) return 0.0f;
        float minimumY = positions.front().y;
        for (const vec3& p : positions) minimumY = std::min(minimumY, p.y);
        return std::max(0.0f, gGround.height + gGround.bodyClearance - minimumY);
    }

    float ApplyBodyGroundConstraint(vector<vec3>& positions)
    {
        const float lift=RequiredBodyGroundLift(positions);
        for (vec3& p:positions) p.y+=lift;
        return lift;
    }

    icicle_growth::RigidPose AttachmentPose(const SelectedSurface& root)
    {
        return {GrowthVector(root.pos),icicle_growth::OrientationFromAxes(
            GrowthVector(root.frame[0]),GrowthVector(root.frame[1]),GrowthVector(root.frame[2]))};
    }

    void UpdateFlow(
        FlowState& flow,
        const SelectedSurface& selected)
    {
        // 1. Root Velocity
        const vec3 rootVelocity =
            selected.valid
            ? selected.velocity
            : vec3(0.0f);

        // 2. Relative Wind
        // v_rel = v_wind - v_root
        flow.relative =
            flow.wind - rootVelocity;

        // 3. Normalize Gravity
        const vec3 gravityDir =
            SafeNormalize(
                flow.gravity,
                vec3(0.0f, -1.0f, 0.0f));

        // 4. Normalize Relative Wind
        const vec3 relativeDir =
            SafeNormalize(
                flow.relative,
                vec3(0.0f));

        // 5. Relative Wind Influence
        // omega = 0 : gravity only
        // omega = 1 : relative wind only
        const float omega =
            ClampValue(
                flow.wm,
                0.0f,
                1.0f);

        // 6. Growth Direction
        flow.growth =
            SafeNormalize(
                (1.0f - omega) * gravityDir +
                omega * relativeDir,
                gravityDir);
    }

    void StartFreeze(FreezeState& freeze, const SelectedSurface& selected, float time)
    {
        if (!selected.valid)
        {
            cout << "Select triangle first.\n";
            return;
        }

        freeze.active = true;
        freeze.startTime = time;
        freeze.startFrame = static_cast<int>(round(time * kDisplayFps));
        freeze.ResetPath();


    }

    // Use the same public per-vertex data as the transport overlay; no change
    // to WaterTransport.h or an invented physical water velocity is required.
    vector<icicle_growth::SurfaceSample> CaptureGrowthSamples(
        const WaterTransportSimulation& water, const vector<vec3>& positions)
    {
        const auto& vertices = water.Vertices();
        vector<icicle_growth::SurfaceSample> samples(vertices.size());
        if (positions.size() != vertices.size()) return {};
        for (size_t i=0; i<vertices.size(); ++i)
        {
            samples[i].position = GrowthVector(positions[i]);
            samples[i].normal = GrowthVector(vertices[i].normal);
            samples[i].surfaceVelocity = GrowthVector(vertices[i].surfaceVelocity);
            samples[i].flowDirection = GrowthVector(vertices[i].flowDirection);
            samples[i].water = vertices[i].water;
            samples[i].dripMass = vertices[i].dripMass;
        }
        return samples;
    }

    bool CheckExistingIceGroundContact(FreezeState& freeze,
                                       const SelectedSurface& selected)
    {
        freeze.existingIceContact = false;
        if (!gGround.enabled || !selected.valid) return false;
        const float limit = gGround.height + gGround.radius;
        // A linear segment attains minimum height at an endpoint. The radius
        // is a conservative constant offset, not a reconstructed ice surface.
        for (size_t i=1; i<freeze.localPath.size(); ++i)
        {
            const vec3 world = selected.pos + selected.frame * freeze.localPath[i];
            if (world.y < limit - 1e-5f)
            {
                freeze.existingIceContact = true;
                freeze.contactWorld = world;
                return true;
            }
        }
        return false;
    }

    void DepositPendingWithoutElongation(FreezeState& freeze)
    {
        // This mass remains ice in the accounting. Radial accretion geometry
        // is outside the current centreline model; do not delete the old path.
        freeze.nonAxialFrozenMass += freeze.frozenAtTip;
        freeze.frozenAtTip = 0;
        freeze.weight = 0;
        freeze.sumLocal = vec3(0);
        freeze.growthBuffer.Clear();
    }

    void AppendFrozenSegments(FreezeState& freeze, const SelectedSurface& selected,
                              const vec3& instantLocal, bool useInstant)
    {
        if (!(freeze.segmentFreezeAmount > 0) || !(freeze.segmentLength > 0)) return;
        freeze.groundContact = false;
        const vec3 worldTip = selected.pos + selected.frame * freeze.localPath.back();
        if (gGround.enabled && worldTip.y <= gGround.height + gGround.radius + 1e-6f)
        {
            freeze.groundContact = true;
            freeze.contactWorld = worldTip;
            DepositPendingWithoutElongation(freeze);
            return;
        }
        while (freeze.frozenAtTip + 1e-12 >= freeze.segmentFreezeAmount)
        {
            const vec3 direction = SafeNormalize(
                useInstant ? instantLocal : freeze.sumLocal, instantLocal);
            const vec3 localFrom = freeze.localPath.back();
            const vec3 localTo = localFrom + direction * freeze.segmentLength;
            const vec3 from = selected.pos + selected.frame * localFrom;
            const vec3 to = selected.pos + selected.frame * localTo;
            icicle_growth::GroundClip clip;
            clip.endpoint = GrowthVector(to);
            if (gGround.enabled)
                clip = icicle_growth::ClipGrowthToGround(
                    GrowthVector(from), GrowthVector(to), gGround.height, gGround.radius);
            if (clip.fraction > 1e-6f)
            {
                // Only new geometry is clipped. Existing local points stay fixed.
                const vec3 endpoint = clip.hit
                    ? transpose(selected.frame) * (ViewerVector(clip.endpoint) - selected.pos)
                    : localTo;
                freeze.localPath.push_back(endpoint);
                freeze.freezeTimes.push_back(static_cast<float>(freeze.totalGrowthTime));
                freeze.pathLength += length(endpoint-localFrom);
            }
            const double consumed = freeze.segmentFreezeAmount * clip.fraction;
            const double before = freeze.frozenAtTip;
            freeze.axialFrozenMass += consumed;
            freeze.frozenAtTip = std::max(0.0, before-consumed);
            const float keep = before>1e-12
                ? static_cast<float>(freeze.frozenAtTip/before) : 0.0f;
            freeze.sumLocal *= keep;
            freeze.weight = freeze.frozenAtTip;
            if (clip.hit)
            {
                freeze.groundContact = true;
                freeze.contactWorld = ViewerVector(clip.endpoint);
                DepositPendingWithoutElongation(freeze);
                break;
            }
        }
    }

    void AppendGrowthChunks(FreezeState& freeze,const SelectedSurface& selected,
                            const vector<icicle_growth::GrowthChunk>& chunks)
    {
        freeze.groundContact=false;
        const double density=gGrowthResolution.linearMassDensity;
        if (!(density>0)) return;
        for (const auto& chunk:chunks)
        {
            const vec3 localFrom=freeze.localPath.back();
            const float chunkLength=static_cast<float>(chunk.mass/density);
            if (!(chunkLength>0) || !std::isfinite(chunkLength)) continue;
            auto heading=icicle_growth::Unit(chunk.directionLocal);
            // Spatial curvature limit: an extremely short frozen segment must
            // not turn 120 degrees in 1e-4 model units. Existing ice is untouched.
            if (freeze.localPath.size()>1) {
                const vec3 previous=localFrom-freeze.localPath[freeze.localPath.size()-2];
                if (length(previous)>1e-8f) {
                    const float angleLimit=radians(gGrowthResolution.maxTurnDegreesPerMaxLength)*
                        std::clamp(chunkLength/std::max(1e-6f,freeze.segmentLength),0.f,1.f);
                    const auto limited=icicle_growth::LimitDirectionTurn(
                        GrowthVector(previous),heading,angleLimit);
                    if (icicle_growth::Dot(limited,heading)<1.f-1e-5f)
                        ++freeze.turnLimitedSegments;
                    heading=limited;
                }
            }
            const vec3 localTo=localFrom+ViewerVector(heading)*chunkLength;
            const vec3 from=selected.pos+selected.frame*localFrom;
            const vec3 to=selected.pos+selected.frame*localTo;
            icicle_growth::GroundClip clip;
            clip.endpoint=GrowthVector(to);
            if (gGround.enabled)
                clip=icicle_growth::ClipGrowthToGround(
                    GrowthVector(from),GrowthVector(to),gGround.height,gGround.radius);
            bool surfaceHit=false;
            if (gSurfaceClearanceEnabled) {
                const auto surface=gSurfaceGrid.ClipSegment(GrowthVector(from),GrowthVector(to));
                if (surface.hit && surface.fraction<=clip.fraction) { clip=surface; surfaceHit=true; }
            }
            const vec3 endpoint=clip.hit?
                transpose(selected.frame)*(ViewerVector(clip.endpoint)-selected.pos):localTo;
            const float addedLength=length(endpoint-localFrom);
            if (clip.fraction>1e-6f && addedLength>1e-9f)
            {
                bool merged=false;
                if (freeze.localPath.size()>1)
                {
                    const vec3 before=freeze.localPath[freeze.localPath.size()-2];
                    const vec3 oldEdge=localFrom-before,newEdge=endpoint-localFrom;
                    // Extending an exactly straight tip preserves old geometry
                    // while avoiding redundant time-window vertices at rest.
                    if (length(oldEdge)+addedLength<=freeze.segmentLength+1e-7f &&
                        dot(SafeNormalize(oldEdge),SafeNormalize(newEdge))>1.f-1e-7f)
                    {
                        freeze.localPath.back()=endpoint;
                        freeze.freezeTimes.back()=static_cast<float>(chunk.time);
                        merged=true;
                    }
                }
                if (!merged) {
                    freeze.localPath.push_back(endpoint);
                    freeze.freezeTimes.push_back(static_cast<float>(chunk.time));
                }
                freeze.pathLength+=addedLength;
                freeze.addedPathLength+=addedLength;
                freeze.emittedGrowthLocal=SafeNormalize(endpoint-localFrom);
            }
            const double consumed=chunk.mass*clip.fraction;
            freeze.axialFrozenMass+=consumed;
            freeze.frozenAtTip=std::max(0.,freeze.frozenAtTip-consumed);
            if (clip.hit) {
                if (surfaceHit) {
                    // Blocked frozen mass is retained as non-axial ice, never silently lost.
                    // No geometry is invented for this deposit; frozen segments stay rigid.
                    DepositPendingWithoutElongation(freeze);
                    return;
                }
                freeze.groundContact=true;
                freeze.contactWorld=ViewerVector(clip.endpoint);
                DepositPendingWithoutElongation(freeze);
                return;
            }
        }
        freeze.sumLocal=ViewerVector(freeze.growthBuffer.moment);
        freeze.weight=freeze.frozenAtTip;
    }

    // Rebuilt once per physics tick from the same world positions the water graph uses.
    void BuildSurfaceGrid(const vector<vec3>& worldPositions,
        const vector<WaterVertexState>& vertices,const vector<unsigned int>& triangleIndices)
    {
        if (vertices.size()!=worldPositions.size()) { gSurfaceGrid=icicle_growth::SurfaceGrid{}; return; }
        gGridPositions.resize(worldPositions.size()); gGridNormals.resize(worldPositions.size());
        for (size_t i=0;i<worldPositions.size();++i) {
            gGridPositions[i]=GrowthVector(worldPositions[i]);
            gGridNormals[i]=GrowthVector(vertices[i].normal);
        }
        static float cell=0.0f;
        if (cell<=0.0f) { // ~2.5x the mean edge length, measured once
            double sum=0; size_t n=0;
            for (size_t t=0;t+2<triangleIndices.size() && n<6000;t+=3,n+=1)
                sum+=length(worldPositions[triangleIndices[t+1]]-worldPositions[triangleIndices[t]]);
            cell=n? static_cast<float>(2.5*sum/n) : 0.03f;
        }
        gSurfaceGrid.Build(gGridPositions,gGridNormals,triangleIndices,cell);
    }

    void UpdateFreeze(FreezeState& freeze, const SelectedSurface& selected,
        FlowState& flow, const icicle_growth::Supply& supply, float freezeRate,
        float dt, bool advance, bool discontinuity, bool airEnabled, bool loopBoundary=false)
    {
        freeze.lastFrozenMass=freeze.lastDetachedMass=0;
        freeze.lastDirectionCoherence=1.f;
        freeze.unstableDirectionSteps=freeze.turnLimitedSegments=0;
        freeze.rawGrowthLocal=freeze.respondedGrowthLocal=freeze.emittedGrowthLocal=vec3(0);
        freeze.attachmentTurnDegrees=0;
        freeze.addedPathLength=0;
        if (!freeze.active || !selected.valid || freeze.localPath.empty()) return;
        // Conservative seam policy: preserve state and skip this single growth
        // step. Endpoint continuity is not inferable from two rendered poses alone.
        // In particular, never turn the seam's translation into a physical velocity.
        if (loopBoundary && !discontinuity)
        {
            const auto oldOrientation=freeze.hasAttachmentPose
                ? freeze.previousAttachmentPose.orientation
                : AttachmentPose(selected).orientation;
            const auto newOrientation=AttachmentPose(selected).orientation;
            const auto oldCarrier=GrowthVector(freeze.previousTipVelocity);
            // Rotate the reference carrier velocity into the new attachment
            // frame instead of forcing its velocity to zero at every loop.
            // This preserves liquid-minus-carrier velocity through a periodic
            // seam (it does not simulate the unobserved motion inside a jump).
            const auto newCarrier=icicle_growth::RotateBetweenFrames(
                oldCarrier,oldOrientation,newOrientation);
            const auto rebase=[&](icicle_growth::Vec3 v) {
                return icicle_growth::RebaseCarrierVelocity(
                    v,oldCarrier,newCarrier,oldOrientation,newOrientation);
            };
            // Carry the optional velocity-filter history into the next
            // attachment orientation. Do not erase the physical flow history.
            freeze.carrierFilter.TransformSamples([&](icicle_growth::Vec3 v) {
                return icicle_growth::RotateBetweenFrames(
                    v,oldOrientation,newOrientation);
            });
            // Only the derivatives across a discontinuous FBX seam are invalid.
            // Keep motionGrowth.localVelocity, directionResponse, growthBuffer,
            // frozen mass and all existing localPath segments unchanged.
            freeze.parcelCloud.Rebase(oldCarrier,newCarrier,oldOrientation,newOrientation);
            freeze.motionGrowth.ResetKinematics();
            freeze.liquid.transitVelocityWorld=rebase(freeze.liquid.transitVelocityWorld);
            freeze.liquid.tipVelocityWorld=rebase(freeze.liquid.tipVelocityWorld);
            freeze.inletVelocityWorld=ViewerVector(rebase(GrowthVector(freeze.inletVelocityWorld)));
            freeze.liquid.lastDirection=icicle_growth::Rotate(newOrientation,
                icicle_growth::Rotate(icicle_growth::Conjugate(oldOrientation),freeze.liquid.lastDirection));
            flow.growth=ViewerVector(freeze.liquid.lastDirection);
            freeze.previousTipVelocity=freeze.tipVelocityDebug=ViewerVector(newCarrier);
            freeze.tipAccelerationDebug=vec3(0);
            freeze.hasPreviousTipVelocity=true;
            freeze.previousTipWorld=selected.pos+selected.frame*freeze.localPath.back();
            freeze.hasPreviousTip=true;
            // A seam is a coordinate jump, not a zero-duration freeze event.
            // Account for the single skipped physics tick in the growth clock;
            // keep the liquid, mass buffer and all solid geometry intact.
            if (advance && dt>0.f) freeze.totalGrowthTime+=dt;
            return;
        }
        const auto currentOrientation=AttachmentPose(selected).orientation;
        const auto previousOrientation=freeze.hasAttachmentPose?
            freeze.previousAttachmentPose.orientation:currentOrientation;
        if (!discontinuity && dt>1e-6f)
        {
            const auto angularVelocity=icicle_growth::AngularVelocity(
                previousOrientation,currentOrientation,dt);
            freeze.attachmentTurnDegrees=degrees(icicle_growth::Length(angularVelocity)*dt);
        }
        const vec3 tip = selected.pos + selected.frame * freeze.localPath.back();
        vec3 tipVelocity(0.0f);
        if (freeze.hasPreviousTip && !discontinuity && dt>1e-6f)
            tipVelocity = (tip-freeze.previousTipWorld)/dt;
        freeze.tipVelocityRawDebug=tipVelocity;
        if (discontinuity) freeze.carrierFilter.Reset();
        else if (gCarrierFilterEnabled)
            tipVelocity=ViewerVector(freeze.carrierFilter.Push(GrowthVector(tipVelocity),
                icicle_growth::CarrierWindowTicks(1.0f/kCarrierWindowSeconds,gMotionSpeed,dt)));
        freeze.tipVelocityDebug=tipVelocity;
        freeze.tipAccelerationDebug=freeze.hasPreviousTipVelocity && !discontinuity && dt>1e-6f?
            (tipVelocity-freeze.previousTipVelocity)/dt:vec3(0);
        freeze.previousTipVelocity=tipVelocity;
        freeze.hasPreviousTipVelocity=true;
        freeze.previousTipWorld = tip;
        freeze.hasPreviousTip = true;
        CheckExistingIceGroundContact(freeze,selected);
        if (discontinuity)
        {
            // A loop reset is a teleport, not a physical velocity impulse.
            freeze.directionResponse.Reset();
            freeze.liquid.totalDripped+=freeze.liquid.tipMass;
            freeze.liquid.tipMass=0;
            freeze.parcelCloud.Reset();
            freeze.motionGrowth.ResetKinematics();
            freeze.liquid.transitVelocityWorld = GrowthVector(selected.velocity);
            freeze.liquid.tipVelocityWorld = GrowthVector(selected.velocity);
            freeze.inletVelocityWorld = selected.velocity;
            return;
        }
        if (!advance || !(dt>0) || freeze.groundLocked) return;
        const double stepStart=freeze.totalGrowthTime;
        freeze.totalGrowthTime += dt;
        freeze.inletVelocityWorld = ViewerVector(supply.velocityWorld);
        vec3 directionMoment(0.0f);
        double frozenNow = 0;
        vector<icicle_growth::GrowthChunk> chunks;
        if (gGrowthModel == GrowthModel::LegacyRootWind)
        {
            freeze.tipWater += static_cast<float>(supply.mass);
            frozenNow = std::min(static_cast<double>(freeze.tipWater),
                std::max(0.0,static_cast<double>(freezeRate))*freeze.tipWater*dt);
            freeze.tipWater = static_cast<float>(freeze.tipWater-frozenNow);
            directionMoment = flow.growth*static_cast<float>(frozenNow);
        }
        else
        {
            auto parameters = gLiquidParameters;
            if (!airEnabled) parameters.airCouplingRate=0;
            // Pending frozen mass already represents solid length. Including
            // it keeps liquid travel time independent of polyline emission.
            const float physicalLength=static_cast<float>(
                (freeze.axialFrozenMass+freeze.frozenAtTip)/gGrowthResolution.linearMassDensity);
            icicle_growth::ParcelParameters parcelParameters;
            parcelParameters.couplingSeconds=gMotionGrowthParameters.relaxationSeconds;
            const auto step = gUseAccumulatedSteeringDirection
                ? icicle_growth::AdvanceParcels(freeze.parcelCloud,freeze.liquid,supply,dt,
                    physicalLength,GrowthVector(tipVelocity),GrowthVector(flow.gravity),
                    freezeRate,parameters,parcelParameters)
                : icicle_growth::AdvanceLiquid(
                freeze.liquid,supply,dt,physicalLength,GrowthVector(tipVelocity),
                GrowthVector(flow.gravity),GrowthVector(flow.wind),freezeRate,parameters);
            frozenNow = step.frozenMass;
            freeze.lastDetachedMass=step.detachedMass;
            freeze.lastDirectionCoherence=step.minimumDirectionCoherence;
            freeze.unstableDirectionSteps=step.unstableDirectionSteps;
            directionMoment = ViewerVector(step.frozenDirectionMoment);
            freeze.tipWater = static_cast<float>(freeze.liquid.tipMass);
            flow.growth = ViewerVector(freeze.liquid.lastDirection);
            auto resolution=gGrowthResolution;
            resolution.maxSegmentLength=freeze.segmentLength;
            chunks.reserve(step.samples.size()+1);
            for (const auto& sample:step.samples)
            {
                const auto orientation=icicle_growth::Interpolate(previousOrientation,
                    currentOrientation,std::clamp(sample.timeOffset/dt,0.f,1.f));
                auto directionLocal=icicle_growth::Rotate(
                    icicle_growth::Conjugate(orientation),sample.directionWorld);
                const auto fallbackLocal=icicle_growth::Rotate(
                    icicle_growth::Conjugate(orientation),
                    icicle_growth::Unit(GrowthVector(flow.gravity)));
                const auto respondedDirection=icicle_growth::RelaxGrowthDirection(
                    freeze.directionResponse,directionLocal,sample.relativeSpeed,
                    sample.duration,fallbackLocal,gDirectionResponse);
                freeze.rawGrowthLocal=ViewerVector(directionLocal);
                freeze.respondedGrowthLocal=ViewerVector(respondedDirection);
                icicle_growth::AccumulateFrozenSample(freeze.growthBuffer,sample.mass,
                    respondedDirection,stepStart+sample.timeOffset,sample.duration,resolution,chunks);
            }

        }
        freeze.lastFrozenMass=frozenNow;
        freeze.frozenAtTip += frozenNow;
        freeze.sumLocal += transpose(selected.frame)*directionMoment;
        freeze.weight += frozenNow;
        if (freeze.existingIceContact)
        {
            DepositPendingWithoutElongation(freeze);
        }
        else
        {
            if (gGrowthModel==GrowthModel::LiquidFlow)
                AppendGrowthChunks(freeze,selected,chunks);
            else {
                const vec3 fallbackLocal=SafeNormalize(transpose(selected.frame)*flow.growth);
                AppendFrozenSegments(freeze,selected,fallbackLocal,kUseInstantGrowthDirection);
            }
        }
        // Record the post-growth tip: next frame's velocity measures only
        // rigid attachment motion, not this frame's added segment length.
        freeze.previousTipWorld = selected.pos + selected.frame*freeze.localPath.back();
    }

    struct GroundLockLedger
    {
        size_t growthContacts=0,motionContacts=0,archivedCount=0;
        double archivedIce=0,archivedWater=0,archivedInput=0,archivedRunoff=0;
        double maxArchivedMassError=0;
    };

    bool LockIcicleAtPose(IcicleState& icicle,icicle_growth::RigidPose pose,
                          GroundLockLedger& ledger,bool fromGrowth)
    {
        auto& freeze=icicle.freeze;
        if (freeze.groundLocked || !icicle.root.valid) return false;
        // Freeze the entire icicle in WORLD space. Its previous character
        // attachment no longer drives this root, liquid state, or local frame.
        auto& root=icicle.root;
        root.pos=root.prevPos=ViewerVector(pose.position);
        root.frame=mat3(ViewerVector(icicle_growth::Rotate(pose.orientation,{1,0,0})),
            ViewerVector(icicle_growth::Rotate(pose.orientation,{0,1,0})),
            ViewerVector(icicle_growth::Rotate(pose.orientation,{0,0,1})));
        root.normal=ViewerVector(icicle_growth::Rotate(pose.orientation,{0,0,1}));
        root.velocity=root.rawVelocity=vec3(0);
        root.speed=root.normalDegSec=0;
        root.hasPrev=false;
        freeze.groundLocked=true;
        freeze.active=false;
        freeze.groundContact=true;
        freeze.existingIceContact=false;
        freeze.previousAttachmentPose=pose;
        freeze.hasAttachmentPose=true;
        vec3 contact=root.pos;
        float minimum=numeric_limits<float>::max();
        for (const auto& p:freeze.localPath) {
            const vec3 world=root.pos+root.frame*p;
            if (world.y<minimum) { minimum=world.y; contact=world; }
        }
        // A changed ground/teleport may start below the plane: translate this
        // icicle as one rigid object, without lifting the animated character.
        const float lift=std::max(0.f,gGround.height+gGround.radius-minimum);
        root.pos.y+=lift; root.prevPos=root.pos;
        contact.y+=lift;
        freeze.previousAttachmentPose.position=GrowthVector(root.pos);
        freeze.previousTipWorld=root.pos+root.frame*freeze.localPath.back();
        freeze.contactWorld=contact;
        if (fromGrowth) ++ledger.growthContacts; else ++ledger.motionContacts;
        return true;
    }

    int HandleIcicleGroundContacts(vector<IcicleState>& icicles,
        GroundLockLedger& ledger,float dt,bool discontinuity)
    {
        if (!gGround.enabled) return 0;
        int locked=0;
        for (auto& icicle:icicles)
        {
            auto& freeze=icicle.freeze;
            if (freeze.groundLocked || !icicle.root.valid || freeze.localPath.size()<2) continue;
            vector<icicle_growth::Vec3> local;
            local.reserve(freeze.localPath.size());
            for (const auto& p:freeze.localPath) local.push_back(GrowthVector(p));
            const auto current=AttachmentPose(icicle.root);
            const bool history=freeze.hasAttachmentPose && !discontinuity && dt>1e-6f;
            const auto previous=history?freeze.previousAttachmentPose:current;
            const auto contact=icicle_growth::SweepRigidPath(
                local,previous,current,gGround.height,gGround.radius);
            if (contact.hit && LockIcicleAtPose(icicle,contact.pose,ledger,false)) ++locked;
        }
        return locked;
    }

    void ArchiveGroundedIcicles(vector<IcicleState>& icicles,GroundLockLedger& ledger,
                                size_t keepCount=kMaxGroundedIcicles)
    {
        size_t grounded=static_cast<size_t>(count_if(icicles.begin(),icicles.end(),
            [](const auto& i){return i.freeze.groundLocked;}));
        while (grounded>keepCount)
        {
            const auto oldest=find_if(icicles.begin(),icicles.end(),
                [](const auto& i){return i.freeze.groundLocked;});
            const auto& f=oldest->freeze;
            const double ice=f.axialFrozenMass+f.nonAxialFrozenMass+f.frozenAtTip;
            const bool liquidModel=gGrowthModel==GrowthModel::LiquidFlow;
            const double water=liquidModel?f.liquid.transitMass+f.liquid.tipMass:f.tipWater;
            const double input=liquidModel?f.liquid.totalInput:ice+water;
            const double runoff=liquidModel?f.liquid.totalDripped:0;
            ledger.archivedIce+=ice; ledger.archivedWater+=water;
            ledger.archivedInput+=input; ledger.archivedRunoff+=runoff;
            ledger.maxArchivedMassError=std::max(ledger.maxArchivedMassError,
                std::abs(input-ice-water-runoff));
            ++ledger.archivedCount;
            icicles.erase(oldest); // budget/manual cleanup; preserve mass totals
            --grounded;
        }
    }

    int SpawnEligibleIcicles(
        vector<IcicleState>& icicles,
        const WaterTransportSimulation& water,
        const vector<vec3>& worldPositions,
        const vector<unsigned int>& triangleIndices,
        const vector<int>& vertexTriangle,
        const vector<int>& vertexCorner,
        const FlowState& globalFlow,
        float animationTime,size_t rootLimit=kMaxIcicleRoots)
    {
        struct Candidate
        {
            size_t vertex;
            float dripMass;
        };

        size_t growingCount=static_cast<size_t>(count_if(icicles.begin(),icicles.end(),
            [](const auto& i){return !i.freeze.groundLocked;}));
        if (growingCount>=rootLimit) return 0;
        vector<Candidate> candidates;

        const auto& states = water.Vertices();

        if (states.size() != worldPositions.size())
            return 0;

        // 1. 생성 가능한 Drip Candidate 수집
        for (size_t i = 0; i < states.size(); ++i)
        {
            if (!states[i].dripCandidate)
                continue;

            if (states[i].dripMass <=
                kMinIcicleDripMass)
                continue;

            candidates.push_back({
                i,
                states[i].dripMass
                });
        }

        // 2. 이번 스텝의 Drip Mass가 큰 순서대로 정렬
        sort(
            candidates.begin(),
            candidates.end(),
            [](const Candidate& a, const Candidate& b)
            {
                if (a.dripMass != b.dripMass)
                    return a.dripMass > b.dripMass;

                return a.vertex < b.vertex;
            });

        int createdCount = 0;

        // 3. 후보별 생성 조건 검사
        for (const Candidate& candidate : candidates)
        {
            if (growingCount>=rootLimit || createdCount>=kMaxNewRootsPerStep) break;
            const size_t vertex = candidate.vertex;

            if (vertex >= vertexTriangle.size())
                continue;

            // 현재 Vertex에 대응하는 Triangle 확인
            const int triangleID =
                vertexTriangle[vertex];

            const int corner =
                vertexCorner[vertex];

            if (triangleID < 0 || corner < 0)
                continue;

            const vec3 candidatePosition =
                worldPositions[vertex];

            // 4. 기존 Root와 너무 가까운지 확인
            bool rootAlreadyExists = false;

            for (const IcicleState& existing : icicles)
            {
                if (!existing.root.valid)
                    continue;

                const vec3 delta=candidatePosition-existing.root.pos;
                // Match the capture allocator's vertex-normal criterion so a
                // blocked nearby source can actually feed this same root.
                vec3 captureNormal=existing.root.normal;
                const auto& idx=existing.root.idx;
                if (!existing.freeze.groundLocked && idx[0]<states.size() &&
                    idx[1]<states.size() && idx[2]<states.size())
                    captureNormal=SafeNormalize(existing.root.bary.x*states[idx[0]].normal+
                        existing.root.bary.y*states[idx[1]].normal+
                        existing.root.bary.z*states[idx[2]].normal,captureNormal);
                if (dot(delta,delta)<kMinRootSpacing*kMinRootSpacing &&
                    dot(states[vertex].normal,captureNormal)>=0.5f)
                {
                    rootAlreadyExists = true;
                    break;
                }
            }

            if (rootAlreadyExists)
                continue;

            // 5. 새로운 Root 생성
            const size_t base =
                static_cast<size_t>(triangleID) * 3;

            if (base + 2 >= triangleIndices.size())
                continue;

            IcicleState icicle;

            icicle.root.valid = true;

            icicle.root.tri =
                static_cast<size_t>(triangleID);

            icicle.root.idx = {
                triangleIndices[base + 0],
                triangleIndices[base + 1],
                triangleIndices[base + 2]
            };

            // Vertex 기반 Barycentric Coordinate
            if (corner == 0)
            {
                icicle.root.bary =
                    vec3(1.0f, 0.0f, 0.0f);
            }
            else if (corner == 1)
            {
                icicle.root.bary =
                    vec3(0.0f, 1.0f, 0.0f);
            }
            else
            {
                icicle.root.bary =
                    vec3(0.0f, 0.0f, 1.0f);
            }

            // 6. 현재 Animated Mesh 위치에서 Root 초기화
            UpdateSelected(
                icicle.root,
                worldPositions,
                0.0f,
                true);

            // 7. 해당 Root의 환경 및 성장 방향 초기화
            icicle.flow = globalFlow;

            UpdateFlow(
                icicle.flow,
                icicle.root);

            // 8. 기존 결빙 모델 사용
            icicle.freeze.segmentLength = gGrowthResolution.maxSegmentLength;

            StartFreeze(
                icicle.freeze,
                icicle.root,
                animationTime);

            // 9. 독립적인 Icicle State 저장
            icicles.push_back(std::move(icicle));

            ++createdCount;
            ++growingCount;

        }
        if (createdCount>0)
            cout << "New roots +" << createdCount << " | total " << icicles.size()
                 << " | growing " << growingCount << "/" << rootLimit << "\n";

        return createdCount;
    }

    // For a controlled one-root comparison the right-clicked surface point
    // must remain the SAME attachment across speed/freeze/dt experiments.
    // Auto-selecting the highest-drip vertex each run would change the input
    // location and confound the motion-speed comparison.
    bool SpawnPinnedSingleRoot(vector<IcicleState>& icicles,
        const SelectedSurface& pinned, const WaterTransportSimulation& water,
        const vector<vec3>& worldPositions, const FlowState& globalFlow,
        float animationTime)
    {
        if (!icicles.empty() || !pinned.valid) return false;
        const auto& states=water.Vertices();
        if (pinned.idx[0]>=states.size() || pinned.idx[1]>=states.size() ||
            pinned.idx[2]>=states.size()) return false;
        if (states.size()!=worldPositions.size()) return false;
        const vec3 targetNormal=SafeNormalize(
            pinned.bary.x*states[pinned.idx[0]].normal+
            pinned.bary.y*states[pinned.idx[1]].normal+
            pinned.bary.z*states[pinned.idx[2]].normal,pinned.normal);
        // Require an actual nearby water-drip event, not merely a mouse click.
        // Match the allocator's radius and normal rule, including triangle
        // vertices that supply the root exactly.
        bool supplied=false;
        for (size_t i=0;i<states.size();++i)
        {
            if (states[i].dripMass<=kMinIcicleDripMass) continue;
            bool exact=false;
            for (int k=0;k<3;++k)
                exact=exact || (pinned.idx[k]==i && pinned.bary[k]>0);
            if (!exact && dot(states[i].normal,targetNormal)<0.5f) continue;
            const vec3 delta=worldPositions[i]-pinned.pos;
            if (dot(delta,delta)<=kMinRootSpacing*kMinRootSpacing) {
                supplied=true;
                break;
            }
        }
        if (!supplied) return false;
        IcicleState icicle;
        icicle.root=pinned;
        icicle.flow=globalFlow;
        UpdateFlow(icicle.flow,icicle.root);
        icicle.freeze.segmentLength=gGrowthResolution.maxSegmentLength;
        StartFreeze(icicle.freeze,icicle.root,animationTime);
        icicles.push_back(std::move(icicle));
        cout << "Pinned single root on triangle " << pinned.tri
             << " at barycentric " << V3(pinned.bary) << "\n";
        return true;
    }

    vector<vec3> WorldPath(
        const FreezeState& freeze,
        const SelectedSurface& selected)
    {
        vector<vec3> result;

        if (!selected.valid)
            return result;

        result.reserve(freeze.localPath.size());

        for (const vec3& point : freeze.localPath)
            result.push_back(selected.pos + selected.frame * point);

        return result;
    }

    void PrintState(
        const Model& model,
        const Animation* animation,
        const Animator* animator,
        const SelectedSurface& selected,
        const FlowState& flow,
        const FreezeState& freeze)
    {
        cout << "\n=== Research State ===\n"
            << "V " << model.GetVertexCount()
            << " | T " << model.GetTriangleCount()
            << " | B " << model.GetBoneCount()
            << "\n";

        if (animation)
        {
            cout << "TPS " << animation->GetTicksPerSecond()
                << " | channels " << animation->GetChannelCount()
                << "\n";
        }

        if (animator)
        {
            const float time = animator->GetCurrentTimeSeconds();
            cout << "time " << time
                << " sec | frame30 "
                << static_cast<int>(round(time * kDisplayFps))
                << "\n";
        }

        if (selected.valid)
        {
            cout << "tri " << selected.tri
                << " idx " << selected.idx[0]
                << "," << selected.idx[1]
                << "," << selected.idx[2]
                << " bary " << V3(selected.bary)
                << "\np0 " << V3(selected.p0)
                << " p1 " << V3(selected.p1)
                << " p2 " << V3(selected.p2)
                << "\ndrip " << V3(selected.pos)
                << " normal " << V3(selected.normal)
                << "\nvelocity " << V3(selected.velocity)
                << " speed " << selected.speed
                << " normalChange " << selected.normalDegSec
                << " deg/s\n";
        }
        else
        {
            cout << "no selected triangle\n";
        }

        cout << "wind " << V3(flow.wind)
            << " relative " << V3(flow.relative)
            << " growth " << V3(flow.growth)
            << "\n";

        cout << "tipWater "
            << freeze.tipWater
            << " | frozenAtTip "
            << freeze.frozenAtTip
            << " | growthTime "
            << freeze.totalGrowthTime
            << " | segments "
            << freeze.localPath.size() - 1
            << "\n";

        cout << "model " << (gGrowthModel==GrowthModel::LiquidFlow ? "LiquidFlow" : "Legacy")
            << " | inletWorld " << V3(freeze.inletVelocityWorld)
            << " | transitWater " << freeze.liquid.transitMass
            << " | liquidMassError " << freeze.liquid.MassError()
            << " | axialIce " << freeze.axialFrozenMass
            << " | nonAxialIceUnrendered " << freeze.nonAxialFrozenMass
            << " | groundGrowthContact " << freeze.groundContact
            << " | existingIceContact " << freeze.existingIceContact
            << " | bodyLift " << gGround.characterLift
            << " | groundLocked " << freeze.groundLocked << "\n";
        cout << "======================\n";
    }

    void WriteTraceHeader(ostream& out)
    {
        out << "simulation_time,animation_time,cycle,discontinuity,model,triangle,vertex,"
               "root_x,root_y,root_z,tip_vx,tip_vy,tip_vz,tip_ax,tip_ay,tip_az,"
               "inlet_vx,inlet_vy,inlet_vz,water_vx,water_vy,water_vz,"
               "growth_x,growth_y,growth_z,input_mass,frozen_this_step,total_frozen,"
               "axial_ice,pending_ice,transit_water,tip_water,path_length,segments,ground_locked,"
               "animation_wrapped,wind_enabled,response_enabled,response_tau,"
               "attachment_turn_degrees,new_path_length,"
               "raw_local_x,raw_local_y,raw_local_z,"
               "response_local_x,response_local_y,response_local_z,"
               "emitted_local_x,emitted_local_y,emitted_local_z,freeze_rate_scale,motion_speed,tau_ref,temperature_C,flow_motion_gain,physics_dt,effective_freeze_rate,linear_mass_density,"
               "tip_raw_vx,tip_raw_vy,tip_raw_vz,carrier_filter_enabled,"
               "integrated_motion,one_shot,motion_memory_seconds,"
               "integrated_local_vx,integrated_local_vy,integrated_local_vz,"
               "prewet_enabled,animation_playing,waiting_for_ice,"
               "detached_mass,direction_coherence,unstable_direction_steps,"
               "turn_limited_segments,pending_growth_length\n";
    }

    void WriteTraceRoot(ostream& out,double simulationTime,float animationTime,
                        size_t cycle,bool discontinuity,const IcicleState& icicle,
                        bool animationWrapped,bool windEnabled,
                        bool animationPlaying,bool waitingForIce)
    {
        const auto& root=icicle.root;
        const auto& f=icicle.freeze;
        size_t corner=root.bary.y>root.bary.x?1:0;
        if (root.bary.z>(corner==0?root.bary.x:root.bary.y)) corner=2;
        const bool liquid=gGrowthModel==GrowthModel::LiquidFlow;
        const double totalFrozen=liquid?f.liquid.totalFrozen:
            f.axialFrozenMass+f.nonAxialFrozenMass+f.frozenAtTip;
        const double input=liquid?f.liquid.totalInput:totalFrozen+f.tipWater;
        out << setprecision(10) << simulationTime << ',' << animationTime << ',' << cycle
            << ',' << discontinuity << ',' << (gGrowthModel==GrowthModel::LiquidFlow?"LiquidFlow":"Legacy")
            << ',' << root.tri << ',' << root.idx[corner];
        const auto vectorOut=[&out](const vec3& v){out << ',' << v.x << ',' << v.y << ',' << v.z;};
        vectorOut(root.pos); vectorOut(f.tipVelocityDebug); vectorOut(f.tipAccelerationDebug);
        vectorOut(f.inletVelocityWorld); vectorOut(ViewerVector(f.liquid.tipVelocityWorld));
        vectorOut(icicle.flow.growth);
        out << ',' << input << ',' << f.lastFrozenMass << ',' << totalFrozen
            << ',' << f.axialFrozenMass << ',' << f.frozenAtTip << ',' << f.liquid.transitMass
            << ',' << (liquid?f.liquid.tipMass:f.tipWater) << ',' << f.pathLength << ',' << f.localPath.size()-1
            << ',' << f.groundLocked << ',' << animationWrapped << ',' << windEnabled
            << ',' << gDirectionResponse.enabled << ',' << gDirectionResponse.responseTimeSeconds
            << ',' << f.attachmentTurnDegrees << ',' << f.addedPathLength;
        vectorOut(f.rawGrowthLocal);
        vectorOut(f.respondedGrowthLocal);
        vectorOut(f.emittedGrowthLocal);
        out << ',' << gFreezeRateScale << ',' << gMotionSpeed << ',' << gFreezing.referenceTimeSeconds
            << ',' << gFreezing.temperatureC << ',' << gLiquidParameters.flowMotionGain
            << ',' << gPhysicsStep << ',' << icicle_growth::EffectiveFreezeRate(gFreezing)*gFreezeRateScale
            << ',' << gGrowthResolution.linearMassDensity;
        vectorOut(f.tipVelocityRawDebug);
        out << ',' << gCarrierFilterEnabled << ',' << gUseAccumulatedSteeringDirection
            << ',' << gPlayKickOnce << ',' << gMotionGrowthParameters.relaxationSeconds;
        vectorOut(ViewerVector(f.motionGrowth.localVelocity));
        out << ',' << gPreWetBeforeKick << ',' << animationPlaying
            << ',' << waitingForIce
            << ',' << f.lastDetachedMass
            << ',' << f.lastDirectionCoherence
            << ',' << f.unstableDirectionSteps
            << ',' << f.turnLimitedSegments
            << ',' << f.growthBuffer.mass/std::max(1e-12,gGrowthResolution.linearMassDensity)
            << '\n';
    }

    void DrawDebug(
        DebugRenderer& renderer,
        const mat4& vp,
        const vec3& cameraPosition,
        const SelectedSurface& selected,
        const FlowState& flow,
        const FreezeState& freeze, bool showVectors=true)
    {
        if (!selected.valid)
            return;

        constexpr float offset = 0.006f;

        const vec3 a = selected.p0 + selected.normal * offset;
        const vec3 b = selected.p1 + selected.normal * offset;
        const vec3 c = selected.p2 + selected.normal * offset;
        const vec3 d = selected.pos + selected.normal * offset * 1.5f;

        if (gDebug && showVectors && !freeze.groundLocked)
        {
            renderer.Tri(a, b, c, vec3(1.0f, 0.85f, 0.15f), vp);
            renderer.Point(d, vec3(1.0f, 0.18f, 0.18f), vp, 12.0f);
            renderer.Arrow(d, selected.normal * 0.24f, vec3(0.2f, 0.6f, 1.0f), vp, cameraPosition);
            renderer.Arrow(d, selected.velocity * 0.15f, vec3(1.0f, 0.48f, 0.12f), vp, cameraPosition);
            renderer.Arrow(d, flow.wind * 0.22f, vec3(0.2f, 0.85f, 1.0f), vp, cameraPosition);
            renderer.Arrow(d, flow.relative * 0.15f, vec3(0.2f, 1.0f, 0.42f), vp, cameraPosition);
            renderer.Arrow(d, flow.growth * 0.34f, vec3(1.0f, 0.2f, 0.55f), vp, cameraPosition);
        }

        const vector<vec3> path = WorldPath(freeze, selected);
        if (path.size() > 1)
            renderer.Polyline(path, vec3(0.75f, 0.9f, 1.0f), vp);
        if (gDebug && showVectors && !freeze.groundLocked)
            renderer.Arrow(d, freeze.inletVelocityWorld*0.15f,
                vec3(0.65f,0.35f,1.0f),vp,cameraPosition);
        if (gGround.enabled && (freeze.groundContact || freeze.existingIceContact))
            renderer.Point(freeze.contactWorld,
                freeze.groundLocked ? vec3(1.0f,0.65f,0.1f) : vec3(1.0f,0.75f,0.1f),
                vp,10.0f);
    }

} // namespace

void DrawGrowthGround(DebugRenderer& renderer,const mat4& vp)
{
    if (!gGround.enabled || !gGround.initialized) return;
    const float h=gGround.halfSize, y=gGround.height;
    const vec3 vertices[6]={{-h,y,-h},{-h,y,h},{h,y,h},
                            {-h,y,-h},{h,y,h},{h,y,-h}};
    renderer.Draw(GL_TRIANGLES,vertices,6,vec3(0.09f,0.11f,0.13f),vp);
    for (int i=-20;i<=20;++i)
    {
        const float offset=h*static_cast<float>(i)/20.0f;
        renderer.Line(vec3(-h,y+0.0005f,offset),vec3(h,y+0.0005f,offset),
            vec3(0.18f,0.21f,0.24f),vp,1.0f);
        renderer.Line(vec3(offset,y+0.0005f,-h),vec3(offset,y+0.0005f,h),
            vec3(0.18f,0.21f,0.24f),vp,1.0f);
    }
}

vec3 WaterHeatColor(float t)
{
    t = ClampValue(t, 0.0f, 1.0f);

    if (t < 0.5f)
    {
        const float u = t * 2.0f;
        return mix(
            vec3(0.08f, 0.35f, 1.0f),
            vec3(0.15f, 0.95f, 1.0f),
            u);
    }

    const float u = (t - 0.5f) * 2.0f;
    return mix(
        vec3(0.15f, 0.95f, 1.0f),
        vec3(1.0f, 0.2f, 0.08f),
        u);
}

float TotalIceForDisplay(const WaterTransportSimulation& water)
{
    float total = 0.0f;

    for (const WaterVertexState& state : water.Vertices())
        total += state.ice;

    return total;
}

void DrawWaterOverlay(
    DebugRenderer& renderer,
    const mat4& vp,
    const vec3& cameraPosition,
    const vector<vec3>& positions,
    const WaterTransportSimulation& water)
{
    const auto& states = water.Vertices();

    if (states.size() != positions.size())
        return;

    const float maxWater = water.MaxWater();

    constexpr float kWaterDisplayThreshold = 0.001f;
    constexpr float kWaterDisplayMax = 0.30f;
    constexpr float kIceThreshold = 0.003f;

    // Liquid Water debug visualization
    // 현재 물이 아주 적어도 보이도록 threshold를 매우 낮추고,
    // 표면보다 조금 더 띄운 큰 파란 점으로 표시한다.
    // Liquid Water State visualization
    if (gShowWater && maxWater > 1e-8f)
    {
        constexpr int bins = 8;
        array<vector<vec3>, bins> groups;

        for (size_t i = 0; i < states.size(); ++i)
        {
            if (states[i].water <= kWaterDisplayThreshold)
                continue;

            float t = states[i].water / kWaterDisplayMax;

            t = ClampValue(t, 0.0f, 1.0f);

            // Water 분포 차이 강조
            t = pow(t, 0.35f);

            int bin =
                static_cast<int>(
                    t * static_cast<float>(bins));

            bin = MinValue(bin, bins - 1);

            groups[static_cast<size_t>(bin)].push_back(
                positions[i] +
                states[i].normal * 0.018f);
        }

        for (int bin = 0; bin < bins; ++bin)
        {
            auto& group =
                groups[static_cast<size_t>(bin)];

            if (group.empty())
                continue;

            const float t =
                (static_cast<float>(bin) + 0.5f) /
                static_cast<float>(bins);

            renderer.Draw(
                GL_POINTS,
                group.data(),
                static_cast<GLsizei>(group.size()),
                WaterHeatColor(t),
                vp,
                1.0f,
                5.0f);
        }
    }

    // 47주차: Ice State visualization
    if (gShowIce)
    {
        vector<vec3> icePoints;
        icePoints.reserve(states.size());

        for (size_t i = 0; i < states.size(); ++i)
        {
            if (states[i].ice < kIceThreshold)
                continue;

            icePoints.push_back(
                positions[i] +
                states[i].normal * 0.014f);
        }

        if (!icePoints.empty())
        {
            renderer.Draw(
                GL_POINTS,
                icePoints.data(),
                static_cast<GLsizei>(icePoints.size()),
                vec3(0.92f, 0.97f, 1.0f),
                vp,
                1.0f,
                10.0f);
        }
    }

    // Drip Candidate visualization
    if (gShowDrips)
    {
        vector<vec3> dripPoints;
        dripPoints.reserve(water.DripCandidateCount());

        for (size_t i = 0; i < states.size(); ++i)
        {
            if (states[i].dripCandidate)
            {
                dripPoints.push_back(
                    positions[i] +
                    states[i].normal * 0.020f);
            }
        }

        if (!dripPoints.empty())
        {
            renderer.Draw(
                GL_POINTS,
                dripPoints.data(),
                static_cast<GLsizei>(dripPoints.size()),
                vec3(1.0f, 0.12f, 0.72f),
                vp,
                1.0f,
                11.0f);
        }
    }

    // 1-ring strongest transition direction
    if (gShowFlow && !states.empty())
    {
        const size_t stride =
            MaxValue<size_t>(1, states.size() / 300);

        for (size_t i = 0; i < states.size(); i += stride)
        {
            if (states[i].strongestNeighbor < 0)
                continue;

            const unsigned int neighbor =
                static_cast<unsigned int>(
                    states[i].strongestNeighbor);

            if (neighbor >= positions.size())
                continue;

            const vec3 start =
                positions[i] +
                states[i].normal * 0.010f;

            const vec3 delta =
                (positions[neighbor] -
                    positions[i]) * 1.2f;

            renderer.Arrow(
                start,
                delta,
                vec3(0.15f, 1.0f, 0.35f),
                vp,
                cameraPosition);
        }
    }
}

void ApplyWaterPreset(
    int preset,
    Animator* animator,
    FlowState& flow,
    WaterTransportSimulation& water,
    bool& autoIcicleGeneration)
{
    gWaterPaused = false;
    autoIcicleGeneration = true;

    if (animator)
    {
        animator->Reset();
        animator->SetPlaying((preset == 2 || preset == 4) && !gPreWetBeforeKick);
    }

    
    water.ResetWater();

    flow.wind = vec3(0.0f);
    flow.wm = 0.0f;
    water.SetWindInfluenceEnabled(false);
    water.SetRainEnabled(true);

    cout << "Motion-only preset " << preset << " : "
         << (preset == 1 ? "Static / No Wind" : "Animation / No Wind")
         << "\n";
}

int main(int argc, char** argv)
{
    try
    {
        const string modelPath =
            argc >= 2 ? argv[1] : "assets/rotation.fbx";

        if (!filesystem::exists(modelPath))
        {
            cerr << "FBX not found: " << modelPath << "\n";
            return 1;
        }

        if (!glfwInit())
            throw runtime_error("GLFW init failed");

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

        GLFWwindow* window = glfwCreateWindow(
            kWindowWidth,
            kWindowHeight,
            "Mixamo Research Viewer",
            nullptr,
            nullptr);

        if (!window)
        {
            glfwTerminate();
            throw runtime_error("Window failed");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(0); // fixed simulation pacing, independent of display refresh

        glfwSetFramebufferSizeCallback(window, FramebufferSizeCallback);
        glfwSetMouseButtonCallback(window, MouseButtonCallback);
        glfwSetCursorPosCallback(window, CursorPositionCallback);
        glfwSetScrollCallback(window, ScrollCallback);

        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK)
            throw runtime_error("GLEW init failed");

        glGetError();
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_CULL_FACE);
        glCullFace(GL_BACK);

        Shader shader(kVertexShader, kFragmentShader);
        shader.BindUniformBlock("BoneMatrices", 0);

        unsigned int boneUBO = 0;
        glGenBuffers(1, &boneUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, boneUBO);
        glBufferData(
            GL_UNIFORM_BUFFER,
            kMaxShaderBones * sizeof(mat4),
            nullptr,
            GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_UNIFORM_BUFFER, 0, boneUBO);
        glBindBuffer(GL_UNIFORM_BUFFER, 0);

        DebugRenderer debugRenderer;
        Model model(modelPath);

        unique_ptr<Animation> animation;
        unique_ptr<Animator> animator;

        try
        {
            // FBX files can contain an empty Take 001 followed by the actual
            // Mixamo clip. Prefer the first stack with usable channels.
            // Leave other existing models on stack 0 when it is already valid.
            unsigned int selectedAnimationIndex = 0;
            for (unsigned int candidateIndex = 0; candidateIndex < 2; ++candidateIndex)
            {
                try
                {
                    auto candidate = make_unique<Animation>(modelPath, &model, candidateIndex);
                    if (candidate->GetChannelCount() == 0) continue;
                    selectedAnimationIndex = candidateIndex;
                    animation = std::move(candidate);
                    break;
                }
                catch (const exception&)
                {
                    // An empty stack may fail to load; try the next one.
                }
            }
            if (!animation)
                throw runtime_error("FBX has no usable animation in stacks 0 or 1");
            animator = make_unique<Animator>(animation.get());
            cout << "Selected animation stack " << selectedAnimationIndex
                 << " | " << animation->GetName()
                 << " | channels " << animation->GetChannelCount() << "\n";
        }
        catch (const exception& e)
        {
            cerr << "Animation unavailable: " << e.what() << "\n";
        }

        cout << "Loaded " << modelPath
            << "\nMeshes " << model.GetMeshCount()
            << "\nVertices " << model.GetVertexCount()
            << "\nTriangles " << model.GetTriangleCount()
            << "\nSkin bones " << model.GetSkinBoneCount()
            << "\nBone palette " << model.GetBoneCount()
            << "\n";

        if (animation)
        {
            gAnimationTicksPerSecond=static_cast<float>(animation->GetTicksPerSecond());
            cout << "Animation " << animation->GetName()
                << "\nChannels " << animation->GetChannelCount()
                << "\nSkeleton nodes " << animation->GetSkeletonNodeCount()
                << "\nTicks/sec " << animation->GetTicksPerSecond()
                << "\nDuration " << animator->GetDurationSeconds()
                << " sec\n";
        }

        cout
            << "Controls: LeftDrag orbit | Wheel zoom | RightClick select | "
            << "Space animation play/pause | . step | F auto-spawn roots | C clear | "
            << "P print | D vectors | W wire | R animation reset | "
            << "Wind keys disabled | [/] ground height | 5 ground | 6 disabled | "
            << "7 single root (right-click to pin) | 8 CSV trace | 9 clear grounded icicles\n";

        cout
            << "Water: 1 Static/NoWind | 2 Anim/NoWind | 3 Static/NoWind | "
            << "4 Anim/NoWind | T rain | X reset water+ice | Y pause simulation | "
            << "H water heatmap | Z ice state | G drip points | "
            << "B flow arrows | V disabled\n";
        cout << "PROCEDURAL ACCUMULATED STEERING: Ground OFF, loop ON, pre-grow OFF. Not a fluid prediction.\n";
        cout << "Motion study: F1 speed | F2 freeze time | F3 temperature | F4 flow-motion gain | F5 dt | F6 linear mass density | F7 carrier filter (reset)\n"
             << "Trajectory diagnostic: surface freezing OFF; tip freezing ON. Defaults: speed 1, tauRef .25s at -10C, gain 1, dt 1/120s, wind OFF.\n";
        cout << "F8 response filter | F9 loop/one-shot (diagnostic) | F10 parcels/legacy | F11 liquid coupling | F12 pre-grow (diagnostic), all reset\n";
        cout << "DEFAULT: looping spinning kick + concurrent ice growth; no pre-growth; wind OFF.\n";

        const vec3 center = model.GetBoundsCenter();
        const float modelScale = model.GetRecommendedScale();

        mat4 modelMatrix(1.0f);
        modelMatrix = scale(modelMatrix, vec3(modelScale));
        modelMatrix = translate(modelMatrix, -center);

        const vector<unsigned int> triangleIndices =
            model.GetFlattenedTriangleIndices();

        // 각 Vertex가 포함된 대표 Triangle 저장
        vector<int> vertexTriangle(
            model.GetVertexCount(),
            -1);

        vector<int> vertexCorner(
            model.GetVertexCount(),
            -1);

        for (size_t t = 0;
            t + 2 < triangleIndices.size();
            t += 3)
        {
            const int triangleID =
                static_cast<int>(t / 3);

            for (int corner = 0; corner < 3; ++corner)
            {
                const unsigned int vertex =
                    triangleIndices[t + corner];

                if (vertex >= vertexTriangle.size())
                    continue;

                if (vertexTriangle[vertex] >= 0)
                    continue;

                vertexTriangle[vertex] = triangleID;
                vertexCorner[vertex] = corner;
            }
        }

        const auto growthAdjacency = icicle_growth::BuildAdjacency(
            model.GetVertexCount(),triangleIndices);
        SelectedSurface selected;
        FlowState flow;
        vector<IcicleState> icicles;
        GroundLockLedger groundLockLedger;
        bool autoIcicleGeneration = false; 
        FreezeState freeze;

        WaterTransportSimulation water(
            model.GetVertexCount(),
            triangleIndices);

        water.Params().gravity = flow.gravity;
        water.Params().rainDirection = flow.rain;

        float lastTime = static_cast<float>(glfwGetTime());
        float titleAccumulator = 0.0f;
        float previousAnimationTime = 0.0f;
        bool hasPreviousAnimationTime = false;

        vector<vec3> cachedLocalPositions;
        bool hasCachedDeformation=false;
        unsigned int renderedFrames=0;
        double experimentTime=0;
        size_t animationCycle=0;
        bool waitingForPreWet=false;
        ofstream traceLog;
        string tracePath;

        // Accumulate wall time; every iteration evaluates a fresh animated pose,
        // surface transport and icicle state at ONE fixed simulation step.
        // Rendering remains once per step. Slow machines may run behind wall time.
        double tickAccumulator=0;
        double tickClock=glfwGetTime();
        while (!glfwWindowShouldClose(window))
        {
            const double tickNow=glfwGetTime();
            tickAccumulator+=tickNow-tickClock;
            tickClock=tickNow;
            if (tickAccumulator<gPhysicsStep) {
                glfwWaitEventsTimeout(gPhysicsStep-tickAccumulator);
                continue;
            }
            const float tickDt=gPhysicsStep;
            tickAccumulator-=tickDt;
            const float now = static_cast<float>(glfwGetTime());
            const float frameElapsed=std::max(0.0f,now-lastTime);
            float realDt=tickDt;
            lastTime=now;
            titleAccumulator+=frameElapsed;
            ++renderedFrames;

            Commands command = ProcessKeyboard(
                window,
                animator.get(),
                realDt,
                flow);

            float simulationDt = 0.0f;

            bool geometryResetRequested=false;
            bool groundLayoutChanged=false;
            int requestedPreset=0;
            if (NumberRising(window,1,g1WasDown)) requestedPreset=1;
            if (NumberRising(window,2,g2WasDown)) requestedPreset=2;
            if (NumberRising(window,3,g3WasDown)) requestedPreset=3;
            if (NumberRising(window,4,g4WasDown)) requestedPreset=4;
            if (requestedPreset)
            {
                ApplyWaterPreset(requestedPreset==3?1:requestedPreset==4?2:requestedPreset,animator.get(),flow,water,autoIcicleGeneration);
                waitingForPreWet=(requestedPreset==2 || requestedPreset==4) && gPreWetBeforeKick;
                geometryResetRequested=true;
                icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0;
                autoIcicleGeneration=true;
                selected.hasPrev=false;
                hasPreviousAnimationTime=false;
            }
            if (NumberRising(window,5,g5WasDown))
            {
                gGround.enabled=!gGround.enabled;
                groundLayoutChanged=true;
                cout << "Ground " << (gGround.enabled?"ON":"OFF") << "\n";
            }
            if (Rising(window,GLFW_KEY_F8,gF8ResponseWasDown))
            {
                gDirectionResponse.enabled=!gDirectionResponse.enabled;
                water.ResetWater(); icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0; autoIcicleGeneration=true;
                geometryResetRequested=true;
                selected.hasPrev=false; hasPreviousAnimationTime=false;
                gWaterPaused=false;
                if (animator) { animator->Reset(); animator->SetPlaying(!gPreWetBeforeKick); }
                waitingForPreWet=gPreWetBeforeKick;
                cout << "Direction response " << (gDirectionResponse.enabled?"ON":"OFF")
                     << " | tau=" << gDirectionResponse.responseTimeSeconds
                     << "s; water, ice and animation reset.\n";
            }
            if (NumberRising(window,7,g7WasDown))
            {
                gSingleRootExperiment=!gSingleRootExperiment;
                water.ResetWater(); icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0;
                geometryResetRequested=true;
                selected.hasPrev=false; hasPreviousAnimationTime=false;
                autoIcicleGeneration=true; gWaterPaused=false;
                if (animator) { animator->Reset(); animator->SetPlaying(!gPreWetBeforeKick); }
                waitingForPreWet=gPreWetBeforeKick;
                cout << "Single Root experiment " << (gSingleRootExperiment?"ON":"OFF")
                     << "; right-click a drip region to pin one root, then press 2.\n";
            }
            if (NumberRising(window,8,g8WasDown))
            {
                if (traceLog.is_open()) {
                    traceLog.close(); cout << "Trace saved: " << tracePath << "\n";
                } else {
                    const auto stamp=chrono::duration_cast<chrono::milliseconds>(
                        chrono::system_clock::now().time_since_epoch()).count();
                    const string stem="icicle_trace_"+to_string(stamp);
                    tracePath=stem+".csv";
                    for (size_t suffix=1;filesystem::exists(tracePath);++suffix)
                        tracePath=stem+"_"+to_string(suffix)+".csv";
                    traceLog.open(tracePath);
                    if (traceLog) { WriteTraceHeader(traceLog); cout << "Tracing all roots: " << tracePath << "\n"; }
                    else { traceLog.clear(); cerr << "Cannot open trace file.\n"; }
                }
            }
            if (Rising(window,GLFW_KEY_LEFT_BRACKET,gLBWasDown))
            {
                gGround.height-=0.02f;
                groundLayoutChanged=true;
            }
            if (Rising(window,GLFW_KEY_RIGHT_BRACKET,gRBWasDown))
            {
                gGround.height+=0.02f;
                groundLayoutChanged=true;
            }

            if (Rising(window, GLFW_KEY_T, gTWasDown))
            {
                water.SetRainEnabled(!water.IsRainEnabled());
                cout << "Rain "
                    << (water.IsRainEnabled() ? "ON" : "OFF")
                    << "\n";
            }

            if (Rising(window, GLFW_KEY_X, gXWasDown))
            {
                water.ResetWater();

                icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0;
                waitingForPreWet=false;
                autoIcicleGeneration = false;

                cout << "Water + Ice + Icicles reset\n";
            }

            if (Rising(window, GLFW_KEY_H, gHWasDown)) {
                gShowWater = !gShowWater;
                cout << "Water Heatmap "
                    << (gShowWater ? "ON" : "OFF")
                    << "\n";
            }
                

            if (Rising(window, GLFW_KEY_Z, gZWasDown))
                gShowIce = !gShowIce;

            if (Rising(window, GLFW_KEY_G, gGWasDown))
                gShowDrips = !gShowDrips;

            if (Rising(window, GLFW_KEY_B, gBWasDown))
                gShowFlow = !gShowFlow;

            // No wind in the motion study, including stationary-air drag.
            flow.wind=vec3(0);
            flow.wm=0;
            water.SetWindInfluenceEnabled(false);
            gGrowthModel=GrowthModel::LiquidFlow;
            bool experimentChanged=false;
            if (Rising(window,GLFW_KEY_F1,gExperimentKeys[0])) {
                gMotionSpeed=gMotionSpeed==0.5f?1.0f:gMotionSpeed==1.0f?2.0f:0.5f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F2,gExperimentKeys[1])) {
                const float v=gFreezing.referenceTimeSeconds;
                gFreezing.referenceTimeSeconds=v<0.1f?0.25f:v<0.5f?1.0f:0.05f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F3,gExperimentKeys[2])) {
                const float v=gFreezing.temperatureC;
                gFreezing.temperatureC=v==-10.0f?-20.0f:v==-20.0f?-5.0f:-10.0f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F4,gExperimentKeys[3])) {
                const float v=gLiquidParameters.flowMotionGain;
                gLiquidParameters.flowMotionGain=v==0?1.0f:v==1?2.0f:v==2?4.0f:0.0f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F5,gExperimentKeys[4])) {
                gPhysicsStep=gPhysicsStep>0.01f?1.0f/120.0f:
                    gPhysicsStep>0.006f?1.0f/240.0f:1.0f/60.0f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F6,gExperimentKeys[5])) {
                const double v=gGrowthResolution.linearMassDensity;
                gGrowthResolution.linearMassDensity=v<0.2?0.3:v<0.6?1.0:0.1;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F7,gCarrierToggleWasDown)) {
                gCarrierFilterEnabled=!gCarrierFilterEnabled;
                experimentChanged=true;
                cout << "Carrier velocity filter " << (gCarrierFilterEnabled?"ON":"OFF")
                     << "; restarting comparison\n";
            }
            if (Rising(window,GLFW_KEY_F9,gF9OneShotWasDown)) {
                gPlayKickOnce=!gPlayKickOnce;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F10,gF10MotionModelWasDown)) {
                gUseAccumulatedSteeringDirection=!gUseAccumulatedSteeringDirection;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_F11,gF11MemoryWasDown)) {
                const float v=gMotionGrowthParameters.relaxationSeconds;
                gMotionGrowthParameters.relaxationSeconds=
                    v<0.2f ? 0.35f : v<0.5f ? 0.70f : 0.15f;
                experimentChanged=true;
            }
            if (Rising(window,GLFW_KEY_K,gClearanceToggleWasDown)) {
                gSurfaceClearanceEnabled=!gSurfaceClearanceEnabled;
                experimentChanged=true;
                cout << "Surface clearance " << (gSurfaceClearanceEnabled?"ON":"OFF") << " (reset)\n";
            }
            if (Rising(window,GLFW_KEY_F12,gF12PreWetWasDown)) {
                gPreWetBeforeKick=!gPreWetBeforeKick;
                experimentChanged=true;
            }
            if (experimentChanged) {
                if (std::abs(realDt-gPhysicsStep)>1e-7f) {
                    // F5 changed dt: drop the old tick accumulator and let
                    // the newly reset experiment start on the new time grid.
                    realDt=gPhysicsStep;
                    tickAccumulator=0.0;
                    tickClock=glfwGetTime();
                }
                water.ResetWater(); icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0; autoIcicleGeneration=true;
                geometryResetRequested=true; selected.hasPrev=false;
                hasPreviousAnimationTime=false; gWaterPaused=false;
                if (animator) { animator->Reset(); animator->SetPlaying(!gPreWetBeforeKick); }
                waitingForPreWet=gPreWetBeforeKick;
                cout << "Motion=" << gMotionSpeed << " tauRef=" << gFreezing.referenceTimeSeconds
                     << " T=" << gFreezing.temperatureC << " gain=" << gLiquidParameters.flowMotionGain
                     << " mu=" << gGrowthResolution.linearMassDensity << " dt=" << gPhysicsStep
                     << " | kickOnce=" << gPlayKickOnce
                     << " | preWet=" << gPreWetBeforeKick
                     << " | memory=" << gMotionGrowthParameters.relaxationSeconds
                     << " | integratedMotion=" << gUseAccumulatedSteeringDirection
                     << " (reset)\n";
            }
            // Trajectory diagnostic: do not consume the inlet as surface ice.
            // Surface and tip freezing are DIFFERENT processes. Restore/calibrate
            // surface freezing separately after checking that roots receive water.
            water.Params().freezeRate=0.0f;

            if (animator)
            {
                // Space can manually override the pre-wet wait.
                if (waitingForPreWet && animator->IsPlaying()) waitingForPreWet=false;
                if (command.step && !animator->IsPlaying())
                {
                    animator->SetPlaying(true);
                    animator->UpdateAnimation(realDt*gMotionSpeed);
                    animator->SetPlaying(false);
                    simulationDt = realDt;
                }
                else
                {
                    // Normal mode loops the kick continuously while ice grows.
                    // Optional F9 diagnostic: stop at the final pose after
                    // one performance; do not stop or reset in loop mode.
                    if (gPlayKickOnce && animator->IsPlaying() &&
                        animator->GetDurationSeconds()>0 &&
                        animator->GetCurrentTimeSeconds()+realDt*gMotionSpeed >=
                            animator->GetDurationSeconds()-1e-5f)
                    {
                        animator->SetPlaying(false);
                        cout << "Kick finished: holding final pose; ice still grows.\n";
                    }
                    if (animator->IsPlaying())
                    {
                        animator->UpdateAnimation(realDt*gMotionSpeed);
                        simulationDt=realDt;
                    }
                }
            }

            if (command.reset)
            {
                selected.hasPrev = false;

                icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0;
                waitingForPreWet=false;
                autoIcicleGeneration = false;

                hasPreviousAnimationTime = false;
            }

            if (command.clear)
            {
                selected = SelectedSurface{};
                icicles.clear(); groundLockLedger=GroundLockLedger{};
                experimentTime=0; animationCycle=0; 
                autoIcicleGeneration = false;  
            }

            int framebufferWidth = 1;
            int framebufferHeight = 1;
            glfwGetFramebufferSize(
                window,
                &framebufferWidth,
                &framebufferHeight);

            framebufferWidth = MaxValue(framebufferWidth, 1);
            framebufferHeight = MaxValue(framebufferHeight, 1);

            const vec3 cameraPosition = GetCameraPosition();

            const mat4 view = lookAt(
                cameraPosition,
                vec3(0.0f),
                vec3(0.0f, 1.0f, 0.0f));

            const mat4 projection = perspective(
                radians(45.0f),
                static_cast<float>(framebufferWidth) /
                static_cast<float>(framebufferHeight),
                0.01f,
                100.0f);

            const float animationTime=animator?animator->GetCurrentTimeSeconds():0.0f;
            vector<mat4> boneMatrices;
            if (animator) boneMatrices=animator->GetFinalBoneMatrices();
            const bool animationWrapped=hasPreviousAnimationTime &&
                animationTime+1e-4f<previousAnimationTime;
            const bool discontinuity=geometryResetRequested || command.reset ||
                groundLayoutChanged;
            const bool loopBoundary=animationWrapped && !discontinuity;
            // Geometry consumers still need to know the pose jumped, independently
            // of whether persistent icicle liquid/filter state must reset.
            const bool poseJump=discontinuity || loopBoundary;
            if (animationWrapped && !geometryResetRequested && !command.reset) ++animationCycle;
            const bool deformationChanged=!hasCachedDeformation || simulationDt>0.0f ||
                geometryResetRequested || command.reset;
            if (deformationChanged)
            {
                cachedLocalPositions=model.ComputeDeformedPositions(boneMatrices);
                hasCachedDeformation=true;
            }
            const auto& localPositions=cachedLocalPositions;
            vector<vec3> worldPositions(localPositions.size());
            for (size_t i=0;i<localPositions.size();++i)
                worldPositions[i]=vec3(modelMatrix*vec4(localPositions[i],1.0f));
            if (!gGround.initialized && !worldPositions.empty())
            {
                float minimumY=worldPositions.front().y;
                for (const vec3& p:worldPositions) minimumY=std::min(minimumY,p.y);
                gGround.height=minimumY-gGround.bodyClearance;
                gGround.initialized=true;
                cout << "Ground initial height " << gGround.height << "\n";
            }
            // The body alone sets carrier height. Ice contact cannot reject an
            // animation pose or lift the whole character to protect an icicle.
            gGround.characterLift=ApplyBodyGroundConstraint(worldPositions);
            const mat4 renderModelMatrix=translate(mat4(1.0f),
                vec3(0,gGround.characterLift,0))*modelMatrix;

            array<mat4,kMaxShaderBones> palette;
            palette.fill(mat4(1.0f));
            copy_n(boneMatrices.begin(),MinValue(boneMatrices.size(),palette.size()),palette.begin());
            glBindBuffer(GL_UNIFORM_BUFFER,boneUBO);
            glBufferSubData(GL_UNIFORM_BUFFER,0,sizeof(palette),palette.data());
            glBindBuffer(GL_UNIFORM_BUFFER,0);

            if (gPickRequested)
            {
                gPickRequested = false;

                const Ray ray = MouseRay(
                    gPickX,
                    gPickY,
                    framebufferWidth,
                    framebufferHeight,
                    view,
                    projection,
                    cameraPosition);

                if (Pick(ray, worldPositions, triangleIndices, selected))
                {
                    cout << "Debug triangle selected.\n";
                }
                else
                {
                    cout << "No triangle hit\n";
                }
            }

            previousAnimationTime=animationTime;
            hasPreviousAnimationTime=(animator!=nullptr);

            UpdateSelected(
                selected,
                worldPositions,
                simulationDt,
                discontinuity,loopBoundary);

            UpdateFlow(
                flow,
                selected);

            water.Params().wind =
                flow.wind;

            water.Params().gravity =
                flow.gravity;

            water.Params().rainDirection =
                flow.rain;


            const float physicsDt = simulationDt>0.0f ? simulationDt : realDt;
            if (!gWaterPaused) experimentTime+=physicsDt;


            if (!gWaterPaused)
            {
                water.Update(
                    worldPositions,
                    physicsDt,
                    poseJump || command.reset);
            }

            if (gSurfaceClearanceEnabled)
                BuildSurfaceGrid(worldPositions,water.Vertices(),triangleIndices);
            else gSurfaceGrid=icicle_growth::SurfaceGrid{};

            if (command.freeze)
            {
                autoIcicleGeneration =
                    !autoIcicleGeneration;

                cout
                    << "Auto Icicle Generation "
                    << (autoIcicleGeneration
                        ? "ON"
                        : "OFF")
                    << "\n";
            }

            for (auto& icicle:icicles)
                if (!icicle.freeze.groundLocked)
                    UpdateSelected(icicle.root,worldPositions,simulationDt,discontinuity,loopBoundary);
            const int lockedByMotion=HandleIcicleGroundContacts(
                icicles,groundLockLedger,physicsDt,poseJump);
            int lockedByGrowth=0;
            if (autoIcicleGeneration && !gWaterPaused && !poseJump &&
                (!gSingleRootExperiment || icicles.empty()))
            {
                if (gSingleRootExperiment && selected.valid)
                    SpawnPinnedSingleRoot(icicles,selected,water,worldPositions,
                        flow,animationTime);
                else
                    SpawnEligibleIcicles(icicles,water,worldPositions,triangleIndices,
                        vertexTriangle,vertexCorner,flow,animationTime,
                        gSingleRootExperiment?1:kMaxIcicleRoots);
            }

            vector<icicle_growth::RootBinding> bindings(icicles.size());
            for (size_t i=0;i<icicles.size();++i)
            {
                const auto& root=icicles[i].root;
                bindings[i].valid=root.valid && icicles[i].freeze.active &&
                    !icicles[i].freeze.groundLocked;
                bindings[i].indices=root.idx;
                bindings[i].weights={{root.bary.x,root.bary.y,root.bary.z}};
            }
            vector<icicle_growth::Supply> supplies(icicles.size());
            if (!icicles.empty() && !gWaterPaused && !poseJump)
            {
                const auto samples=CaptureGrowthSamples(water,worldPositions);
                const auto relativeVelocities=icicle_growth::EstimateDripVelocities(
                    samples,growthAdjacency,water.Params().kFlow,
                    water.Params().transitionGamma);
                supplies=icicle_growth::AllocateNearbyDripSupplies(
                    samples,relativeVelocities,bindings,kMinRootSpacing);
            }
            for (size_t i=0;i<icicles.size();++i)
            {
                auto& icicle=icicles[i];
                icicle.flow.wind=flow.wind;
                icicle.flow.gravity=flow.gravity;
                icicle.flow.rain=flow.rain;
                icicle.flow.wm=water.IsWindInfluenceEnabled()?flow.wm:0.0f;
                // Only the baseline uses this direction for new geometry.
                UpdateFlow(icicle.flow,icicle.root);
                UpdateFreeze(icicle.freeze,icicle.root,icicle.flow,supplies[i],
                    icicle_growth::EffectiveFreezeRate(gFreezing)*gFreezeRateScale,physicsDt,!gWaterPaused,discontinuity,
                    water.IsWindInfluenceEnabled(),loopBoundary);
                if (!icicle.freeze.groundLocked && icicle.freeze.groundContact &&
                    LockIcicleAtPose(icicle,AttachmentPose(icicle.root),groundLockLedger,true))
                    ++lockedByGrowth;
                icicle.freeze.previousAttachmentPose=AttachmentPose(icicle.root);
                icicle.freeze.hasAttachmentPose=icicle.root.valid;
            }
            // Optional F12 diagnostic ONLY: grow a small piece of ice before
            // starting motion. Normal looping mode never waits for pre-growth.
            if (waitingForPreWet && animator && !gWaterPaused)
            {
                const float threshold=std::max(1e-5f,gGrowthResolution.maxSegmentLength*0.2f);
                for (const auto& icicle:icicles)
                    if (icicle.freeze.active && !icicle.freeze.groundLocked &&
                        icicle.freeze.pathLength>=threshold)
                    {
                        waitingForPreWet=false;
                        animator->SetPlaying(true);
                        cout << "Initial ice grown (length=" << icicle.freeze.pathLength
                             << "): kick animation starts at sim=" << experimentTime << "s.\n";
                        break;
                    }
            }
            // Clear after contact handling so this frame's newly locked ice is
            // included. Growing attachments, animation and surface water stay live.
            const bool clearGrounded=NumberRising(window,9,g9WasDown);
            const size_t countBeforeCleanup=icicles.size();
            ArchiveGroundedIcicles(icicles,groundLockLedger,
                clearGrounded?0:kMaxGroundedIcicles);
            if (clearGrounded)
                cout << "Cleared " << countBeforeCleanup-icicles.size()
                     << " grounded icicles; attached ice continues.\n";
            if (traceLog.is_open())
                for (const auto& icicle:icicles)
                    WriteTraceRoot(traceLog,experimentTime,animationTime,animationCycle,discontinuity,
                        icicle,animationWrapped,water.IsWindInfluenceEnabled(),
                        animator && animator->IsPlaying(),waitingForPreWet);
            if (lockedByMotion+lockedByGrowth>0)
                cout << "Ground locked +" << lockedByMotion+lockedByGrowth
                     << " | animation continues\n";
            if (command.print)
            {
                cout
                    << "\n=== Multiple Icicle State ===\n"
                    << "Icicle Count: "
                    << icicles.size()
                    << " | GroundLocks " << groundLockLedger.growthContacts+groundLockLedger.motionContacts
                    << " | ArchivedIce " << groundLockLedger.archivedIce
                    << " | ArchivedWater " << groundLockLedger.archivedWater
                    << " | MaxArchiveError " << groundLockLedger.maxArchivedMassError
                    << "\n";

                for (size_t i = 0; i < icicles.size(); ++i)
                {
                    const IcicleState& icicle = icicles[i];

                    cout
                        << "\nIcicle "
                        << i + 1
                        << "\n";

                    PrintState(
                        model,
                        animation.get(),
                        animator.get(),
                        icicle.root,
                        icicle.flow,
                        icicle.freeze);
                }
            }

            glClearColor(0.055f, 0.07f, 0.09f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            shader.Use();
            shader.SetMat4("uModel", renderModelMatrix);
            shader.SetMat4("uView", view);
            shader.SetMat4("uProjection", projection);
            shader.SetVec3("uCameraPosition", cameraPosition);
            shader.SetVec3("uBaseColor", vec3(0.62f, 0.74f, 0.88f));

            model.Draw(shader);
            DrawGrowthGround(debugRenderer,projection*view);

            size_t debugRootIndex=0;
            for (const IcicleState& icicle : icicles)
            {
                DrawDebug(
                    debugRenderer,
                    projection * view,
                    cameraPosition,
                    icicle.root,
                    icicle.flow,
                    icicle.freeze,debugRootIndex++<kMaxDebugRootVectors);
            }

            DrawWaterOverlay(
                debugRenderer,
                projection * view,
                cameraPosition,
                worldPositions,
                water);
            debugRenderer.Flush(projection*view);

            if (titleAccumulator >= 0.2f)
            {
                ostringstream title;
                title << "Research Viewer | FPS " << fixed << setprecision(1)
                      << renderedFrames/std::max(titleAccumulator,1e-5f);

                if (animator)
                {
                    const float currentAnimationTime=animationTime;

                    title << " | Frame "
                        << static_cast<int>(
                            round(currentAnimationTime * kDisplayFps))
                        << " | "
                        << fixed << setprecision(2)
                        << currentAnimationTime
                        << "/"
                        << animator->GetDurationSeconds()
                        << "s"
                        << (animator->IsPlaying()
                            ? " | Playing"
                            : " | Paused");
                }

                if (selected.valid)
                {
                    title << " | Tri " << selected.tri
                        << " | Speed "
                        << fixed << setprecision(2)
                        << selected.speed;
                }

                //50주차 수정
                title << " | Growth " << (gGrowthModel==GrowthModel::LiquidFlow?"LiquidFlow":"Legacy")
                    << (gGrowthModel==GrowthModel::LiquidFlow?
                        (gDirectionResponse.enabled?" | Response ON":" | Response OFF"):"")
                    << " | Tip freeze x" << gFreezeRateScale
                    << " | Ground " << (gGround.enabled?"ON":"OFF")
                    << " | BodyLift " << fixed << setprecision(3) << gGround.characterLift
                    << " | Sim " << setprecision(2) << experimentTime << "s"
                    << " | Cycle " << animationCycle
                    << (gSingleRootExperiment?" | SingleRoot":"")
                    << (traceLog.is_open()?" | CSV REC":"")
                    << " | GroundLocked " << groundLockLedger.growthContacts+groundLockLedger.motionContacts
                    << (gWaterPaused?" | Growth PAUSED":"");
                size_t totalSegments = 0;

                for (const IcicleState& icicle : icicles)
                {
                    if (!icicle.freeze.localPath.empty())
                    {
                        totalSegments +=
                            icicle.freeze.localPath.size() - 1;
                    }
                }

                title
                    << " | Icicles "
                    << icicles.size()
                    << " | TotalSeg "
                    << totalSegments;

                if (autoIcicleGeneration)
                {
                    title << " | AutoIcicle ON";
                }

                title << " | Water "
                    << setprecision(4)
                    << water.TotalWater()
                    << " | Ice "
                    << TotalIceForDisplay(water)
                    << " | MaxWater "
                    << water.MaxWater()
                    << " | Drip "
                    << water.DripCandidateCount();

                if (gWaterPaused)
                    title << " | SimPaused";

                if (gWireframe)
                    title << " | Wireframe";

                title << " | speed " << gMotionSpeed << " tau " << gFreezing.referenceTimeSeconds
                      << " T " << gFreezing.temperatureC << " gain " << gLiquidParameters.flowMotionGain
                      << " mu " << gGrowthResolution.linearMassDensity << " dt " << gPhysicsStep;
                title << " | CarrierFilter " << (gCarrierFilterEnabled?"ON":"OFF")
                      << " | " << (gUseAccumulatedSteeringDirection?"LiquidParcels":"LegacyDirection")
                      << " | " << (gPlayKickOnce?"OneShotDiagnostic":"LoopGrowth")
                      << " | mem " << gMotionGrowthParameters.relaxationSeconds
                      << (waitingForPreWet?" | WaitingForIce":"");
                glfwSetWindowTitle(window, title.str().c_str());
                titleAccumulator = 0.0f;
                renderedFrames=0;
            }

            glfwSwapBuffers(window);
            glfwPollEvents();
        }

        glDeleteBuffers(1, &boneUBO);
        glfwDestroyWindow(window);
        glfwTerminate();

        return 0;
    }
    catch (const exception& e)
    {
        cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }
}
