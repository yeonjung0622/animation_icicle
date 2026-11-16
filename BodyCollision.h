#pragma once
#include "IcicleGrowth.h"
#include <limits>
#include <numeric>
#include <map>
#include <tuple>

namespace icicle_growth {
// Two-sided triangle/capsule queries against the CURRENT skinned mesh.
// Topology is built once and bounding boxes are refitted at every pose.
// Inside tests require a closed mesh. Open clothes still get two-sided surface
// collision; they cannot define a reliable solid interior.
struct BodyContact {
    bool hit=false;
    float distance=std::numeric_limits<float>::max();
    Vec3 point{},normal{0,0,1};
};
class BodyCollisionMesh {
    struct Box {
        Vec3 lo{std::numeric_limits<float>::max()},hi{-std::numeric_limits<float>::max()};
        void Add(Vec3 p) {
            lo={std::min(lo.x,p.x),std::min(lo.y,p.y),std::min(lo.z,p.z)};
            hi={std::max(hi.x,p.x),std::max(hi.y,p.y),std::max(hi.z,p.z)};
        }
        void Add(const Box& b) { Add(b.lo); Add(b.hi); }
        bool Overlaps(const Box& b) const {
            return lo.x<=b.hi.x && hi.x>=b.lo.x && lo.y<=b.hi.y &&
                hi.y>=b.lo.y && lo.z<=b.hi.z && hi.z>=b.lo.z;
        }
    };
    struct Triangle { unsigned int a,b,c; };
    struct Node { Box box; int left=-1,right=-1; size_t begin=0,end=0; };
    std::vector<Vec3> vertices;
    std::vector<Triangle> triangles;
    std::vector<size_t> order;
    std::vector<Node> nodes;
    std::vector<int> closedGroup;
    void ClassifyClosedComponents() {
        // FBX exports often split coincident vertices at UV/normal seams.
        // Weld for topology classification only; collision positions stay intact.
        Box bounds;for(auto p:vertices) bounds.Add(p);
        const float tolerance=std::max(1e-8f,Length(bounds.hi-bounds.lo)*1e-6f);
        std::map<std::tuple<long long,long long,long long>,size_t> welded;
        std::vector<size_t> vertexId(vertices.size());
        for(size_t i=0;i<vertices.size();++i) {
            const auto p=vertices[i];const auto key=std::make_tuple(std::llround(p.x/tolerance),
                std::llround(p.y/tolerance),std::llround(p.z/tolerance));
            auto it=welded.emplace(key,welded.size()).first;vertexId[i]=it->second;
        }
        std::vector<size_t> parent(triangles.size());std::iota(parent.begin(),parent.end(),size_t{0});
        const auto find=[&](size_t a) {while(parent[a]!=a) {parent[a]=parent[parent[a]];a=parent[a];}return a;};
        std::map<std::pair<size_t,size_t>,std::vector<size_t>> edges;
        for(size_t i=0;i<triangles.size();++i) {
            const auto& t=triangles[i];const size_t v[3]={vertexId[t.a],vertexId[t.b],vertexId[t.c]};
            for(int e=0;e<3;++e) {
                const size_t a=std::min(v[e],v[(e+1)%3]),b=std::max(v[e],v[(e+1)%3]);
                edges[{a,b}].push_back(i);
            }
        }
        for(const auto& edge:edges) {
            const auto& ts=edge.second;
            for(size_t i=1;i<ts.size();++i) parent[find(ts[i])]=find(ts[0]);
        }
        std::vector<bool> closed(triangles.size(),true);
        for(const auto& edge:edges) if(edge.second.size()!=2 || edge.first.first==edge.first.second)
            for(size_t t:edge.second) closed[find(t)]=false;
        closedGroup.resize(triangles.size());
        for(size_t i=0;i<triangles.size();++i) {
            const size_t root=find(i);closedGroup[i]=closed[root]?static_cast<int>(root):-1;
        }
    }
    static float Component(Vec3 p,int a) { return a==0?p.x:a==1?p.y:p.z; }
    Box TriangleBox(size_t i) const {
        Box b; const auto& t=triangles[i];
        b.Add(vertices[t.a]); b.Add(vertices[t.b]); b.Add(vertices[t.c]); return b;
    }
    int Build(size_t begin,size_t end) {
        const int n=static_cast<int>(nodes.size()); nodes.push_back({});
        nodes[n].begin=begin; nodes[n].end=end;
        Box b; for (size_t i=begin;i<end;++i) b.Add(TriangleBox(order[i]));
        nodes[n].box=b;
        if (end-begin>8) {
            const Vec3 extent=b.hi-b.lo;
            const int axis=extent.x>extent.y?(extent.x>extent.z?0:2):(extent.y>extent.z?1:2);
            const size_t mid=begin+(end-begin)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,
                [&](size_t a,size_t c) {
                    const auto x=TriangleBox(a),y=TriangleBox(c);
                    return Component(x.lo+x.hi,axis)<Component(y.lo+y.hi,axis);
                });
            const int left=Build(begin,mid),right=Build(mid,end);
            nodes[n].left=left; nodes[n].right=right;
        }
        return n;
    }
    void Refit(int n) {
        auto& node=nodes[n]; Box b;
        if (node.left<0) {
            for (size_t i=node.begin;i<node.end;++i) b.Add(TriangleBox(order[i]));
        } else { Refit(node.left); Refit(node.right); b.Add(nodes[node.left].box); b.Add(nodes[node.right].box); }
        node.box=b;
    }
    static Vec3 ClosestEdge(Vec3 p,Vec3 a,Vec3 b) {
        const Vec3 ab=b-a; const float d=Dot(ab,ab);
        return a+ab*(d>1e-18f?std::clamp(Dot(p-a,ab)/d,0.f,1.f):0.f);
    }
    static Vec3 ClosestTriangle(Vec3 p,Vec3 a,Vec3 b,Vec3 c) {
        const Vec3 ab=b-a,ac=c-a,n=Cross(ab,ac);
        const float nn=Dot(n,n);
        if (nn>1e-18f) {
            const Vec3 q=p-n*(Dot(p-a,n)/nn);
            if (Dot(Cross(ab,q-a),n)>=0 && Dot(Cross(c-b,q-b),n)>=0 &&
                Dot(Cross(a-c,q-c),n)>=0) return q;
        }
        const Vec3 x=ClosestEdge(p,a,b),y=ClosestEdge(p,b,c),z=ClosestEdge(p,c,a);
        const float dx=Dot(p-x,p-x),dy=Dot(p-y,p-y),dz=Dot(p-z,p-z);
        return dx<dy?(dx<dz?x:z):(dy<dz?y:z);
    }
    static void ClosestEdges(Vec3 a,Vec3 b,Vec3 c,Vec3 d,Vec3& p,Vec3& q) {
        const Vec3 u=b-a,v=d-c,w=a-c;
        const float aa=Dot(u,u),bb=Dot(u,v),cc=Dot(v,v),dd=Dot(u,w),ee=Dot(v,w);
        float s=0,t=0;
        if (aa<=1e-18f) t=cc>1e-18f?std::clamp(ee/cc,0.f,1.f):0;
        else if (cc<=1e-18f) s=std::clamp(-dd/aa,0.f,1.f);
        else {
            const float denom=aa*cc-bb*bb;
            s=denom>1e-18f?std::clamp((bb*ee-cc*dd)/denom,0.f,1.f):0;
            t=(bb*s+ee)/cc;
            if (t<0) {t=0;s=std::clamp(-dd/aa,0.f,1.f);}
            else if (t>1) {t=1;s=std::clamp((bb-dd)/aa,0.f,1.f);}
        }
        p=a+u*s; q=c+v*t;
    }
    static bool RayTriangle(Vec3 p,Vec3 direction,Vec3 a,Vec3 b,Vec3 c,float& t) {
        const Vec3 edge1=b-a,edge2=c-a,h=Cross(direction,edge2);
        const float det=Dot(edge1,h);
        const float scale=Length(edge1)*Length(edge2)*Length(direction);
        if (std::abs(det)<=1e-8f*std::max(scale,1e-15f)) return false;
        const Vec3 s=p-a; const float u=Dot(s,h)/det;
        if (u<0 || u>1) return false;
        const Vec3 q=Cross(s,edge1); const float v=Dot(direction,q)/det;
        if (v<0 || u+v>1) return false;
        t=Dot(edge2,q)/det; return t>=0;
    }
    BodyContact TriangleDistance(Vec3 from,Vec3 to,size_t i) const {
        const auto& tri=triangles[i];
        const Vec3 a=vertices[tri.a],b=vertices[tri.b],c=vertices[tri.c];
        BodyContact r; Vec3 onSegment=from;
        const auto test=[&](Vec3 p,Vec3 q) {
            const float distance=Length(p-q);
            if (distance<r.distance) {r.distance=distance;onSegment=p;r.point=q;}
        };
        float t=0;
        if (RayTriangle(from,to-from,a,b,c,t) && t<=1.f) {
            r.distance=0; r.point=from+(to-from)*t; onSegment=r.point;
        } else {
            test(from,ClosestTriangle(from,a,b,c)); test(to,ClosestTriangle(to,a,b,c));
            Vec3 p,q;
            ClosestEdges(from,to,a,b,p,q); test(p,q);
            ClosestEdges(from,to,b,c,p,q); test(p,q);
            ClosestEdges(from,to,c,a,p,q); test(p,q);
        }
        Vec3 normal=Unit(Cross(b-a,c-a),{0,0,1});
        if (Dot(normal,from-r.point)<0) normal=normal*-1.f;
        r.normal=Unit(onSegment-r.point,normal);
        return r;
    }
    static bool RayBox(Vec3 p,Vec3 d,const Box& box) {
        float low=0,high=std::numeric_limits<float>::max();
        for (int i=0;i<3;++i) {
            const float x=Component(p,i),v=Component(d,i),a=Component(box.lo,i),b=Component(box.hi,i);
            if (std::abs(v)<1e-12f) {if (x<a || x>b) return false;continue;}
            float t0=(a-x)/v,t1=(b-x)/v; if(t0>t1) std::swap(t0,t1);
            low=std::max(low,t0);high=std::min(high,t1);if(low>high) return false;
        }
        return true;
    }
public:
    void Initialize(const std::vector<Vec3>& positions,const std::vector<unsigned int>& indices) {
        vertices=positions; triangles.clear();nodes.clear();order.clear();
        for(size_t i=0;i+2<indices.size();i+=3)
            if(indices[i]<vertices.size() && indices[i+1]<vertices.size() && indices[i+2]<vertices.size())
                triangles.push_back({indices[i],indices[i+1],indices[i+2]});
        order.resize(triangles.size());std::iota(order.begin(),order.end(),size_t{0});
        if (!order.empty()) {ClassifyClosedComponents();Build(0,order.size());}
    }
    bool Update(const std::vector<Vec3>& positions) {
        if(positions.size()!=vertices.size()) return false;
        vertices=positions; if(!nodes.empty()) Refit(0);return true;
    }
    bool Empty() const {return nodes.empty();}
    BodyContact Contact(Vec3 from,Vec3 to,float radius) const {
        BodyContact result; if(nodes.empty()) return result;
        const Vec3 padding{std::max(0.f,radius)}; Box query;
        query.Add(from-padding);query.Add(from+padding);query.Add(to-padding);query.Add(to+padding);
        std::vector<int> stack{0};
        while(!stack.empty()) {
            const auto& node=nodes[stack.back()];stack.pop_back();
            if(!node.box.Overlaps(query)) continue;
            if(node.left>=0) {stack.push_back(node.left);stack.push_back(node.right);continue;}
            for(size_t i=node.begin;i<node.end;++i) {
                const auto hit=TriangleDistance(from,to,order[i]);
                if(hit.distance<result.distance) result=hit;
            }
        }
        result.hit=result.distance<=std::max(0.f,radius);return result;
    }
    bool Inside(Vec3 p) const {
        if(nodes.empty()) return false;
        const Vec3 direction=Unit(Vec3{1,.37139f,.52911f});
        std::vector<std::pair<int,float>> hits;std::vector<int> stack{0};
        while(!stack.empty()) {
            const auto& node=nodes[stack.back()];stack.pop_back();
            if(!RayBox(p,direction,node.box)) continue;
            if(node.left>=0) {stack.push_back(node.left);stack.push_back(node.right);continue;}
            for(size_t i=node.begin;i<node.end;++i) {
                const size_t index=order[i];if(closedGroup[index]<0) continue;
                const auto& tri=triangles[index];float t;
                if(RayTriangle(p,direction,vertices[tri.a],vertices[tri.b],vertices[tri.c],t) && t>1e-6f)
                    hits.push_back({closedGroup[index],t});
            }
        }
        std::sort(hits.begin(),hits.end());size_t crossings=0;float last=-1;int group=-1;
        for(const auto& hit:hits) {
            if(hit.first!=group) {
                if(crossings%2==1) return true;
                group=hit.first;crossings=0;last=-1;
            }
            const float t=hit.second;
            if(last<0 || t-last>1e-5f*std::max(1.f,t)) {++crossings;last=t;}
        }
        return crossings%2==1;
    }
};

struct SafeTipStep { bool accepted=false;Vec3 endpoint{},direction{}; };
inline SafeTipStep ChooseSafeTipStep(const BodyCollisionMesh& body,Vec3 from,
    Vec3 previous,Vec3 desired,Vec3 rootNormal,float pathLength,float stepLength,
    bool hasPrevious,const TipShapeParameters& p,Vec3 gravityDown={0,-1,0})
{
    SafeTipStep result;result.endpoint=from;
    if(!(stepLength>1e-9f)) return result;
    const float radius=std::max(0.f,p.iceRadius)+std::max(0.f,p.collisionMargin);
    if(body.Inside(from) || body.Contact(from,from,radius).hit) return result;
    desired=Unit(desired,rootNormal);previous=Unit(previous,desired);
    const Vec3 down=Unit(gravityDown);
    // Surface escape is a constraint, not a source of upward water momentum.
    // It must not turn a downward flow into an upward-growing tentacle.
    const float requestedDown=Dot(desired,down);
    // Launch the first few millimetres outward, then release the surface bias.
    const float exit=std::exp(-std::max(0.f,pathLength)/std::max(.001f,p.rootExitLength));
    desired=Unit(desired+rootNormal*(.65f*exit),desired);
    const float lookAhead=std::max(stepLength,p.avoidanceLookAhead);
    const auto future=body.Contact(from,from+desired*lookAhead,radius*1.25f);
    if(future.hit) {
        const Vec3 slide=desired-future.normal*std::min(0.f,Dot(desired,future.normal));
        desired=Unit(slide+future.normal*.55f,rootNormal);
    }
    const float downFloor=std::min(0.f,requestedDown);
    if(Dot(desired,down)<downFloor-1e-5f) return result;
    const float angle=std::min(std::max(0.f,p.maximumCurvature)*stepLength,
        std::clamp(p.maximumCornerDegrees,0.f,90.f)*.01745329252f);
    const Vec3 central=hasPrevious?TurnTowards(previous,desired,angle):desired;
    const auto valid=[&](Vec3 d) {
        // A previously tilted solid can point upward after the body moves.
        // Allow gradual recovery, but no extra upward turn merely to escape.
        const float recoveryFloor=hasPrevious?std::min(downFloor,Dot(previous,down)):downFloor;
        return Dot(d,down)>=recoveryFloor-1e-5f &&
               !body.Contact(from,from+d*stepLength,radius).hit &&
               !body.Inside(from+d*stepLength);
    };
    Vec3 best=central;float bestScore=-std::numeric_limits<float>::max();
    const auto tryDirection=[&](Vec3 d) {
        const float score=Dot(d,desired);
        if(score>bestScore && valid(d)) {best=d;bestScore=score;}
    };
    tryDirection(central);
    if(bestScore>-2) {result={true,from+best*stepLength,best};return result;}
    // Search only INSIDE the allowed curvature cone. If collision and smoothness
    // cannot both be satisfied, withhold elongation instead of making a kink.
    const Vec3 axis=hasPrevious?previous:central;
    const Vec3 seed=std::abs(axis.x)<.8f?Vec3{1,0,0}:Vec3{0,1,0};
    const Vec3 u=Unit(Cross(axis,seed)),v=Cross(axis,u);
    tryDirection(axis);
    for(int ring=1;ring<=2;++ring) for(int j=0;j<24;++j) {
        const float a=(hasPrevious?angle:.5f)*ring/2.f,phi=6.28318530718f*j/24.f;
        tryDirection(Unit(axis*std::cos(a)+(u*std::cos(phi)+v*std::sin(phi))*std::sin(a)));
    }
    if(bestScore>-2) result={true,from+best*stepLength,best};
    return result;
}
} // namespace icicle_growth
