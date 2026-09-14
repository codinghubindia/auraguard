# NAVIGUARD Phase 10 — RELLIS-Derived 3D World Representation & Architecture

**Author**: NAVIGUARD Simulation & Realistic Environment Engineer  
**Target Simulator**: Gazebo Harmonic (Gz Sim 8.x) / ROS 2 Jazzy  
**Dataset Reference**: Texas A&M RELLIS Campus Off-Road Track (Sequence 00000)  

---

## 1. Technical Evaluation of Terrain Representations

Gazebo Harmonic uses the DART physics engine by default. Off-road unstructured environments present unique challenges for real-time physics and optical perception:

| Representation Method | Real-time Physics Viability | Visual Realism for Monocular Camera | Gazebo Harmonic Native Support | Engineering Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Raw 3D Point Cloud Mesh** | Extremely Poor (millions of facets cause physics solver slowdown or crash) | Poor (untextured vertex cloud) | Incompatible | **REJECTED** |
| **Uniform Flat Plane** | Excellent ($O(1)$ collision) | Unrealistic (does not stress off-road suspension or visual pitch/roll) | Full | **REJECTED** (Unsuitable for off-road evaluation) |
| **Pure Heightmap** | Good | Moderate (textures stretched on steep grades) | Full (via SDF `<heightmap>`) | Partial (useful for ground, but fails for discrete 3D obstacles) |
| **Hybrid Terrain + Discrete Obstacle Geometry** | **Optimal** (simplified collision geometries with high-fidelity visual textures) | **Superior** (real vegetation, mud patches, tree trunks, off-road track) | **Native & Robust** | **SELECTED** |

---

## 2. Hybrid World Architecture

The RELLIS-derived simulation world (`rellis_outdoor_world.sdf`) employs a **strict decoupling between the Visual Representation and the Collision Representation**:

```
                         RELLIS SEQUENCE 00000
                                   │
                ┌──────────────────┴──────────────────┐
                ▼                                     ▼
        VISUAL PIPELINE                       COLLISION PIPELINE
   (Camera Perception & RViz)              (DART Physics Engine)
                │                                     │
   • High-resolution dirt track          • Low-polygon collision bounds
   • Photorealistic grass & soil         • Cylinder collision proxies for trees
   • Detailed tree trunks & foliage      • Box proxies for fallen logs & barriers
   • Mud depressions & water puddles     • Bounded traversability friction zones
                │                                     │
                └──────────────────┬──────────────────┘
                                   ▼
                      GAZEBO HARMONIC SIMULATION
```

### 2.1 Terrain Layout & Features
The world recreates an authentic $40\text{ m} \times 40\text{ m}$ section of the RELLIS off-road track:
* **Main Track**: $3.5\text{ m}$-wide compacted gravel and dirt trail curving gently from the spawn point $(0, 0)$ toward destination coordinates $(6, 4)$ and $(12, 0)$.
* **Shoulder & Off-Trail**: Rough grass terrain with friction coefficients reflecting unpaved ground ($\mu_1 = 0.8, \mu_2 = 0.7$).
* **Obstacle Obstructions**:
  * Off-road trees (cylindrical trunks with organic visual crowns).
  * Fallen logs and barrier posts blocking direct straight-line paths, forcing the Phase 9 A* planner to navigate winding natural corridors.
  * Low-texture mud hazards and dark puddle patches that challenge visual odometry feature tracking, exciting the Phase 7 Confidence Engine and Phase 8 Autonomous Recovery.

### 2.2 Collision vs. Visual Separation Rules
1. **Tree Trunks**:
   * Visual: Detailed trunk mesh with bark texture and canopy.
   * Collision: Lightweight vertical `<cylinder>` with radius $0.30\text{ m}$ matching the trunk base.
2. **Logs & Barriers**:
   * Visual: Natural wooden log texture.
   * Collision: Oriented `<box>` matching the physical bounding box.
3. **Terrain Surface**:
   * Multi-segmented terrain plane with elevation variations and defined track borders.
