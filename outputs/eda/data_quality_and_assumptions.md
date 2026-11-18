# Data-Quality Issues, Assumptions, and Pipeline Decisions

1. **Global Multi-City Coordinate Distribution:**
   - Trips span longitudes `[-100.42°, 137.24°]` and latitudes `[-34.64°, 61.70°]` across 65+ regional clusters.
   - *Decision:* Never use raw `(lon, lat)` coordinates as ML features or a fixed regional UTM zone. Every trip is projected independently into local metric meters `(x, y)` centered on its median coordinate using WGS84 radii of curvature.

2. **Two-Way Directed Road Segments:**
   - ~97.7% of road segments in urban extracts have an opposite-direction twin `(v, u)` sharing the exact same physical geometry as `(u, v)`.
   - *Decision:* Distance features alone cannot distinguish opposite lanes; windowed GPS movement heading (`cos_heading_diff`, `cos_heading_weighted`) and graph transition continuity are included.

3. **Long Highway Arcs (up to 9.3 km) vs. Midpoint Indexing:**
   - While median arc length is ~25.5 m, highway arcs reach `9,326.9 m` (`train`) and `7,007.2 m` (`student_test`).
   - *Decision:* Arcs are densified every `25.0 m` prior to `cKDTree` spatial indexing (`TOP_K=10`, `radius=150 m`), achieving a `0.3810%` Candidate Miss Rate on `student_test`.

4. **Irregular Timestamp Gaps & Multipath Speed Spikes:**
   - Although ~96.5% (`train`) and ~98.6% (`student_test`) of steps are 1 s apart, 40 `train` trips and 2 `student_test` trips (`00000065`: 210 s, `00000076`: 32 s) contain gaps >10 s (up to `153,064 s` in `train`), and raw GPS jitter creates implied speeds up to `180.14 m/s`.
   - *Decision:* Speeds are smoothed and clipped to `[0, 60] m/s`, gaps use `log1p(dt)`, and HMM transitions bridge multi-hop road paths via bounded Dijkstra search.

5. **Reference Route Limitations & Disconnected Graph Islands:**
   - Road networks contain multiple weak components per trip (map-border clipping), and `.route` itself has 13 disjoint transitions in `train` and 1 in `student_test` (`00000076`). Furthermore, outlier training trips `00000054` and `00000033` have median GPS-to-route distances >120 m.
   - *Decision:* Offline `.track`→`.route` alignment uses order-preserving monotonic dynamic programming, and the Viterbi decoder includes a finite `teleport_penalty` fallback so disconnected components never cause decoding failure.
