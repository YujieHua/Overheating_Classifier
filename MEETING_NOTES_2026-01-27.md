# Meeting Notes: SRG Meeting - January 27, 2026

## Attendees
- **Ali** (Industry partner - SmartFusion)
- **Prof. Ehsan Toyserkani** (Supervisor)
- **Yuji Hua** (You - Overheating Predictor)
- Nadia (Plackett-Burman DOE - separate project)
- Farima (OT + UNet patching - separate project)

---

## Overheating Predictor Discussion Summary

### Ali's Proposed Approach: Energy-Based (Simplified)

Ali proposed a **simpler alternative** to temperature simulation:

**Core Idea:** Track **cumulative Joules per layer** instead of actual temperature.

> "Let's not make it complicated and let's not figure out the temperature - let's just simply say cumulation of joules cause that would be a good indication"

#### Key Points from Ali:

1. **Input is simple:** Joule per mm³ from laser parameters
   - Known accurately from process settings
   - Example material: Inconel 718 (latent heat of fusion, thermal conductivity matter)

2. **Energy consumption has two paths:**
   - **Latent heat of fusion** - consumed to melt material
   - **Heat dissipation** - rate per unit area (function of thermal conductivity, radiation)

3. **Accumulation happens when:**
   - Not enough cooling time given, OR
   - Geometry has a "neck" (bottleneck)

4. **Geometry affects two things:**
   - **Energy INPUT:** Cross-sectional area × energy density = total energy for layer
   - **Energy DISSIPATION:** Previous layer's cross-sectional area determines heat escape rate
   - If top layer > bottom layer → Joules accumulate

5. **Accuracy estimate:** ~60-70% (simple but useful)

---

### Prof. Toyserkani's Key Inputs

1. **Gas flow velocity is CRITICAL:**
   > "One thing is very important... it's a function of gas flow"

   - Need gas velocity on top of part to calculate convection coefficient
   - Ali confirmed: velocity data is available
   - Gas temperature: ambient (~25°C)

2. **Simplified scope is feasible:**
   > "If you're not really looking for the entire geometry this is quite straightforward"

   - Look at each slice individually
   - Identify average temperature per slice
   - Consider dwell time between layers
   - Calculate energy remaining / temperature drop

3. **Future integration potential:**
   - Can link with residual stress model
   - Can integrate with Optical Tomography (OT) data
   - Would provide more tangible output for users

---

### Yuji's Clarifications & Contributions

1. **Energy → Temperature relationship:**
   - Q: "Energy input proportional to temperature rise?"
   - A (Ali): Yes, BUT previous layer temperature matters
     - Starting from 1000°C vs 25°C requires different energy to melt
     - Remaining energy contributes to temperature rise

2. **Geometry Multiplier proposal (your approach):**
   - You showed the 3D Gaussian convolutional filter concept
   - Classifies each voxel based on support from below
   - Overhangs get higher scores
   - Ali's response: "Very similar but might be more advanced version"
   - Ali: "If you think this is going to be a good method and it's easier since you have it already... we can give it a try"

3. **Validation approach agreed:**
   - Use SmartFusion "manifolds geometry" (they have it)
   - Compare with their OT results
   - Look at TRENDS (high/low values by height) not absolute values

---

### Action Items from Meeting

| # | Action | Owner | Notes |
|---|--------|-------|-------|
| 1 | Get gas flow velocity data | Ali | Needed for convection coefficient |
| 2 | Test geometry multiplier on manifolds geometry | Yuji | Ali will provide the geometry file |
| 3 | Compare predictions vs OT trends | Yuji | They have OT data for validation |
| 4 | Consider simplified joules approach | Yuji | Alternative to full temperature model |

---

### Key Quotes

**Ali on simplicity:**
> "Let's not make it complicated... just simply say cumulation of joules cause that would be a good indication"

**Ali on geometry:**
> "Your previous layer cross sectional area would actually play a big role... that's basically the majority of the consumption"

**Prof. Toyserkani on gas flow:**
> "That velocity would be good enough for us"

**Prof. Toyserkani's endorsement:**
> "What you did here is quite useful and then we can link it with some other program"

---

### Timeline Context

- **3-4 week R&D window** - Printing paused at SmartFusion
- This is "golden time" to develop and test the model
- Can use this window to:
  - Implement approach
  - Test on their geometry
  - Compare with OT data

---

## Non-Relevant Meeting Content (For Reference)

### Nadia's Section (Skip)
- Plackett-Burman DOE for process parameters
- Gas flow 330 vs 380 experiments
- HSS hard recoater considerations

### Farima's Section (Skip)
- OT image analysis with UNet
- Patching approach for geometry invariance
- Void detection improvements

---

## Meeting Date Context
- This meeting occurred **6 days after** the Jan 21 meeting where Prof. Toyserkani approved V1.0 concept
- Introduces industry partner (Ali/SmartFusion) perspective
- Provides simpler alternative approach + validation data
