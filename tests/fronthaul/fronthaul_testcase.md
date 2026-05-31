# Chanel Model Test

## 1. Manual Calculation Report

### A2G Probabilistic Path Loss
**Parameters:**
*   Environment: `Urban` ($\alpha_U = 0.3, \beta_U = 500, \gamma_U = 15$)
*   Frequency: $f_c = 2.0 \times 10^9$ Hz (2 GHz)
*   UAV Height: $h_{UAV} = 50$ m, UE Height: $h_{UE} = 1.5$ m
*   Horizontal Distance: $d_{2D} = 100$ m
*   $\eta_{LOS} = 1.0$ dB, $\eta_{NLOS} = 20.0$ dB (from `Urban` 2GHz profile)

#### 1.1 Environmental Parameters $\alpha$ and $\beta$
Using Equation (12) and Table 1 & 2 coefficients (where $X = \alpha_U \beta_U = 150$ and $Y = \gamma_U = 15$):
*   $\alpha \approx 9.6117$
*   $\beta \approx 0.1581$

Note: These values are derived from the formulas used in the code.

#### 1.2 Geometry
*   Height difference: $h = 50 - 1.5 = 48.5$ m
*   3D Distance: $d_{3D} = \sqrt{100^2 + 48.5^2} \approx 111.14$ m
*   Elevation Angle ($\theta$): $\arctan(\frac{48.5}{100}) \cdot \frac{180}{\pi} \approx 25.87^\circ$

#### 1.3 Probabilities and Path Loss
*   **$P_{LOS}$ (Eq 1):**
    $$P_{LOS} = \frac{1}{1 + 9.61 \exp(-0.158(25.87 - 9.61))} \approx 0.5735$$
*   **FSPL (Eq 2):**
    $$FSPL = 20 \log_{10}\left(\frac{4 \pi \cdot 2 \cdot 10^9 \cdot 111.14}{2.998 \cdot 10^8}\right) \approx 79.39 \text{ dB}$$
*   **Expected Path Loss (Eq 3):**
    $$L = FSPL + P_{LOS} \cdot \eta_{LOS} + (1 - P_{LOS}) \cdot \eta_{NLOS}$$
    $$L = 79.39 + (0.5735 \cdot 1) + (0.4265 \cdot 20) \approx 79.39 + 0.5735 + 8.53 \approx 88.49 \text{ dB}$$

---

### Scenario B: G2G Urban Macro (UMa) Path Loss
**Parameters:**
*   Frequency: $f_c = 2$ GHz
*   BS Height: $h_{BS} = 25$ m, UE Height: $h_{UE} = 1.5$ m
*   Horizontal Distance: $d_{2D} = 500$ m
*   Condition: Line-of-Sight (LOS)

#### 2.1 Breakpoint Distance
TR 38.901 defines $d_{BP} = \frac{4 h'_{BS} h'_{UE} f_c}{c}$.
Since $h_{UE} < 13$m, the effective environment height $h_E = 1.0$m.
*   $h'_{BS} = 25 - 1 = 24$ m
*   $h'_{UE} = 1.5 - 1 = 0.5$ m
*   $d_{BP} = \frac{4 \cdot 24 \cdot 0.5 \cdot 2 \cdot 10^9}{299792458} \approx 320.22$ m

#### 2.2 Path Loss (LOS)
Since $d_{2D} > d_{BP}$, we use $PL_2$:
*   $d_{3D} = \sqrt{500^2 + (25-1.5)^2} \approx 500.55$ m
*   $PL_{LOS} = 28.0 + 40 \log_{10}(d_{3D}) + 20 \log_{10}(f_{GHz}) - 9 \log_{10}(d_{BP}^2 + (h_{BS} - h_{UE})^2)$
*   $PL_{LOS} = 28.0 + 40 \log_{10}(500.55) + 20 \log_{10}(2) - 9 \log_{10}(320.22^2 + 23.5^2)$
*   $PL_{LOS} = 28.0 + 107.97 + 6.02 - 45.10 \approx 96.89 \text{ dB}$
This report provides the mathematical validation and the corresponding `pytest` implementation for the `FHLayer` module.

---

# FH Layer Test

**Setup:** 2 UEs, 2 Sources (Source 0: UAV, Source 1: BS).
*   **$P_T$:** UAV = 23 dBm, BS = 46 dBm.
*   **SNR Threshold:** 10.0 dB.

| Link | $P_T$ [dBm] | $L$ [dB] | $P_{R}$ [dBm] | $P_{R}$ [mW] | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **UE0 to UAV** | 23 | **70** | **-47** | $2.0 \cdot 10^{-5}$ | **Assoc: YES** (Strongest) |
| **UE0 to BS** | 46 | **120** | -74 | $4.0 \cdot 10^{-8}$ | Interference only |
| **UE1 to UAV** | 23 | **110** | -87 | $2.0 \cdot 10^{-9}$ | Interference only |
| **UE1 to BS** | 46 | **70** | **-24** | $4.0 \cdot 10^{-3}$ | **Assoc: YES** (Strongest) |