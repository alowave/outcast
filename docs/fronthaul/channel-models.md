# Fronthaul Channel Modeling

This repo uses a **large-scale (mean) propagation model** for UAV ↔ ground links. The goal is to capture
**average path loss and received power** for placement/association/throughput studies, while
**ignoring small-scale fading** (multipath, fast time variations, Doppler spread). Those fast effects are typically
handled at other layers (e.g., link adaptation, scheduling, HARQ), so we model the channel at the “mean link budget”
level.

---

## DBS–GN Path Loss Model

We use the commonly adopted air-to-ground path loss model from Al-Hourani et al. (2014) . It combines:

* Free-space path loss (FSPL)
* A probabilistic **LOS/NLOS** mixture
* Fixed excess losses for LOS and NLOS to capture average shadowing/scattering

### LOS probability

For DBS (j) and ground node (n):

$$
P_{\text{LOS}}(j,n)=\frac{1}{1+\alpha \exp\left(-\beta(\omega-\alpha)\right)}
$$
<p align="right">(1)</p>

where ($\alpha,\beta$) are environment-dependent constants and

$$
\omega=\arctan\left(\frac{h_{j}}{d_{2d}(j,n)}\right)
$$

is the elevation angle, using DBS altitude ($h_j$) and 2D distance ($d_{2d}$).

### Free-space path loss (dB)

$$
\mathrm{FSPL}=20 \log_{10}\left(\frac{4\pi f_{c}d(j,n)}{c}\right)
$$
<p align="right">(2)</p>

where ($f_c$) is carrier frequency, ($c$) is the speed of light, and ($d(j,n)$) is 3D distance.

### Expected path loss (dB)

Using fixed additional losses ($\eta_{\mathrm{LOS}}$) and ($\eta_{\mathrm{NLOS}}$):

$$
L^{[\mathrm{dB}]}(j,n)=\mathrm{FSPL}+P_{\text{LOS}}(j,n)\eta_{\mathrm{LOS}}+
\left(1-P_{\text{LOS}}(j,n)\right)\eta_{\mathrm{NLOS}}
$$
<p align="right">(3)</p>

---

## Received Power, SNR, SINR, and Rate

Let the DBS transmit powers be:

$$
\mathbf{P}=[P_{T}(1),...,P_{T}(M)]
$$
<p align="right">(4)</p>

Assume AWGN with noise power ($\sigma_{0}^{2}$). For simplicity (and to keep results comparable across algorithms),
many experiments use **equal per-UE bandwidth** ($B_n=B_0$). If ($B_D$) is the total DBS bandwidth, the received power
at user ($n$) from DBS ($j$) is:

$$
P_{R}(j,n)=\frac{B_{n}P_{T}(j)10^{\frac{-L^{[\mathrm{dB}]}(j,n)}{10}}}{B_{D}}
$$
<p align="right">(5)</p>

Then:

$$
\mathrm{SNR}(j,n)=\frac{P_{R}(j,n)}{\sigma_{0}^{2}}
$$
<p align="right">(6)</p>

$$
\Gamma(j,n)=\frac{P_{R}(j,n)}{\sigma_{0}^{2}+\sum_{i\ne j}^{M}P_{R}(i,n)}
$$
<p align="right">(7)</p>

Using Shannon capacity as an upper bound:

$$
R(j,n)=B_{n}\log_{2}(1+\Gamma(j,n))
$$
<p align="right">(8)</p>

---

## Association Rule (max-SINR)

We use a max-SINR association, encoded by ($W(j,n)\in{0,1}$):

$$
W(j,n)=
\begin{cases}
1 & \text{if } j = \mathop{\mathrm{argmax}}_{j^{\prime}} \Gamma(j^{\prime}, n) \\ 
0 & \text{otherwise}. 
\end{cases}
$$
<p align="right">(9)</p>

A UE is considered “served” if its SINR exceeds a threshold ($\Gamma_{th}$). For DBS ($j$):

$$
N_{j}=\sum_{n=1}^{U}W(j,n)H_{H}(\Gamma(j,n)-\Gamma_{th})
$$
<p align="right">(10)</p>

Total served users:

$$
N=\sum_{j=1}^{M}N_{j}
$$
<p align="right">(11)</p>

where ($H_H(\cdot)$) is the Heaviside step function.

---

## Environment Parameters ($\alpha$) and ($\beta$)

The LOS probability in (1) is based on an ITU-inspired geometric LOS model (see Al-Hourani 2014 and ITU-R P.1410-6). 

In short:

* ($\alpha_U$): built-up area fraction
* ($\beta_U$): average building density ($\mathrm{buildings/km^2}$)
* ($\gamma_U$): Rayleigh height scale for buildings

Al-Hourani provides a polynomial surface fit to map (($\alpha_U,\beta_U,\gamma_U$)) → ($\alpha,\beta$):

$$
z=\sum_{j=0}^{3}\sum_{i=0}^{3-j}C_{ij}(\alpha_{U}\beta_{U})^{i}\gamma_{U}^{j}
$$
<p align="right">(12)</p>

If you have a detailed city model, you can compute ($\alpha,\beta$) from those parameters. Otherwise, treat
($\alpha,\beta$) as **scenario constants** (urban/suburban/rural presets).

**Surface polynomial coefficients** for ($\alpha$) and ($\beta$) are listed in Al-Hourani (2014).

---

## Urban Macro (UMa) Path Loss Model (TR 38.901)

For ground-to-ground (G2G) links, the 3GPP TR 38.901 Urban Macro (UMa) path loss model is implemented. While the air-to-ground model uses fixed excess losses, the UMa model relies on a more granular evaluation of Line-of-Sight (LOS) probability, breakpoint distances, and specific corrections for base station ($h_{BS}$) and user equipment ($h_{UT}$) heights.

### LOS Probability

Based on **Table I** of TR 38.901, the line-of-sight probability ($P_{\text{LOS}}$) is a function of the 2D horizontal distance ($d_{2D}$) in meters and the user equipment height ($h_{UT}$):

$$
P_{\text{LOS}} = 
\begin{cases} 
1 & \text{if } d_{2D} \le 18 \text{ m} \\ 
\left[ \frac{18}{d_{2D}} + \exp\left(-\frac{d_{2D}}{63}\right)\left(1-\frac{18}{d_{2D}}\right) \right] \left[ 1 + C'(h_{UT}) \frac{5}{4} \left(\frac{d_{2D}}{100}\right)^3 \exp\left(-\frac{d_{2D}}{150}\right) \right] & \text{if } d_{2D} > 18 \text{ m} 
\end{cases}
$$
<p align="right">(13)</p>

where the height-dependent adjustment term ($C'(h_{UT})$) captures the likelihood of clearance over clutter:

$$
C'(h_{UT}) = 
\begin{cases} 
0 & \text{if } h_{UT} \le 13 \text{ m} \\ 
\left(\frac{h_{UT}-13}{10}\right)^{1.5} & \text{if } h_{UT} > 13 \text{ m} 
\end{cases}
$$
<p align="right">(14)</p>

### Effective Environment Height ($h_E$)

The effective environment height ($h_E$) accounts for surrounding buildings modifying the effective antenna heights. Per 3GPP standards, this is a random variable. Let $C$ be the entire adjustment term from the right side of Equation 13: $C = C'(h_{UT}) \frac{5}{4} (\dots)$. 

Physically, $h_E$ is 1.0 m if the link is relatively clear. If obstructed, it is sampled from discrete building heights starting at 12 m and increasing in 3-meter increments (representing standard floor heights), bounded to remain at least 1.5 m below the UE.

Our code supports two modes for $h_E$:

**1. Probabilistic Mode (Monte Carlo simulations):**
$$
h_E =
\begin{cases}
1.0 \text{ m} & \text{with probability } \frac{1}{1+C} \\
\text{Uniform Discrete} \{12.0, 15.0, 18.0, \dots, \max(h_{UT}-1.5, 12.0)\} & \text{with probability } 1 - \frac{1}{1+C}
\end{cases}
$$

**2. Expected Mode (Mean calculation for link budget):**
To calculate the expected path loss directly without stochastic sampling, we use the expected value of this distribution:

$$
\mathbb{E}[h_E] = \left(\frac{1}{1+C}\right) \cdot 1.0 + \left(1 - \frac{1}{1+C}\right) \cdot \frac{12.0 + \max(h_{UT}-1.5, 12.0)}{2}
$$
<p align="right">(15)</p>

### Breakpoint Distance and Base Path Loss

The 2D breakpoint distance ($d'_{BP}$) governs the transition between free-space-like decay and faster decay due to ground reflections (using $f_c$ in **Hz**):

$$
d'_{BP} = \max\left(4 \frac{(h_{BS} - h_E)(h_{UT} - h_E) f_c}{c}, 10^{-6}\right)
$$
<p align="right">(16)</p>

Based on **Table II** of TR 38.901, the path loss components use the 3D distance ($d_{3D}$) in meters and carrier frequency ($f_{c,\text{GHz}}$) in **GHz**:

**LOS Path Loss:**

$$
PL_{1} = 28.0 + 22.0 \log_{10}(d_{3D}) + 20.0 \log_{10}(f_{c,\text{GHz}})
$$
<p align="right">(17)</p>

$$
PL_{2} = 28.0 + 40.0 \log_{10}(d_{3D}) + 20.0 \log_{10}(f_{c,\text{GHz}}) - 9.0 \log_{10}\left((d'_{BP})^2 + (h_{BS} - h_{UT})^2\right)
$$
<p align="right">(18)</p>

$$
PL_{\text{LOS}} = 
\begin{cases}
PL_{1} & \text{if } d_{2D} \le d'_{BP} \\
PL_{2} & \text{if } d_{2D} > d'_{BP}
\end{cases}
$$
<p align="right">(19)</p>

**NLOS Path Loss:**

$$
PL'_{\text{NLOS}} = 13.54 + 39.08 \log_{10}(d_{3D}) + 20.0 \log_{10}(f_{c,\text{GHz}}) - 0.6(h_{UT} - 1.5)
$$
<p align="right">(20)</p>

NLOS is naturally bounded by the LOS path loss:

$$
PL_{\text{NLOS}} = \max(PL_{\text{LOS}}, PL'_{\text{NLOS}})
$$
<p align="right">(21)</p>

### Final UMa Path Loss and Shadowing

The implementation branches based on whether a deterministic LOS state is provided by the simulator. 

*Optional Shadow Fading:* If `enable_shadowing` is true, zero-mean Gaussian variables $\mathcal{N}(0, \sigma_{\text{LOS}}^2)$ and $\mathcal{N}(0, \sigma_{\text{NLOS}}^2)$ (configured via `uma_cfg`) are added to $PL_{\text{LOS}}$ and $PL_{\text{NLOS}}$, respectively.

**Branch 1: Deterministic Mask Provided**
If an explicit `is_los` boolean array is passed (e.g., from a ray-tracer or an external drop state), the code assigns the exact state:

$$
L_{\text{UMa}}^{[\mathrm{dB}]}(j,n) = 
\begin{cases}
PL_{\text{LOS}} & \text{if } \text{is\\_los}(j,n) \text{ is True} \\
PL_{\text{NLOS}} & \text{if } \text{is\\_los}(j,n) \text{ is False}
\end{cases}
$$
<p align="right">(22)</p>

**Branch 2: Probabilistic Link Budget**
If no explicit LOS mask is provided, the code computes the expected path loss by weighting the components using the LOS probability from Equation 13. This is useful for large-scale association and capacity studies where average throughput is required:

$$
L_{\text{UMa}}^{[\mathrm{dB}]}(j,n) = P_{\text{LOS}} \cdot PL_{\text{LOS}} + (1 - P_{\text{LOS}}) \cdot PL_{\text{NLOS}}
$$
<p align="right">(23)</p>

## References

1. C. Yan, L. Fu, J. Zhang, and J. Wang, “A Comprehensive Survey on UAV Communication Channel Modeling,” *IEEE Access*, 2019.
2. A. A. Khuwaja et al., “A Survey of Channel Modeling for UAV Communications,” *IEEE Communications Surveys & Tutorials*, 2018.
3. A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP altitude for maximum coverage,” *IEEE Wireless Communications Letters*, 2014.
4. ITU-R P.1410-6 (08/2023), “Propagation data and prediction methods required for the design of terrestrial broadband radio access systems…”
5. 3GPP, "Study on channel model for frequencies from 0.5 to 100 GHz," 3rd Generation Partnership Project (3GPP), Technical Report (TR) 38.901, V16.1.0, Jan. 2020.