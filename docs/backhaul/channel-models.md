# 1. FSO Channel Model (Backhaul Layer)

To model the wireless backhaul layer in the simulations, we employ Free-Space Optical (FSO) links.  
The implemented model follows the statistical framework proposed in [1].  
Although FSO is used in this work, the proposed algorithms are applicable to any wireless backhaul technology (e.g., mmWave). The only difference lies in the resulting achievable throughput between base station (BS) pairs.

The numbering of the equations below directly corresponds to the implementation in `bh_channel_model.py`, `bh_mmwave_channel_model.py`, `bh_layer.py`.

---

## 1.1 Total FSO Channel Gain

The overall FSO channel gain is modeled as in [1]:

$$
h_{\text{FSO}} = \eta \, h_p \, h_a \, h_g \qquad (1) 
$$

where:

- $\eta$ — photodetector responsivity  
- $h_p$ — atmospheric attenuation loss  
- $h_a$ — atmospheric turbulence-induced fading  
- $h_g$ — geometric and misalignment loss (GML)

Following [1], the expected value of atmospheric turbulence is adopted:

$$
h_a = \mathbb{E}[h_a] = 1
$$

This assumption is justified when adaptive optics (AO) techniques are employed to mitigate turbulence effects [2].

---

## 1.2 Atmospheric Attenuation

Denote the distance between BSs $i$ and $l$ as $d(i,l)$ (in meters).  
The atmospheric attenuation is given by:

$$
h_p = 10^{-\kappa d / 10} \qquad(2)
$$

where:

- $\kappa$ — weather-dependent attenuation coefficient  
- $d$ — link distance  

---

## 1.3 Beam Width at Distance

The beam width at distance $d$ is computed as:

$$
\omega_d =
\omega_0
\sqrt{
1 +
\left(1+\frac{2\omega_0^2}{\rho^2(d)}\right)
\left(\frac{\lambda d}{\pi \omega_0^2}\right)^2
} \qquad (3)
$$

where:

- $\omega_0$ — beam waist radius  
- $\lambda$ — optical wavelength  
- $\rho(d)$ — atmospheric coherence length  

---

## 1.4 Coherence Length

The coherence length is defined as:

$$
\rho(d) =
\left(
0.55 C_n^2 k^2 d
\right)^{-3/5} \qquad (4)
$$

with

$$
k = \frac{2\pi}{\lambda} \qquad (5)
$$

and

$$
C_n^2 \approx C_0^2 e^{-h/100} \qquad (6)
$$

where:

- $C_0^2 = 1.7 \times 10^{-14}$ is the ground refractive index structure parameter  
- $h$ — UAV/DBS altitude  

---

## 1.5 Maximum Captured Power Fraction

Define the normalized aperture parameter:

$$
v_1 =
\frac{r_0}{\omega_d}
\sqrt{\frac{\pi}{2}} \qquad (7)
$$

The maximum fraction of optical power captured by the receiver lens is:

$$
A_0 = \mathrm{erf}(v_1)\,\mathrm{erf}(v_2) \qquad (8)
$$

where $r_0$ denotes the receiver lens radius.

---

## 1.6 Geometric and Misalignment Loss (GML)

The geometric and misalignment loss is approximated as:

$$
h_g =
A_0
\exp
\left(
-\frac{2 u^2}{\zeta \omega_d^2}
\right) \qquad (9)
$$

where:

- $u$ — misalignment distance  
- $\zeta$ — geometric correction parameter  

Following [1], statistical misalignment variations are modeled via Gaussian perturbations of position and orientation.

---

## 1.7 Misalignment Fluctuation Parameters

The fluctuation parameters are defined as:

$$
\lambda_1 = \sigma_y^2 + d^2 \sigma_\theta^2 \qquad (10)
$$

$$
\lambda_2 = \sigma_z^2 + d^2 \sigma_\phi^2 \qquad (11)
$$

where:

- $\sigma_y, \sigma_z$ — position variances  
- $\sigma_\theta, \sigma_\phi$ — orientation variances  

---

## 1.8 Achievable FSO Rate

The achievable FSO rate is given by [1]:
 
$$
R_{\text{FSO}} =
\frac{1}{2}
\log_2
\left(
\frac{e}{2\pi}
\eta^2 h_p^2
\bar{\gamma}
A_0^2
\right)
\frac{2}{\zeta \omega_d^2 \ln(2)}
(\lambda_1 + \lambda_2) \qquad (12)
$$

where the transmit SNR is

$$
\bar{\gamma} =
\frac{P_{\text{FSO}}^2}{\sigma_n^2} \qquad (13)
$$

with:

- $P_{\text{FSO}}$ — FSO transmit power  
- $\sigma_n^2$ — receiver noise power  

---

## 1.9 Distance Constraint

To enforce a maximum link distance $d_{\max}$, the final achievable throughput between BSs $i$ and $l$ is:

$$
R(i,l) =
\begin{cases}
R_{\text{FSO}}(i,l), & d(i,l) \le d_{\max} \\
0, & \text{otherwise}
\end{cases} \qquad (14)
$$

If the distance constraint is violated, the backhaul link is considered unavailable.

---
## 1.10 Implementation Note: Height Information

The statistical FSO model requires the **actual node heights** in order to correctly estimate the refractive index structure parameter used in the atmospheric turbulence model in (6).

In the simulator, the backhaul layer receives the **3D link distance** together with the node heights from the link layer.

Given the full 3D distance between nodes $i$ and $l$:

$$
d(i,l)
$$

and their heights

$$
h_i, \quad h_l
$$

the horizontal distance used to construct the temporary coordinates for the FSO model is reconstructed as

$$
d_{\text{hor}} =
\sqrt{
d(i,l)^2 - (h_i - h_l)^2
}
\qquad (15)
$$

Temporary coordinates are then defined as

$$
\mathbf{p}_i = (0,0,h_i),
\qquad
\mathbf{p}_l = (d_{\text{hor}},0,h_l)
\qquad (16)
$$

These coordinates are passed to the statistical FSO channel model so that the refractive index parameter in (6)

$$
C_n^2 \approx C_0^2 e^{-h/100}
$$

is computed using the correct altitude values.

This modification ensures that the simulator properly accounts for **altitude-dependent atmospheric turbulence effects** as described in \[1\].

---
## 2. mmWave Channel Modeling

To model the mmWave channel between base stations, a LOS-only path loss model is used.  
The path loss in dB is computed as

$$
L^{\mathrm{[dB]}}_{\text{mm}}(i,l) =
32.4 +
20 \log_{10}(f) +
10 \alpha_{\text{mm}} \log_{10}\left(d(i,l)\right)
\qquad (17)
$$

where

- $f$ — carrier frequency in GHz  
- $\alpha_{\text{mm}}$ — path loss exponent  
- $d(i,l)$ — distance between base stations $i$ and $l$ in meters

The constant term is given by

$$
\mathrm{FSPL}_{1\text{m}} = 32.4 + 20\log_{10}(f)
\qquad (18)
$$

which corresponds to the free-space path loss at a reference distance of 1 meter.

In the implemented model, the path loss exponent is set to

$$
\alpha_{\text{mm}} = 2.13
\qquad (19)
$$

for mmWave transmissions at 38 GHz.

---
## 2.1 Received Power

The received power is computed from the path loss in linear scale as

$$
P_r = \frac{P^{\mathrm{TX}}}{L_i^{\mathrm{linear}}}
\qquad (20)
$$

where

- $P^{\mathrm{TX}}$ — transmission power of the backhaul transceivers  
- $L_i^{\mathrm{linear}}$ — path loss in linear scale computed from (17)

---

## 2.2 Achievable mmWave Throughput

The throughput of the $i$-th backhaul link is computed using the modified Shannon formula:

$$
C_i =
\eta^B B^{\mathrm{eff}}
\log_2
\left(
1 +
\frac{P^{\mathrm{TX}}}
{L_i^{\mathrm{linear}} \sigma^2}
\right)
\qquad (21)
$$

where

- $\eta^B$ — throughput efficiency of the system  
- $B^{\mathrm{eff}}$ — effective bandwidth  
- $P^{\mathrm{TX}}$ — transmission power of the backhaul transceivers  
- $\sigma^2$ — noise power at the backhaul receivers  
- $L_i^{\mathrm{linear}}$ — path loss in linear scale

---

## 2.3 Model Selection

The simulator supports multiple backhaul channel models.  
The active backhaul model is selected as

$$
\mathcal{M}_{\mathrm{BH}} =
\begin{cases}
\mathcal{M}_{\mathrm{FSO}}, & c = 0 \\
\mathcal{M}_{\mathrm{mmWave}}, & c = 1
\end{cases}
\qquad (22)
$$

where

- $c$ — backhaul channel model selection parameter  
- $\mathcal{M}_{\mathrm{FSO}}$ — Free-Space Optical backhaul model  
- $\mathcal{M}_{\mathrm{mmWave}}$ — millimeter-wave backhaul model

---

## 2.4 Implementation Note

The mmWave model is implemented as a separate module from the FSO channel model.  
In the current implementation, the mmWave backhaul model assumes **LOS for all backhaul links** and does not include an NLOS case.

The backhaul layer selects the appropriate model and stores the resulting:

- path loss
- received power
- achievable throughput

for each available backhaul link.

---
# References

[1] M. Najafi, H. Ajam, V. Jamali, P. D. Diamantoulakis, G. K. Karagiannidis, and R. Schober, “Statistical modeling of the FSO fronthaul channel for UAV-based communications,” *IEEE Transactions on Communications*, vol. 68, no. 6, pp. 3720–3736, 2020.

[2] Y. Kaymak, R. Rojas-Cessa, J. Feng, N. Ansari, M. Zhou, and T. Zhang, “A survey on acquisition, tracking, and pointing mechanisms for mobile free-space optical communications,” *IEEE Communications Surveys & Tutorials*, vol. 20, no. 2, pp. 1104–1123, 2018.

[3] T. S. Rappaport, F. Gutierrez, E. Ben-Dor, J. N. Murdock, Y. Qiao, and J. I. Tamir, “Broadband Millimeter-Wave Propagation Measurements and Models Using Adaptive-Beam Antennas for Outdoor Urban Cellular Communications,” IEEE Transactions on Antennas and Propagation, vol. 61, no. 4, pp. 1850–1859, Apr. 2013. 

[4] P. Mogensen, W. Na, I. Z. Kovacs, F. Frederiksen, A. Pokhariyal, K. I. Pedersen, T. Kolding, K. Hugl, and M. Kuusela, “LTE Capacity Compared to the Shannon Bound,” in 2007 IEEE 65th Vehicular Technology Conference - VTC2007-Spring. Dublin, Ireland: IEEE, Apr. 2007, pp. 1234–1238
