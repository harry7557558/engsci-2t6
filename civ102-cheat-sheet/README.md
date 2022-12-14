# Poetry

Structure engineering: *The art and science of designing and making structures with economy and elegance so they can safely resist the forces to which they may be subjected.*

Three principles of engineering
 - $F=ma$
 - You cannot push or a rope
 - To get the answer you must know the answer

3.5 sig figs, engineering notation.


## Equilibrium

$\sum F = 0$, $\sum M = 0$. FBD.

Uniform loads ($W$); Moment = moment calculated from all forces acting on the centroid; $M=WL(L/2) = \frac{1}{2} WL^2$.

Vertical load on a hanging cable: $WL/2$  
Horizontal load on a hanging cable: $\frac{WL^2}{8h}$  
Don't forget a sus bridge has 2 cables.


# Deformation

Hooke's law: $F=k\Delta L$, $k$ is axial stiffness.  
Stress: $\sigma = F/A$; Strain: $\epsilon = \Delta l / l_0$;  
$\sigma = E\epsilon$, $E$ is Young's modulus (or material stiffness);  
$F=EA\Delta l/l_0$, $k=AE/l_0$.

Stress-strain curve of low-alloy steel
 - Small strain: linearly elastic, Hooke's law applies
 - Yield: plastic deformation occurs, $\sigma_{yield}$ changes little as $\epsilon$ increases
 - Strain hardening, necking, fracture

Strain energy: area under stress-strain curve, energy per unit volume
 - Resilience: energy of the elastic part
 - Toughness: energy until breaks (elastic+plastic)

Compressive strength;  
Ductility: strain at fracture;  
Coefficient of thermal expansion (thermal strain) $10^{-6}/K$;

Allowable stress design: $\sigma_{allowed} = \sigma_{yield}/\mathrm{FoS}$; $\sigma_{applied}<\sigma_{allowed}$ = safe.

Limit states design: probability of failure is low enough.


# Dynamics

A mass on a spring being pulled downward
 - $m\ddot{x}+kx=0$
 - $x(t) = A\sin(\omega t+\phi)+\Delta_0$
 - $\omega=\sqrt{k/m}$, $f=\frac{1}{2\pi}\sqrt{k/m}$, $T=f^{-1}=2\pi\sqrt{m/k}$, $\Delta_0=mg/k$, $A$ and $\phi$ depends on initial conditions
 - $f\approx 15.76/\sqrt{\Delta_0}$, $f$ in Hz and $\Delta_0$ in mm

Free vibration
 - $m\ddot{x} + c\dot{x} + kx = mg$
    - $c=2\beta\sqrt{mk}$, $\beta$ is the fraction compared to critical damping
 - $x(t) = A e^{-\beta\omega_n t} \sin(\omega_n t \sqrt{1-\beta^2} + \phi) + \Delta_0$
 - Single DoF: $f\approx15.76/\sqrt{\Delta_0}$
 - Multi-DoF: $f_n\approx17.76/\sqrt{\Delta_0}$, use $\Delta_0$ in the middle

Forced vibration
 - Force $F(t) = F_0\sin(\omega_t)+mg$
 - Steady state: $x(t) = DAF\cdot F_0/k \sin(\omega t + \phi) + \Delta_0$
 - $DAF = 1/\operatorname{hypot}(1-(f/f_n)^2, 2\beta f/f_n)$
  - $f$ is based on $\omega$, $f_n$ is resonance frequency
  - $f/f_n=0$: DAF = 1; $f/f_n=1$: goes to infinity for $\beta=0$
 - Experienced force: $mg+DAF\cdot F_0$


# Geometry

Centroidal axis: $\bar{y}=\sum{Ay}/\sum{A}$

Second moment of area $I$: $\int y^2 dA$ where $y$ is related to $\bar{y}$

Second moment of area of a rectangle with width $b$ and height $h$: $bh^3/12$
 - Add/subtract primitives
 - After translation: $I=I_{\bar{y}}+Ad^2$

Bending of beams: $\phi=d\theta/dl$, $r=\phi^{-1}$, $\epsilon=\phi y$, $M=EI\phi$

First moment of area $Q(y)$: $\int y dA$, $y$ is related to $\bar{y}$, integral starts from the top/bottom
 - Maximized at $y=\bar{y}$



# Buckling

FoS = $3$

Euler buckling load: $\pi^2EI/L^2$
 - Use the direction with the smallest $I$

Buckling load of thin plates: $\frac{k\pi^2}{12(1-\mu^2)}\left(\frac{t}{b}\right)^2$
 - Free edges: use the Euler buckling formula
 - Two fixed edges + uniform stress: $k=4$
 - One fixed edge + uniform stress: $k=0.425$
 - Two fixed edges + "triangle" stress: $k=6$


# Truss

Truss design iteration
 - Geometry
 - Determine applied loads
 - Analyze internal forces
 - Select members
 - Determine maximum displacement
 - Check dynamic properties

Applied loads: Uniformly distributed load equally distributed on each joint by area/length

Solve for reaction forces using equilibrium;  
Method of sections: isolate, F/M equlibrium  
Method of joints: start at one end and solve for reactions for each member  
Positive for tension, negative for compression

Slenderness ratio $r=\sqrt{I/A}$  
Select members: FoS = 2, $L/r<200$, $\sigma_y=350\mathrm{MPa}$ if not given;

Wind pressure: $1/2 \rho v^2 c_D$, $c_D=1.5$ for bridge;  
$W_{wind}=2.0\mathrm{KPa}$;
Area to consider: truss, handrail, etc.; Only consider one face.

Truss deflection
 - Solve the truss, $F$ for each member
 - Each member's elongation $\Delta L = \epsilon L = FL/EA$
 - Apply a dummy load $P^*$ in the same direction of deflection to solve for
 - Solve the truss under the dummy load, each $F^*$
 - Virtual work $P^*\delta = \sum F^* \Delta L$, solve for deflection $\delta$


# Beam

Axial $N(x)$, shear $V(x)$, bending $M(x)$, deflection $\delta(x)$

Solve for reaction forces;  
SFD: integrate applied loads (including reaction forces, up is positive), endpoints are zero; positive y up;  
BMD: integrate SFD, endpoints are zero; Bottom tension = positive, positive y down;

Flectural stress: $\sigma = My/I$

Beam deflection
 - $\phi=M/EI$, integrate $\phi$ -> $\theta=\frac{dy}{dx}$, integrate $\theta$ -> $\delta$
 - Moment area theorem 1: $\Delta\theta$ = area under $\phi$
 - Moment area theorem 2: tangential deviation equals the area under the $Mx/EI$ graph

Shear: $\tau=VQ/Ib$
 - Definition: force divided by parallel area
 - In the material vs. At glue/nail joints


# Concrete

FoS=2.0 for both flexural and shear. Use $\sigma_{ys}=400\mathrm{MPa}$ unless otherwise stated.

Reinforced concrete beam, maximum width $b$, minimum width $b_w$, height $h$, distance from max compression to the centroid of tensile reinforcement steels $d$, stirrups (shear reinforcement) spacing $s$.

Concrete compressive strength in MPa $f_c'$, $E_c=4500\sqrt{f_c'}$ in MPa.

$n=E_s/E_c$, $\rho=A_s/bd$, $k=\sqrt{(n\rho)^2+2n\rho}-n\rho$, $j=1-k/3$, all dimensionless.

Experienced stress $\sigma_{s}=\dfrac{M}{A_s jd}$, $\sigma_{c}=\dfrac{M}{A_sjd}\dfrac{k}{1-k}\frac{1}{n}$.

$d_v=0.9d$. Yield strength $f_y=\sigma_y$.

Shear $V_{max}=0.25f_c'b_wd_v$, without stirrups $V_c=\dfrac{230\sqrt{f_c'}}{1000+d_v}b_wd_v$, with stirrups $V_c=0.18\sqrt{f_c'}b_wd_v$;  
$V_s=\dfrac{A_vf_yd_v}{s}\cot35^\circ$, $V_t=V_c+V_s$. Divide by FoS for safe $V$.

Minimum $s$: $s=\dfrac{A_vf_y}{0.06\sqrt{f_c'}b_w}$, or $\dfrac{A_vf_y}{b_ws}\ge0.06\sqrt{f_c'}$.

Safe $s$: $s=\dfrac{\frac{1}{2.0}A_vf_yd_v\cot35^\circ}{V-\frac{1}{2.0}0.18\sqrt{f_c'}b_wd_v}$

Evaluating concrete:
 - SFD, BMD
 - Check if $s$ is minimum
 - Use $v_c$ without strirrups when $s$ is bigger and with stirrups when $s$ is smaller
 - Calculate $V_s$, $V_t$, $V_{max}$, capacity $V=\min(V_{max}, V_t)$

Concrete design
 - SFD, BMD
 - Check if $V_{max}/2.0$ works, else change $b_w$ and/or $d$
 - Check if $V_c/2.0$ without stirrups works
 - Calculate $V_s$ with minimum $s$, calculate $V_c$ and $V_t$ and see if it works
 - Calculate safe $s$, reiterate the previous step for a good $s$


## Timber

Anisotropic material, stiffer + higher strength at "vertical" direction in a tree.

Use the 5th-percentile strength in design with $\mathrm{FoS}=1.5$.

Use the 50th-pencentile strength in deflection calculation.

