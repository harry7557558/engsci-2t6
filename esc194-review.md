<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>

$\newcommand{\e}{\operatorname{e}}$

$\newcommand{\d}{\mathrm{d}}$
$\newcommand{\dx}{\d{x}}$
$\newcommand{\dy}{\d{y}}$
$\newcommand{\du}{\d{u}}$
$\newcommand{\dt}{\d{t}}$

**If you are reading this, please note that I only include content that I think worth reviewing and not everything covered in the course.**

<hr/><br/>


# Differential equations

Separable differential equations;

First-order linear differential equations
 - $y'+P(x)y=Q(x)$
 - Find $I(x)=\e^{\int P(x)\dx}$
 - $I(x)y'+I(x)P(x)y=I(x)Q(x)$
 - $\frac{\d}{\dx}I(x)y=Q(x)$
 - $I(x)y=\int Q(x)\dx+C$

Logistic model for population growth
 - $\dfrac{\d P}{\dt}=kP\left(1-\dfrac{P}{M}\right)$
 - $P(t)=\dfrac{M}{1+A\e^{-kt}}$, $A=\dfrac{M-P_0}{P_0}$.

Homogeneous second order linear ODE with constant coefficients
 - $y''+ay'+by=0$, $a,b\in R$
 - $y=\e^{rx} \implies (r^2+ar+b)\e^{rx}=0$
 - $r=\dfrac{-a\pm\sqrt{a^2-4b}}{2}$
 - Two real roots, $a^2-4b>0$
   - $y=C_1\e^{r_1x}+C_2\e^{r_2x}$
 - One root, $a^2-4b=0$
   - $y=C_1\e^{rx}+C_2x\e^{rx}$
 - Two complex roots, $a^2-4b<0$
   - $r=\alpha\pm\beta i$, where $\alpha=-\frac{1}{2}a, \beta=\frac{1}{2}\sqrt{4b-a^2}$
   - $y=\e^{\alpha x}(C_1\cos(\beta x)+C_2\sin(\beta x))$

Nonhomogeneous linear equation
 - $y''+ay'+by=\phi(x)$
 - General solution: $y(x)=y_p(x)+y_c(x)$
 - $y_c(x)$ is the solution to $y''+ay'+by=0$
 - Method of undetermined coefficients: guess $\phi(x)$ and determine the coefficients
 - Method of variation of parameters
   - Solve $y_c(x)=C_1y_1(x)+C_2y_2(x)$
   - Let the $y_p(x)=u_1(x)y_1(x)+u_2(x)y_2(x)$
   - It can be proven $u_1'y_1'+u_2'y_2'=\phi(x)$
   - Let $u_1'y_1+u_2'y_2=0$, solve the 2×2 system for $u_1'$ and $u_2'$
   - Integrate $u_1'$, $u_2'$, constant is already in $y_c$.


# $\delta$-$\epsilon$ proof

$\lim\limits_{x\rightarrow c}f(x)=L$: for any $\delta>0$, exists $\epsilon>0$, such that $|f(x)-L|<\epsilon$ when $|x-c|<\delta$.
 - Infinity: replace $||<\epsilon/\delta$ with $>M$

General proof for $\lim\limits_{x\rightarrow c}f(x)=L$:
 - Given $\epsilon>0$, algebraically find $\delta$
 - LHS: $|f(x)-L|<\epsilon$
 - RHS: $|x-c|<\delta$

Prove $\lim\limits_{x\rightarrow5}{x^2}=25$:
 - $|x^2-25|<\epsilon$ when $|x-5|<\delta$
 - Specify $\delta<1$, then $|x-5|<1$, $|x^2-25|<11$
 - $\epsilon = \min(\delta/11, 1)$


# Definitions and theorems

Sandwich limit theorem;

Limit exists: left limit equals right limit;

Continuous at $x=c$: $\lim\limits_{x\rightarrow c}f(x)=f(c)$;  
Continuous in $(a,b)$: continuous for all $x\in(a,b)$;  
Continuous in $[a,b]$: continuous in $(a,b)$ + use left/right limit at endpoints;

Derivative: $\lim\limits_{h\rightarrow 0}f(x)=\dfrac{f(x+h)-f(x)}{h}$;
Differentiable in $(a,b)$, $[a,b]$: similar to continuity;  Differentiability guarantees continuity.

Intermediate value theorem: continuous in $[a,b]$ $\implies$ exists $c\in(a,b)$ such that $f(c)$ is between $f(a)$ and $f(b)$

Extreme value theorem: continuous in $[a,b]$ $\implies$ there's a max/min in $[a,b]$

Mean value theorem: Continuous in $[a,b]$ + differentiable in $(a,b)$ $\implies$ exists $c\in(a,b)$ such that $f'(c)=\dfrac{f(b)-f(a)}{b-a}$

Delta-epsilon definition of definite integral: $\max(||\Delta x_i||)<\delta$ $\implies$ $|I-\sum f(x_i^\ast)\Delta x_i|<\epsilon$

Piecewise continuity: allows a finite number of jump discontinuities in the interval;  
Piecewise continuity in $[a,b]$ guarantees integrability in $[a,b]$.

$\left|\int_a^b f(x)\dx\right| \le \int_a^b |f(x)|\dx$

Foundamental theorem of calculus; Requires $F(x)$ to be continuous in $[a,b]$ and differentiable in $(a,b)$.


# Trigonometric functions

| | $\sin(x)$ | $\cos(x)$ | $\tan(x)$ | $\csc(x)$ | $\sec(x)$ | $\cot(x)$ |
| --- | --- | --- | --- | --- | --- | --- |
| $\frac{\d}{\dx}$ | $\cos(x)$ | $-\sin(x)$ | $\sec^2(x)$ | $-\csc(x)\cot(x)$ | $\sec(x)\tan(x)$ | $-\csc^2(x)$ |
| $\int\dx$ | $-\cos(x)$ | $\sin(x)$ | $-\ln\|\cos(x)\|$ | $-\tanh^{-1}(\cos(x))$ | $\tanh^{-1}\sin(x)$ | $\ln\|\sin(x)\|$ |

$\displaystyle \int\frac{1}{a^2+x^2}\dx=\frac{1}{a}\tan^{-1}\left(\frac{x}{a}\right)+C$,
$\displaystyle\int\frac{1}{a^2-x^2}\dx=\frac{1}{a}\tanh^{-1}\left(\frac{x}{a}\right)+C$,

$\displaystyle \int\frac{1}{\sqrt{a^2+x^2}}\dx=\sinh^{-1}\left(\frac{x}{|a|}\right)+C$,
$\displaystyle \int\frac{1}{\sqrt{a^2-x^2}}\dx = \sin^{-1}\left(\frac{x}{|a|}\right)+C.$