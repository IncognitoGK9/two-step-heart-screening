import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "stix"
EQ = {
 "eq1": r"$\mathrm{Acc}(f)=\dfrac{TP+TN}{TP+TN+FP+FN}$",
 "eq2": r"$w_m=\dfrac{\exp\left((a_m-a_{\max})/\tau\right)}{\sum_{j=1}^{M}\exp\left((a_j-a_{\max})/\tau\right)}$",
 "eq3": r"$\hat{p}(\mathbf{x})=\sum_{m=1}^{M} w_m\,p_m(\mathbf{x}),\qquad \hat{y}(\mathbf{x})=\mathbf{1}\left[\hat{p}(\mathbf{x})\geq 0.5\right]$",
 "eq4": r"$\hat{p}_{\mathrm{stack}}(\mathbf{x})=\sigma\!\left(\beta_0+\sum_{m=1}^{M}\beta_m\,\mathrm{logit}\,\tilde{p}_m(\mathbf{x})\right)$",
 "eq5": r"$q=\sigma(a\,s+b),\qquad \dfrac{q_t}{1-q_t}=\dfrac{q}{1-q}\cdot\dfrac{\pi_t/(1-\pi_t)}{\pi_s/(1-\pi_s)}$",
 "eq6": r"$\mathrm{Low\ risk\ if\ }q<t_L;\qquad \mathrm{High\ risk\ if\ }q>t_H;\qquad \mathrm{Not\ sure\ otherwise}$",
 "eq7": r"$t=\dfrac{\bar{d}}{\sqrt{\left(\dfrac{1}{J}+\dfrac{n_{\mathrm{test}}}{n_{\mathrm{train}}}\right)\hat{\sigma}^{2}}}$",
}
for k, s in EQ.items():
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, s, fontsize=12)
    fig.savefig(f"{k}.png", dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
print("equations ok")
