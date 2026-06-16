#!/usr/bin/env python3
"""
后处理: 读取 FEM CSV 输出 → 绘制 sci 顶刊标准图 (PNG + PDF)
使用 scienceplots + ieee 样式
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots  # noqa: F401
import os, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 全局样式设置
# ============================================================
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 配色 (Nature / Science 常用方案)
C0 = '#0C5DA5'  # blue
C1 = '#D43E2A'  # red
C2 = '#00A676'  # green
C3 = '#F2A900'  # gold
C4 = '#7B2D8E'  # purple
C5 = '#00B4D8'  # cyan
colors = [C0, C1, C2, C3, C4, C5]

OUT_DIR = 'output'


def save_figure(fig, name):
    """保存 PNG 和 PDF"""
    png_path = f'{name}.png'
    pdf_path = f'{name}.pdf'
    fig.savefig(png_path, format='png')
    fig.savefig(pdf_path, format='pdf')
    print(f'  已保存: {png_path}, {pdf_path}')


# ============================================================
# 读取 CSV 数据
# ============================================================
freq_df   = pd.read_csv(f'{OUT_DIR}/frequencies.csv')
modes_df  = pd.read_csv(f'{OUT_DIR}/mode_shapes.csv')
resp_df   = pd.read_csv(f'{OUT_DIR}/response_C.csv')
mass_df   = pd.read_csv(f'{OUT_DIR}/modal_masses.csv')
params_df = pd.read_csv(f'{OUT_DIR}/beam_params.csv')

# 提取参数
params = {row['parameter']: row['value'] for _, row in params_df.iterrows()}
ell     = float(params['l'])
v0      = float(params['v0'])
m_mass  = float(params['m'])
k_spr   = float(params['k'])
EI      = float(params['EI'])
mu      = float(params['mu'])

# 阵列
betas   = freq_df['beta'].values[:8]
omegas  = freq_df['omega_rad_s'].values[:8]
freqs   = freq_df['freq_Hz'].values[:8]
x_modes = modes_df['x_m'].values
n_modes_plot = min(6, len(modes_df.columns) - 1)
phi_mat = np.zeros((len(x_modes), n_modes_plot))
for n in range(n_modes_plot):
    phi_mat[:, n] = modes_df[f'phi_{n+1}'].values

t_resp  = resp_df['t_s'].values
w_resp  = resp_df['displacement_m'].values
v_resp  = resp_df['velocity_m_s'].values
a_resp  = resp_df['acceleration_m_s2'].values

M_n     = mass_df['M_n'].values[:n_modes_plot]
M_dist  = mass_df['M_distributed'].values[:n_modes_plot]
M_conc  = mass_df['M_concentrated'].values[:n_modes_plot]

print('CSV 数据加载完毕。')
print(f'  模态数: {n_modes_plot}, 空间点: {len(x_modes)}, 时间点: {len(t_resp)}')

# ============================================================
# 图 1: FEM vs 传递矩阵法频率对比
# ============================================================
# 传递矩阵法结果 (来自前述解析解)
tmm_omegas = np.array([187.3269, 308.5324, 477.3678, 1113.0062, 1840.1377, 2500.7848])
tmm_freqs  = tmm_omegas / (2 * np.pi)

print('绘制图 1: FEM-TMM 频率对比...')
fig1, ax1 = plt.subplots(1, 1, figsize=(6, 3.8))
n_comp = min(len(omegas), len(tmm_omegas))
x_pos = np.arange(1, n_comp + 1)
w_bar = 0.35
ax1.bar(x_pos - w_bar/2, freqs[:n_comp], w_bar,
        color=C0, alpha=0.85, edgecolor='white', linewidth=0.3,
        label='FEM (40 elements)')
ax1.bar(x_pos + w_bar/2, tmm_freqs[:n_comp], w_bar,
        color=C1, alpha=0.85, edgecolor='white', linewidth=0.3,
        label='TMM (analytical)')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'{i}' for i in range(1, n_comp + 1)])
ax1.set_xlabel('Mode order $n$', fontsize=11)
ax1.set_ylabel('Natural frequency $f_n$ [Hz]', fontsize=11)
ax1.set_title('FEM vs Transfer Matrix Method — Frequency Comparison',
              fontsize=12, fontweight='bold')
ax1.legend(frameon=True, fancybox=False, edgecolor='gray', fontsize=9)
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(labelsize=9)
fig1.tight_layout()
save_figure(fig1, 'fig_fem_1_freq_compare')
plt.close(fig1)

# ============================================================
# 图 2: FEM 振型函数 (前 6 阶)
# ============================================================
print('绘制图 2: 振型函数...')
fig2, axes2 = plt.subplots(2, 3, figsize=(10, 6.5))
axes2 = axes2.flatten()

# 标注特殊位置
special_x = [0, ell, 2*ell, 3*ell, 4*ell]
special_labels = ['A', 'B', 'C', 'D', 'E']

for n in range(n_modes_plot):
    ax = axes2[n]
    ax.plot(x_modes, phi_mat[:, n], color=colors[n], linewidth=1.2)
    ax.fill_between(x_modes, 0, phi_mat[:, n], color=colors[n], alpha=0.12)

    # 标注特殊位置
    for sx, sl in zip(special_x, special_labels):
        ax.axvline(x=sx, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
        ax.text(sx, ax.get_ylim()[1] * 0.92, sl, fontsize=7, ha='center',
                color='gray', fontweight='bold')

    ax.set_title(f'Mode {n+1}  ($f_{{{n+1}}}$ = {freqs[n]:.1f} Hz)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('$x$ [m]', fontsize=9)
    ax.set_ylabel('$\\phi_n(x)$', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)
    ax.axhline(y=0, color='black', linewidth=0.4)

fig2.suptitle('FEM Mode Shapes — Euler-Bernoulli Beam', fontsize=13, fontweight='bold', y=1.01)
fig2.tight_layout()
save_figure(fig2, 'fig_fem_2_mode_shapes')
plt.close(fig2)

# ============================================================
# 图 3: 模态质量分解
# ============================================================
print('绘制图 3: 模态质量分解...')
fig3, ax3 = plt.subplots(1, 1, figsize=(6.5, 3.8))
x_mass = np.arange(1, n_modes_plot + 1)
w_bar3 = 0.55
ax3.bar(x_mass, M_dist, w_bar3, color=C0, alpha=0.85, edgecolor='white',
        linewidth=0.3, label='Distributed $\\int \\rho S \\phi_n^2 dx$')
ax3.bar(x_mass, M_conc, w_bar3, bottom=M_dist, color=C3, alpha=0.85,
        edgecolor='white', linewidth=0.3, label='Concentrated $m\\,\\phi_n^2(2l)$')
ax3.set_xticks(x_mass)
ax3.set_xticklabels([f'{i}' for i in range(1, n_modes_plot + 1)])
ax3.set_xlabel('Mode order $n$', fontsize=11)
ax3.set_ylabel('Modal mass $M_n$ [kg]', fontsize=11)
ax3.set_title('Modal Mass Decomposition', fontsize=12, fontweight='bold')
ax3.legend(frameon=True, fancybox=False, edgecolor='gray', fontsize=9)
ax3.grid(axis='y', alpha=0.3)
ax3.tick_params(labelsize=9)
fig3.tight_layout()
save_figure(fig3, 'fig_fem_3_modal_mass')
plt.close(fig3)

# ============================================================
# 图 4: C 点响应 (位移/速度/加速度)
# ============================================================
print('绘制图 4: C 点时程响应...')
fig4, axes4 = plt.subplots(3, 1, figsize=(7, 6.5), sharex=True)

# 位移 [mm]
axes4[0].plot(t_resp, w_resp * 1000, color=colors[0], linewidth=1.0)
axes4[0].set_ylabel(r'$w(2l,t)$ [mm]', fontsize=10)
axes4[0].grid(True, alpha=0.3)
axes4[0].tick_params(labelsize=9)
axes4[0].set_title('FEM Dynamic Response at Point C ($x = 2l$)', fontsize=11, fontweight='bold')

# 速度
axes4[1].plot(t_resp, v_resp, color=colors[1], linewidth=1.0)
axes4[1].set_ylabel(r'$\dot{w}(2l,t)$ [m/s]', fontsize=10)
axes4[1].axhline(y=v0, color='gray', linewidth=0.6, linestyle=':', alpha=0.7)
axes4[1].text(t_resp[-1]*0.98, v0*1.15, f'$v_0$={v0:.1f} m/s',
              fontsize=7, ha='right', color='gray')
axes4[1].grid(True, alpha=0.3)
axes4[1].tick_params(labelsize=9)

# 加速度
axes4[2].plot(t_resp, a_resp, color=colors[2], linewidth=1.0)
axes4[2].set_xlabel(r'$t$ [s]', fontsize=11)
axes4[2].set_ylabel(r'$\ddot{w}(2l,t)$ [m/s$^2$]', fontsize=10)
axes4[2].grid(True, alpha=0.3)
axes4[2].tick_params(labelsize=9)

fig4.tight_layout()
save_figure(fig4, 'fig_fem_4_response_C')
plt.close(fig4)

# ============================================================
# 图 5: 各阶模态贡献 (频谱)
# ============================================================
print('绘制图 5: 模态贡献谱...')
fig5, ax5 = plt.subplots(1, 1, figsize=(6.5, 3.8))

# 计算各阶模态对 C 点位移的贡献
phi_C_vals = np.zeros(n_modes_plot)
for n in range(n_modes_plot):
    # 找到 x = 2l 处的振型值
    idx_C = np.argmin(np.abs(x_modes - 2*ell))
    phi_C_vals[n] = phi_mat[idx_C, n]

contribs = np.abs(m_mass * v0 * phi_C_vals / (M_n * omegas[:n_modes_plot]) * phi_C_vals) * 1000  # mm

x_stem = np.arange(1, n_modes_plot + 1)
markerline, stemlines, baseline = ax5.stem(x_stem, contribs,
    basefmt=' ', linefmt=colors[0], markerfmt='o')
markerline.set_markersize(8)
stemlines.set_linewidth(1.5)
labels_stem = [f'Mode {i+1}\n{freqs[i]:.1f} Hz' for i in range(n_modes_plot)]
ax5.set_xticks(x_stem)
ax5.set_xticklabels(labels_stem, fontsize=8)
ax5.set_ylabel(r'$|w_n(2l)|_{\max}$ [mm]', fontsize=11)
ax5.set_title('FEM — Modal Contribution to Displacement at Point C',
              fontsize=12, fontweight='bold')
ax5.tick_params(labelsize=9)
ax5.grid(axis='y', alpha=0.3)
fig5.tight_layout()
save_figure(fig5, 'fig_fem_5_modal_contrib')
plt.close(fig5)

# ============================================================
# 图 6: 梁全貌时空响应 (瀑布图)
# ============================================================
print('绘制图 6: 时空响应全貌...')

# 在时空网格上计算 w(x,t) (前 4 阶模态叠加)
n_modes_wf = min(n_modes_plot, 4)
t_wf = np.linspace(0, 0.15, 80)
x_wf = np.linspace(0, 4*ell, 400)
W_wf = np.zeros((len(x_wf), len(t_wf)))

for n in range(n_modes_wf):
    # 找到 x=2l 处的振型值
    idx_C_wf = np.argmin(np.abs(x_modes - 2*ell))
    phi_C_wf = phi_mat[idx_C_wf, n]
    coeff = m_mass * v0 * phi_C_wf / (M_n[n] * omegas[n])

    # 在密集网格上插值振型
    for i_x, xx in enumerate(x_wf):
        phi_xx = np.interp(xx, x_modes, phi_mat[:, n])
        W_wf[i_x, :] += phi_xx * coeff * np.sin(omegas[n] * t_wf)

fig6, ax6 = plt.subplots(1, 1, figsize=(7.5, 5.5))
T_grid, X_grid = np.meshgrid(t_wf * 1000, x_wf)  # t in ms
cf = ax6.contourf(T_grid, X_grid, W_wf * 1000, levels=60, cmap='RdBu_r')  # mm
ax6.contour(T_grid, X_grid, W_wf * 1000, levels=12, colors='black',
            linewidths=0.2, alpha=0.4)
cb = fig6.colorbar(cf, ax=ax6, label=r'$w(x,t)$ [mm]', shrink=0.85)
cb.ax.tick_params(labelsize=8)

# 标注特殊位置
for xc, lab in [(0, 'A'), (ell, 'B'), (2*ell, 'C'), (3*ell, 'D'), (4*ell, 'E')]:
    ax6.axhline(y=xc, color='black', linewidth=0.5, linestyle=':')
    ax6.text(t_wf[-1]*1000 + 2, xc, lab, fontsize=8, va='center', color='black')

ax6.set_xlabel(r'$t$ [ms]', fontsize=11)
ax6.set_ylabel(r'$x$ [m]', fontsize=11)
ax6.set_title('FEM — Spacetime Response $w(x,t)$  (first 4 modes)',
              fontsize=12, fontweight='bold')
ax6.tick_params(labelsize=9)
fig6.tight_layout()
save_figure(fig6, 'fig_fem_6_spacetime')
plt.close(fig6)

# ============================================================
# 图 7: FEM 网格收敛性 (可选，展示方法精度)
# ============================================================
# 由于我们只有一个网格 (ne_per_seg=10), 这里用前 6 阶频率做个表格图
print('绘制图 7: 频率汇总表...')
fig7, ax7 = plt.subplots(1, 1, figsize=(6.5, 3.5))
ax7.axis('off')

# 构建表格数据
table_data = []
table_data.append([f'{i+1}' for i in range(6)])
table_data.append([f'{betas[i]:.5f}' for i in range(6)])
table_data.append([f'{omegas[i]:.2f}' for i in range(6)])
table_data.append([f'{freqs[i]:.2f}' for i in range(6)])
table_data.append([f'{1.0/freqs[i]:.4f}' for i in range(6)])
table_data.append([f'{M_conc[i]/M_n[i]*100:.1f}%' for i in range(6)])

row_labels = ['Mode $n$', '$\\beta_n$', '$\\omega_n$ [rad/s]',
              '$f_n$ [Hz]', '$T_n$ [s]', 'Conc. mass ratio']
col_labels = [f'{i+1}' for i in range(6)]

table = ax7.table(cellText=table_data, rowLabels=row_labels,
                  cellLoc='center', loc='center',
                  colWidths=[0.12]*6)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.6)
for key, cell in table.get_celld().items():
    cell.set_edgecolor('gray')
    cell.set_linewidth(0.3)

ax7.set_title('FEM Results Summary — First 6 Modes  (40 elements, 81 active DOFs)',
              fontsize=11, fontweight='bold', pad=30)
fig7.tight_layout()
save_figure(fig7, 'fig_fem_7_summary_table')
plt.close(fig7)

# ============================================================
# 图 8: FEM 节点振型 vs Hermite 插值验证
# ============================================================
print('绘制图 8: 节点振型验证...')
fig8, ax8 = plt.subplots(1, 1, figsize=(7, 4.5))

# 读取节点数据
nodes_df = pd.read_csv(f'{OUT_DIR}/mode_shapes_nodes.csv')
x_nodes_fem = nodes_df['x_m'].values

# 振型1 (在节点处的 w 值)
w_nodes = np.zeros((len(x_nodes_fem), n_modes_plot))
for n in range(n_modes_plot):
    w_nodes[:, n] = nodes_df[f'w_mode_{n+1}'].values

for n in range(min(n_modes_plot, 4)):
    ax8.plot(x_modes, phi_mat[:, n], color=colors[n], linewidth=1.2, alpha=0.8,
             label=f'Mode {n+1} (interp)')
    ax8.scatter(x_nodes_fem, w_nodes[:, n], color=colors[n], s=15, marker='o',
                edgecolors='white', linewidth=0.3, zorder=5)

for sx, sl in zip(special_x, special_labels):
    ax8.axvline(x=sx, color='gray', linewidth=0.5, linestyle='--', alpha=0.4)
    ax8.text(sx, ax8.get_ylim()[1]*0.95, sl, fontsize=8, ha='center', color='gray')

ax8.set_xlabel('$x$ [m]', fontsize=11)
ax8.set_ylabel('$\\phi_n(x)$', fontsize=11)
ax8.set_title('FEM Nodal Values & Hermite Interpolation', fontsize=12, fontweight='bold')
ax8.legend(fontsize=8, ncol=2)
ax8.grid(True, alpha=0.25)
ax8.axhline(y=0, color='black', linewidth=0.4)
ax8.tick_params(labelsize=9)
fig8.tight_layout()
save_figure(fig8, 'fig_fem_8_nodal_check')
plt.close(fig8)

# ============================================================
# 收尾输出
# ============================================================
print()
print("=" * 65)
print("  后处理完成 — 所有图片已保存 (PNG + PDF)")
print("=" * 65)
print("  图 1: fig_fem_1_freq_compare   — FEM vs TMM 频率对比")
print("  图 2: fig_fem_2_mode_shapes    — 前 6 阶振型")
print("  图 3: fig_fem_3_modal_mass     — 模态质量分解")
print("  图 4: fig_fem_4_response_C     — C 点时程响应")
print("  图 5: fig_fem_5_modal_contrib  — 模态贡献谱")
print("  图 6: fig_fem_6_spacetime      — 时空响应瀑布图")
print("  图 7: fig_fem_7_summary_table  — 结果汇总表")
print("  图 8: fig_fem_8_nodal_check    — 节点振型验证")
