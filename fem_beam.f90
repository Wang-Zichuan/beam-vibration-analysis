!===============================================================================
! 结构动力学 Q3 — Euler-Bernoulli 梁 FEM 求解器
! 传递矩阵法 → 有限元法 (Fortran 2008)
!
! 梁布局 (总长 4l):
!   A(自由) -- B(铰支) -- C(集中质量 m) -- D(弹簧 k) -- E(自由)
!   x=0        x=l         x=2l             x=3l           x=4l
!
! 方法: 两节点 Hermite 梁单元, 一致质量矩阵,
!       广义特征值问题 K·φ = ω²·M·φ
!       用 Cholesky 分解 + Jacobi 法求解
!===============================================================================

program fem_beam
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  integer, parameter :: n_seg = 4           ! 梁段数
  integer, parameter :: ne_per_seg = 10     ! 每段单元数
  integer, parameter :: ne = n_seg * ne_per_seg  ! 总单元数
  integer, parameter :: nn = ne + 1         ! 总节点数
  integer, parameter :: ndof_total = 2 * nn ! 总自由度数 (每节点 w, θ)
  integer, parameter :: max_iter = 500      ! Jacobi 最大迭代

  ! ---- 物理参数 ----
  real(dp), parameter :: ell    = 1.0d0           ! 特征长度 [m]
  real(dp), parameter :: E_mod  = 2.10d11         ! 弹性模量 [Pa]
  real(dp), parameter :: rho    = 7800.0d0        ! 密度 [kg/m³]
  real(dp), parameter :: S_area = 0.01d0          ! 截面积 [m²]
  real(dp), parameter :: J_inert= 8.33d-6         ! 截面惯性矩 [m⁴]
  real(dp), parameter :: EI     = E_mod * J_inert ! 弯曲刚度
  real(dp), parameter :: mu     = rho * S_area    ! 线密度
  real(dp), parameter :: m_mass = mu * ell * 0.8d0! 集中质量 [kg]
  real(dp), parameter :: k_spr  = EI / ell**3 * 30.0d0  ! 弹簧刚度 [N/m]
  real(dp), parameter :: v0     = 0.1d0           ! C 点初始速度 [m/s]

  ! ---- 几何 ----
  real(dp) :: total_len, le
  integer  :: node_B, node_C, node_D
  integer  :: dof_B_w, dof_C_w, dof_D_w

  ! ---- 矩阵 ----
  real(dp), allocatable :: K_global(:,:), M_global(:,:)
  real(dp), allocatable :: K_red(:,:), M_red(:,:)
  real(dp), allocatable :: L_mat(:,:), A_mat(:,:)
  real(dp), allocatable :: eig_vecs(:,:), eig_vals(:)
  real(dp), allocatable :: mode_shapes_full(:,:)  ! 包含被约束 DOF
  real(dp), allocatable :: modal_masses(:)
  integer, allocatable  :: active_dofs(:)

  ! ---- 局部变量 ----
  integer  :: i, j, k, n_active, n_modes, info
  real(dp) :: le_val, x_node(nn), x_out(1001)
  real(dp) :: phi_at_C, M_n, omega_n, coeff_n
  real(dp) :: t_val, dt, w_C, v_C, a_C
  integer  :: n_pts, n_t, i_t, i_x
  character(len=256) :: fname

  ! ---- 计算几何 ----
  total_len = n_seg * ell
  le = ell / real(ne_per_seg, dp)   ! 单元长度

  ! 节点编号 (1-based): A=1, B=ne_per_seg+1, C=2*ne_per_seg+1, D=3*ne_per_seg+1, E=nn
  node_B = ne_per_seg + 1
  node_C = 2 * ne_per_seg + 1
  node_D = 3 * ne_per_seg + 1

  ! DOF 编号: w_i = 2*i-1, θ_i = 2*i
  dof_B_w = 2 * node_B - 1
  dof_C_w = 2 * node_C - 1
  dof_D_w = 2 * node_D - 1

  ! 节点坐标
  do i = 1, nn
    x_node(i) = real(i - 1, dp) * le
  end do

  write(*, '("============================================")')
  write(*, '("  FEM Beam Solver — Fortran 2008")')
  write(*, '("============================================")')
  write(*, '(A,I0,A,I0,A,I0)') "  Nodes: ", nn, "  Elements: ", ne, "  DOFs: ", ndof_total
  write(*, '(A,F10.4,A)')    "  Total length: ", total_len, " m"
  write(*, '(A,F10.4,A)')    "  Element length: ", le, " m"
  write(*, '(A,F8.1,A)')     "  EI = ", EI, " N·m²"
  write(*, '(A,F8.3,A)')     "  μ  = ", mu, " kg/m"
  write(*, '(A,F8.3,A)')     "  m  = ", m_mass, " kg"
  write(*, '(A,F10.1,A)')    "  k  = ", k_spr, " N/m"
  write(*, '(A,I0,A,I0,A,I0)') "  B: node ", node_B, "  C: node ", node_C, "  D: node ", node_D

  ! ---- 分配全局矩阵 ----
  allocate(K_global(ndof_total, ndof_total), M_global(ndof_total, ndof_total))
  K_global = 0.0d0
  M_global = 0.0d0

  ! ---- 组装 ----
  call assemble_system(K_global, M_global)

  ! ---- 添加集中质量与弹簧 (在消去约束前加到完整系统) ----
  K_global(dof_D_w, dof_D_w) = K_global(dof_D_w, dof_D_w) + k_spr
  M_global(dof_C_w, dof_C_w) = M_global(dof_C_w, dof_C_w) + m_mass

  ! ---- 处理铰支约束: 消去 dof_B_w ----
  n_active = ndof_total - 1
  allocate(active_dofs(n_active))
  j = 0
  do i = 1, ndof_total
    if (i /= dof_B_w) then
      j = j + 1
      active_dofs(j) = i
    end if
  end do

  allocate(K_red(n_active, n_active), M_red(n_active, n_active))
  do j = 1, n_active
    do i = 1, n_active
      K_red(i, j) = K_global(active_dofs(i), active_dofs(j))
      M_red(i, j) = M_global(active_dofs(i), active_dofs(j))
    end do
  end do

  write(*, '(A,I0,A,I0)') "  Reduced system: ", n_active, " × ", n_active

  ! ---- 求解广义特征值问题 ----
  n_modes = min(n_active, 12)  ! 求前 12 阶
  allocate(eig_vals(n_active), eig_vecs(n_active, n_active))
  allocate(L_mat(n_active, n_active), A_mat(n_active, n_active))

  call solve_eigenproblem(K_red, M_red, n_active, eig_vals, eig_vecs, L_mat, A_mat, info)

  if (info /= 0) then
    write(*,*) "ERROR: 特征值求解失败, info = ", info
    stop 1
  end if

  ! 输出频率
  write(*, '(/,"  固有频率:")')
  write(*, '("  n",4X,"ω_n [rad/s]",4X,"f_n [Hz]",6X,"β_n")')
  do i = 1, n_modes
    if (eig_vals(i) > 0.0d0) then
      write(*, '(I3,2F14.4,F12.5)') i, sqrt(eig_vals(i)), &
        sqrt(eig_vals(i)) / (2.0d0 * 3.141592653589793d0), &
        (eig_vals(i) * mu / EI)**0.25d0 * ell
    end if
  end do

  ! ---- 重构完整振型 (包含被约束 DOF = 0) ----
  allocate(mode_shapes_full(ndof_total, n_modes))
  mode_shapes_full = 0.0d0
  do j = 1, n_modes
    do i = 1, n_active
      mode_shapes_full(active_dofs(i), j) = eig_vecs(i, j)
    end do
  end do

  ! ---- 计算模态质量 ----
  allocate(modal_masses(n_modes))
  call compute_modal_masses(M_global, mode_shapes_full, ndof_total, n_modes, &
    dof_C_w, m_mass, modal_masses)

  write(*, '(/,"  模态质量:")')
  do i = 1, n_modes
    if (eig_vals(i) > 0.0d0) then
      omega_n = sqrt(eig_vals(i))
      write(*, '(I3,F14.4)') i, modal_masses(i)
    end if
  end do

  ! ---- 导出 CSV ----
  call export_csv_files(x_node, nn, mode_shapes_full, ndof_total, n_modes, &
    eig_vals, modal_masses, dof_C_w, dof_B_w)

  write(*, '(/,"  All CSV files exported successfully.")')
  write(*, '("  Done.")')

  ! ---- 清理 ----
  deallocate(K_global, M_global, K_red, M_red, L_mat, A_mat)
  deallocate(eig_vals, eig_vecs, mode_shapes_full, modal_masses, active_dofs)

contains

  !=============================================================================
  ! 组装全局刚度矩阵和质量矩阵
  !=============================================================================
  subroutine assemble_system(Kg, Mg)
    real(dp), intent(inout) :: Kg(:,:), Mg(:,:)
    real(dp) :: Ke(4,4), Me(4,4)
    integer  :: e, n1, n2, dofs(4), i_loc, j_loc, i_g, j_g
    real(dp) :: le3, le_val

    le_val = le
    le3 = le_val**3

    ! 单元刚度矩阵 (Hermite 梁)
    Ke(1,:) = [ 12.0d0,   6.0d0*le_val, -12.0d0,   6.0d0*le_val]
    Ke(2,:) = [  6.0d0*le_val,  4.0d0*le_val**2,  -6.0d0*le_val,  2.0d0*le_val**2]
    Ke(3,:) = [-12.0d0,  -6.0d0*le_val,  12.0d0,  -6.0d0*le_val]
    Ke(4,:) = [  6.0d0*le_val,  2.0d0*le_val**2,  -6.0d0*le_val,  4.0d0*le_val**2]
    Ke = Ke * (EI / le3)

    ! 单元一致质量矩阵
    Me(1,:) = [156.0d0,   22.0d0*le_val,   54.0d0,  -13.0d0*le_val]
    Me(2,:) = [ 22.0d0*le_val,   4.0d0*le_val**2,  13.0d0*le_val,  -3.0d0*le_val**2]
    Me(3,:) = [ 54.0d0,   13.0d0*le_val,  156.0d0,  -22.0d0*le_val]
    Me(4,:) = [-13.0d0*le_val,  -3.0d0*le_val**2, -22.0d0*le_val,   4.0d0*le_val**2]
    Me = Me * (mu * le_val / 420.0d0)

    do e = 1, ne
      n1 = e
      n2 = e + 1
      dofs = [2*n1-1, 2*n1, 2*n2-1, 2*n2]

      do j_loc = 1, 4
        j_g = dofs(j_loc)
        do i_loc = 1, 4
          i_g = dofs(i_loc)
          Kg(i_g, j_g) = Kg(i_g, j_g) + Ke(i_loc, j_loc)
          Mg(i_g, j_g) = Mg(i_g, j_g) + Me(i_loc, j_loc)
        end do
      end do
    end do
  end subroutine assemble_system

  !=============================================================================
  ! 广义特征值问题 K·x = λ·M·x
  ! 方法: M = L·Lᵀ (Cholesky) → A = L⁻¹·K·L⁻ᵀ → Jacobi 对角化 A
  !=============================================================================
  subroutine solve_eigenproblem(K, M, n, lam, vec, L, A, info)
    integer,  intent(in)    :: n
    real(dp), intent(in)    :: K(n,n), M(n,n)
    real(dp), intent(out)   :: lam(n), vec(n,n), L(n,n), A(n,n)
    integer,  intent(out)   :: info

    real(dp), allocatable :: L_inv(:,:), tmp(:,:)
    integer :: i, j, k, rot_count
    real(dp) :: sum_val, theta, t, c, s, tau, alpha, beta, gamma

    info = 0

    ! ---- Step 1: Cholesky M = L·Lᵀ ----
    L = 0.0d0
    do j = 1, n
      sum_val = M(j, j)
      do k = 1, j-1
        sum_val = sum_val - L(j, k)**2
      end do
      if (sum_val <= 0.0d0) then
        info = -1
        return
      end if
      L(j, j) = sqrt(sum_val)
      do i = j+1, n
        sum_val = M(i, j)
        do k = 1, j-1
          sum_val = sum_val - L(i, k) * L(j, k)
        end do
        L(i, j) = sum_val / L(j, j)
      end do
    end do

    ! ---- Step 2: 求 L⁻¹ ----
    allocate(L_inv(n, n))
    L_inv = 0.0d0
    do i = 1, n
      L_inv(i, i) = 1.0d0 / L(i, i)
      do j = 1, i-1
        sum_val = 0.0d0
        do k = j, i-1
          sum_val = sum_val - L(i, k) * L_inv(k, j)
        end do
        L_inv(i, j) = sum_val / L(i, i)
      end do
    end do

    ! ---- Step 3: A = L⁻¹·K·(L⁻¹)ᵀ ----
    allocate(tmp(n, n))
    ! tmp = K · (L⁻¹)ᵀ
    tmp = 0.0d0
    do j = 1, n
      do i = 1, n
        sum_val = 0.0d0
        do k = 1, n
          sum_val = sum_val + K(i, k) * L_inv(j, k)
        end do
        tmp(i, j) = sum_val
      end do
    end do
    ! A = L⁻¹ · tmp
    A = 0.0d0
    do j = 1, n
      do i = 1, n
        sum_val = 0.0d0
        do k = 1, n
          sum_val = sum_val + L_inv(i, k) * tmp(k, j)
        end do
        A(i, j) = sum_val
      end do
    end do
    deallocate(tmp)

    ! ---- Step 4: Jacobi 方法对角化对称矩阵 A ----
    vec = 0.0d0
    do i = 1, n
      vec(i, i) = 1.0d0
    end do

    do rot_count = 1, max_iter
      ! 找最大非对角元素
      alpha = 0.0d0
      do j = 2, n
        do i = 1, j-1
          if (abs(A(i, j)) > alpha) then
            alpha = abs(A(i, j))
            k = i
            i = j  ! 这两行只是用来记录p,q位置
          end if
        end do
      end do

      ! 重新正确记录p,q
      alpha = 0.0d0
      do j = 2, n
        do i_loc = 1, j-1
          if (abs(A(i_loc, j)) > alpha) then
            alpha = abs(A(i_loc, j))
            k = i_loc
            l = j
          end if
        end do
      end do

      if (alpha < 1.0d-12) exit

      ! 计算旋转角度
      theta = 0.5d0 * atan2(2.0d0 * A(k, l), A(k, k) - A(l, l))
      c = cos(theta)
      s = sin(theta)
      t = s / (1.0d0 + c)

      ! 更新 A: A' = Rᵀ · A · R
      tau = A(k, l)
      A(k, k) = A(k, k) - tau * t
      A(l, l) = A(l, l) + tau * t
      A(k, l) = 0.0d0
      A(l, k) = 0.0d0

      do i_loc = 1, n
        if (i_loc /= k .and. i_loc /= l) then
          alpha = A(i_loc, k)
          beta  = A(i_loc, l)
          gamma = A(l, i_loc)
          A(i_loc, k) = alpha * c - beta * s
          A(k, i_loc) = A(i_loc, k)
          A(i_loc, l) = alpha * s + beta * c
          A(l, i_loc) = A(i_loc, l)
        end if
      end do

      ! 更新特征向量: V' = V · R
      do i_loc = 1, n
        alpha = vec(i_loc, k)
        beta  = vec(i_loc, l)
        vec(i_loc, k) = alpha * c - beta * s
        vec(i_loc, l) = alpha * s + beta * c
      end do
    end do

    write(*, '(A,I0,A)') "  Jacobi converged in ", rot_count, " iterations"

    ! 提取特征值
    do i = 1, n
      lam(i) = A(i, i)
    end do

    ! ---- Step 5: 回代特征向量 x = L⁻ᵀ · y ----
    allocate(tmp(n, n_modes))
    do j_local = 1, n_modes
      do i_loc = 1, n
        sum_val = 0.0d0
        do k = n, 1, -1
          sum_val = sum_val + L_inv(k, i_loc) * vec(k, j_local)
        end do
        tmp(i_loc, j_local) = sum_val
      end do
    end do

    ! ---- Step 6: 排序 (升序) ----
    call sort_eigenpairs(lam, tmp, n, n_modes)

    ! 复制特征向量到输出
    vec = 0.0d0
    vec(:, 1:n_modes) = tmp(:, 1:n_modes)

    deallocate(L_inv, tmp)
  end subroutine solve_eigenproblem

  !=============================================================================
  ! 排序特征对 (升序)
  !=============================================================================
  subroutine sort_eigenpairs(lam, vec, n, nv)
    integer,  intent(in)    :: n, nv
    real(dp), intent(inout) :: lam(n), vec(n, nv)
    integer  :: i, j, i_min
    real(dp) :: tmp_lam, tmp_vec(n)

    do i = 1, nv - 1
      i_min = i
      do j = i + 1, n
        if (lam(j) < lam(i_min)) i_min = j
      end do
      if (i_min /= i) then
        tmp_lam = lam(i)
        lam(i) = lam(i_min)
        lam(i_min) = tmp_lam
        tmp_vec = vec(:, i)
        vec(:, i) = vec(:, i_min)
        vec(:, i_min) = tmp_vec
      end if
    end do
  end subroutine sort_eigenpairs

  !=============================================================================
  ! 计算模态质量 M_n = ∫ μ φ² dx + m·φ²(2l)
  !=============================================================================
  subroutine compute_modal_masses(Mg, phi, ntot, nm, dof_C, m_add, mm_out)
    integer,  intent(in)  :: ntot, nm, dof_C
    real(dp), intent(in)  :: Mg(ntot, ntot), phi(ntot, nm), m_add
    real(dp), intent(out) :: mm_out(nm)
    integer  :: i, j, k_mode
    real(dp) :: sum_val, phi_C

    do k_mode = 1, nm
      ! M_n = φᵀ · M_global · φ  (包含了集中质量在M_global中)
      sum_val = 0.0d0
      do j = 1, ntot
        do i = 1, ntot
          sum_val = sum_val + phi(i, k_mode) * Mg(i, j) * phi(j, k_mode)
        end do
      end do
      ! 注意: M_global 中 dof_C_w 已有 +m_mass, 所以上式已含集中质量贡献
      ! 但为与解析解兼容，这里直接用 M 全局矩阵计算，无需额外加 m
      mm_out(k_mode) = sum_val
    end do
  end subroutine compute_modal_masses

  !=============================================================================
  ! 导出所有 CSV 文件
  !=============================================================================
  subroutine export_csv_files(xn, n_nodes, phi, ntot, nm, lam, mm, dof_C_w, dof_B_w)
    integer,  intent(in) :: n_nodes, ntot, nm, dof_C_w, dof_B_w
    real(dp), intent(in) :: xn(n_nodes), phi(ntot, nm), lam(ntot), mm(nm)

    integer, parameter   :: n_pts_interp = 401  ! 插值点数
    integer, parameter   :: n_t_steps = 500
    real(dp) :: xi_out(n_pts_interp), phi_interp(n_pts_interp)
    real(dp) :: t_arr(n_t_steps), w_C_arr(n_t_steps), v_C_arr(n_t_steps), a_C_arr(n_t_steps)
    real(dp) :: phi_C, omega_n, coeff, M_n, dt, t_val
    integer  :: i, j, i_mode, i_t, i_seg, ios, n_modes_use
    character(len=256) :: fname

    n_modes_use = min(nm, 8)

    ! ---- (a) 频率文件 ----
    open(unit=10, file='fem_frequencies.csv', status='replace', action='write', iostat=ios)
    write(10, '(A)') 'mode,beta,omega_rad_s,freq_Hz,period_s,modal_mass_kg'
    do i = 1, n_modes_use
      if (lam(i) > 0.0d0) then
        omega_n = sqrt(lam(i))
        write(10, '(I0,",",F12.6,",",F14.6,",",F12.6,",",F12.6,",",F14.6)') &
          i, (lam(i) * mu / EI)**0.25d0 * ell, omega_n, &
          omega_n / (2.0d0 * 3.141592653589793d0), &
          2.0d0 * 3.141592653589793d0 / omega_n, mm(i)
      end if
    end do
    close(10)
    write(*,*) '  -> fem_frequencies.csv'

    ! ---- (b) 振型文件 ----
    ! 在采样点上插值振型
    do i_pts = 1, n_pts_interp
      xi_out(i_pts) = real(i_pts - 1, dp) / real(n_pts_interp - 1, dp) * total_len
    end do

    open(unit=11, file='fem_mode_shapes.csv', status='replace', action='write')
    write(11, '(A)', advance='no') 'x_m'
    do i = 1, n_modes_use
      write(11, '(A,I0)', advance='no') ',mode_', i
    end do
    write(11, *)

    do i_pts = 1, n_pts_interp
      write(11, '(F10.4)', advance='no') xi_out(i_pts)
      do i_mode = 1, n_modes_use
        call interpolate_mode(xi_out(i_pts), phi(:, i_mode), xn, n_nodes, le, phi_interp(i_pts))
        write(11, '(A,F14.8)', advance='no') ',', phi_interp(i_pts)
      end do
      write(11, *)
    end do
    close(11)
    write(*,*) '  -> fem_mode_shapes.csv'

    ! ---- (c) C 点响应 ----
    dt = 0.0003d0
    do i_t = 1, n_t_steps
      t_arr(i_t) = real(i_t - 1, dp) * dt
      w_C_arr(i_t) = 0.0d0
      v_C_arr(i_t) = 0.0d0
      a_C_arr(i_t) = 0.0d0
      do i_mode = 1, n_modes_use
        if (lam(i_mode) <= 0.0d0) cycle
        omega_n = sqrt(lam(i_mode))
        M_n = mm(i_mode)
        phi_C = phi(dof_C_w, i_mode)
        coeff = m_mass * v0 * phi_C / (M_n * omega_n)
        w_C_arr(i_t) = w_C_arr(i_t) + phi_C * coeff * sin(omega_n * t_arr(i_t))
        v_C_arr(i_t) = v_C_arr(i_t) + phi_C * coeff * omega_n * cos(omega_n * t_arr(i_t))
        a_C_arr(i_t) = a_C_arr(i_t) - phi_C * coeff * omega_n**2 * sin(omega_n * t_arr(i_t))
      end do
    end do

    open(unit=12, file='fem_response_C.csv', status='replace', action='write')
    write(12, '(A)') 'time_s,displacement_m,velocity_m_s,acceleration_m_s2'
    do i_t = 1, n_t_steps
      write(12, '(F12.6,",",E16.8,",",E16.8,",",E16.8)') &
        t_arr(i_t), w_C_arr(i_t), v_C_arr(i_t), a_C_arr(i_t)
    end do
    close(12)
    write(*,*) '  -> fem_response_C.csv'

    ! ---- (d) 参数文件 ----
    open(unit=13, file='fem_params.csv', status='replace', action='write')
    write(13, '(A)') 'parameter,value,unit'
    write(13, '(A,F12.4,A)')   'l', ell, ',m'
    write(13, '(A,E14.6,A)')   'E', E_mod, ',Pa'
    write(13, '(A,F10.4,A)')   'rho', rho, ',kg/m3'
    write(13, '(A,F8.4,A)')    'S', S_area, ',m2'
    write(13, '(A,E12.4,A)')   'J', J_inert, ',m4'
    write(13, '(A,E12.4,A)')   'EI', EI, ',N·m2'
    write(13, '(A,F10.4,A)')   'mu', mu, ',kg/m'
    write(13, '(A,F10.4,A)')   'm', m_mass, ',kg'
    write(13, '(A,E14.6,A)')   'k', k_spr, ',N/m'
    write(13, '(A,F8.4,A)')    'v0', v0, ',m/s'
    write(13, '(A,I0,A)')      'n_elements', ne, ','
    write(13, '(A,I0,A)')      'n_nodes', nn, ','
    write(13, '(A,I0,A)')      'n_active_dofs', n_active, ','
    write(13, '(A,I0,A)')      'n_seg_per_element', ne_per_seg, ','
    close(13)
    write(*,*) '  -> fem_params.csv'

    ! ---- (e) 模态质量分解 ----
    open(unit=14, file='fem_modal_masses.csv', status='replace', action='write')
    write(14, '(A)') 'mode,modal_mass_kg,distributed_contrib_kg,concentrated_contrib_kg'
    do i = 1, n_modes_use
      if (lam(i) <= 0.0d0) cycle
      phi_C = phi(dof_C_w, i)
      omega_n = sqrt(lam(i))
      write(14, '(I0,",",F14.6,",",F14.6,",",F14.6)') &
        i, mm(i), mm(i) - m_mass * phi_C**2, m_mass * phi_C**2
    end do
    close(14)
    write(*,*) '  -> fem_modal_masses.csv'

  end subroutine export_csv_files

  !=============================================================================
  ! 从 FEM 节点值插值得到任意 x 处的振型值
  !=============================================================================
  subroutine interpolate_mode(x_val, phi_nodal, xn, n_nodes, lelem, phi_out)
    real(dp), intent(in)  :: x_val, phi_nodal(:), xn(n_nodes), lelem
    real(dp), intent(out) :: phi_out
    integer,  intent(in)  :: n_nodes
    integer  :: elem_idx, n1, n2
    real(dp) :: xi, N1_w, N2_w, N1_t, N2_t
    real(dp) :: w1, t1, w2, t2

    ! 找所在单元
    if (x_val <= xn(1)) then
      elem_idx = 1
    else if (x_val >= xn(n_nodes)) then
      elem_idx = n_nodes - 1
    else
      do elem_idx = 1, n_nodes - 1
        if (x_val <= xn(elem_idx + 1)) exit
      end do
    end if

    n1 = elem_idx
    n2 = elem_idx + 1
    ! 本地坐标 (0 到 1)
    xi = (x_val - xn(n1)) / lelem
    if (xi < 0.0d0) xi = 0.0d0
    if (xi > 1.0d0) xi = 1.0d0

    ! Hermite 形函数
    N1_w = 1.0d0 - 3.0d0*xi**2 + 2.0d0*xi**3
    N1_t = lelem * (xi - 2.0d0*xi**2 + xi**3)
    N2_w = 3.0d0*xi**2 - 2.0d0*xi**3
    N2_t = lelem * (-xi**2 + xi**3)

    w1 = phi_nodal(2*n1 - 1)
    t1 = phi_nodal(2*n1)
    w2 = phi_nodal(2*n2 - 1)
    t2 = phi_nodal(2*n2)

    phi_out = N1_w * w1 + N1_t * t1 + N2_w * w2 + N2_t * t2
  end subroutine interpolate_mode

end program fem_beam
