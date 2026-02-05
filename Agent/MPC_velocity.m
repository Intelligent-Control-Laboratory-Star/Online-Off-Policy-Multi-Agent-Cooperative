function [sys,x0,str,ts]=MPC_velocity(t,x,u,flag)
switch flag 
    case 0
        [sys,x0,str,ts]=mdlInitializeSizes;
    case 3
        sys=mdlOutputs(t,x,u);
    case {1, 2, 4, 9}
        sys = [];
    otherwise
        error(['Unhandled flag = ',num2str(flag)]);
end

function [sys,x0,str,ts]=mdlInitializeSizes
    % 定义系统维度
    sizes = simsizes;
    sizes.NumContStates  = 0;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 5;  % 输出控制信号维度 [u1, u2, u3, u4, V]
    sizes.NumInputs      = 7;  % 输入状态和参数维度
    sizes.DirFeedthrough = 1;  % 输入直接影响输出
    sizes.NumSampleTimes = 0;  % 连续时间
    sys=simsizes(sizes);
    x0=[];
    str=[];
    ts  = [];

function sys=mdlOutputs(t, x, newExp)
    persistent param
    if t==0
        % MPC参数初始化
        param.N         = 10;         % 预测步长
        param.dt        = 0.01;       % 时间步长
        param.Q         = diag([100, 50, 50, 10]);    % 状态权重矩阵
        param.R         = diag([0.01, 0.01, 0.01, 0.01]); % 控制权重矩阵
        param.Qf        = 3*param.Q;  % 终端状态权重
        param.x_ref     = [80; 0; 0; 0]; % 参考状态 [vx_ref, gamma_ref, beta_ref, phi_ref]
        param.u_max     = [800; 800; 800; 800];   % 控制输入上限
        param.u_min     = [0; 0; 0; 0]; % 控制输入下限
    end
    
    % 获取当前状态和输入
    exp_time  = newExp(1);
    exp_state = newExp(2:5)';    % 当前状态 [vx, gamma, beta, phi]
    vx = exp_state(1);
    phi = exp_state(4);
    delta = newExp(6:7);         % 前轮转角 [delta1, delta2]
    
    % 获取系统动力学矩阵
    [g_matrix, k_matrix] = nonlinear_dynamics(delta, vx, phi);
    
    % MPC优化求解
    [u_optimal, V_value] = solve_mpc(exp_state, g_matrix, param);
    
    % 修改3: 增加控制输出放大系数
    control_gain = 2.5e0;  % 可调整此系数来放大控制输出
    
    % 输出控制信号
    sys(1) = real(u_optimal(1)) * control_gain;  % u1
    sys(2) = real(u_optimal(2)) * control_gain;  % u2  
    sys(3) = real(u_optimal(3)) * control_gain;  % u3
    sys(4) = real(u_optimal(4)) * control_gain;  % u4
    sys(5) = real(V_value);       % 值函数

function [u_optimal, V_value] = solve_mpc(x_current, g_matrix, param)
    % MPC优化问题求解
    N = param.N;
    dt = param.dt;
    Q = param.Q;
    R = param.R;
    Qf = param.Qf;
    x_ref = param.x_ref;
    u_max = param.u_max;
    u_min = param.u_min;
    
    % 状态和控制维度
    nx = 4;  % 状态维度
    nu = 4;  % 控制维度
    
    % 构建优化变量: [u0, u1, ..., u_{N-1}, x1, x2, ..., x_N]
    n_vars = N*nu + N*nx;
    
    % 构建二次规划问题
    H = zeros(n_vars, n_vars);
    f = zeros(n_vars, 1);
    
    % 目标函数权重矩阵
    for k = 1:N
        % 控制输入权重
        u_idx = (k-1)*nu + 1 : k*nu;
        H(u_idx, u_idx) = R;
        
        % 状态权重
        x_idx = N*nu + (k-1)*nx + 1 : N*nu + k*nx;
        if k == N
            H(x_idx, x_idx) = Qf;  % 终端权重
        else
            H(x_idx, x_idx) = Q;
        end
        
        % 状态跟踪误差线性项
        f(x_idx) = -2 * (Q * x_ref);
        if k == N
            f(x_idx) = -2 * (Qf * x_ref);
        end
    end
    
    % 等式约束: 动力学约束
    Aeq = zeros(N*nx + nx, n_vars);
    beq = zeros(N*nx + nx, 1);
    
    % 初始状态约束
    Aeq(1:nx, N*nu + 1:N*nu + nx) = eye(nx);
    beq(1:nx) = x_current;
    
    % 动力学约束: x_{k+1} = x_k + dt * (f(x_k) + g(x_k) * u_k)
    for k = 1:N-1
        row_idx = nx + (k-1)*nx + 1 : nx + k*nx;
        
        % x_{k+1}
        x_next_idx = N*nu + k*nx + 1 : N*nu + (k+1)*nx;
        Aeq(row_idx, x_next_idx) = eye(nx);
        
        % -x_k
        x_curr_idx = N*nu + (k-1)*nx + 1 : N*nu + k*nx;
        Aeq(row_idx, x_curr_idx) = -eye(nx);
        
        % -dt * g(x_k) * u_k (线性化近似)
        u_curr_idx = (k-1)*nu + 1 : k*nu;
        Aeq(row_idx, u_curr_idx) = -dt * g_matrix;
        
        % 右侧: dt * f(x_k) (非线性项，这里简化为0)
        beq(row_idx) = zeros(nx, 1);
    end
    
    % 不等式约束: 控制输入限制
    A_ineq = [];
    b_ineq = [];
    
    for k = 1:N
        u_idx = (k-1)*nu + 1 : k*nu;
        
        % u_k <= u_max
        A_temp = zeros(nu, n_vars);
        A_temp(:, u_idx) = eye(nu);
        A_ineq = [A_ineq; A_temp];
        b_ineq = [b_ineq; u_max];
        
        % u_k >= u_min  =>  -u_k <= -u_min
        A_temp = zeros(nu, n_vars);
        A_temp(:, u_idx) = -eye(nu);
        A_ineq = [A_ineq; A_temp];
        b_ineq = [b_ineq; -u_min];
    end
    
    % 求解二次规划问题
    options = optimoptions('quadprog', 'Display', 'off', 'Algorithm', 'interior-point-convex');
    
    try
        [z_opt, fval] = quadprog(H, f, A_ineq, b_ineq, Aeq, beq, [], [], [], options);
        
        if ~isempty(z_opt)
            % 提取最优控制输入 (第一个时间步)
            u_optimal = z_opt(1:nu);
            V_value = fval;
        else
            % 优化失败，使用零控制
            u_optimal = zeros(nu, 1);
            V_value = 0;
        end
    catch
        % 求解器出错，使用零控制
        u_optimal = zeros(nu, 1);
        V_value = 0;
    end

function [g, k] = nonlinear_dynamics(delta, vx, phi)
    delta1 = delta(1);
    delta2 = delta(2);
    mass = 1750; 
    lf = 1.556; 
    lr = 1.542;
    a = 1.25; 
    b = 1.40;
    Iz = 2189.7;
    Ix = 628;
    ho = 0.351;
    
    % 控制输入矩阵 g(x) - 4×4矩阵
    g = zeros(4, 4);
    g(1,1) = cos(delta1)/mass; 
    g(1,2) = cos(delta2)/mass; 
    g(1,3) = 1/mass; 
    g(1,4) = 1/mass;
    g(2,1) = (a*sin(delta1)-lf/2*cos(delta1))/Iz; 
    g(2,2) = (a*sin(delta2)+lf/2*cos(delta2))/Iz; 
    g(2,3) = -lf/2/Iz;
    g(2,4) = lf/2/Iz;
    g(3,1) = sin(delta1)/mass/(vx+0.12); 
    g(3,2) = sin(delta2)/mass/(vx+0.12); 
    g(3,3) = 0;
    g(3,4) = 0;
    g(4,1) = sin(delta1)*ho*cos(phi)/Ix; 
    g(4,2) = sin(delta2)*ho*cos(phi)/Ix; 
    g(4,3) = 0;
    g(4,4) = 0;
    
    % 干扰输入矩阵 k(x)
    k = eye(4);