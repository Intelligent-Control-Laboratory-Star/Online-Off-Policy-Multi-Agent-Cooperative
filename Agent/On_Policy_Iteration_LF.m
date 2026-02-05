function [sys,x0,str,ts]=On_Policy_Iteration_LF(t,x,u,flag)
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
end

function [sys,x0,str,ts]=mdlInitializeSizes
    global param NN Iter
    
    % ==== Barrier约束 ====
    param.c = [-30; -10; -10];
    param.d = [ 30;  10;  10];
    
    % ==== RL参数 ====
    param.n_state = 3;  
    param.n_action = 1; 
    param.n_basis = 30;
    param.gamma = 1;
    param.Q = 1.2e-1 * diag([1e1 1e3 1e1]);
    param.R = 1e0;
    ka1up   =  8e2; ka1down = 0;
    param.ka1 = (ka1up-(ka1down))/2; 
    param.ka2 = (ka1up+(ka1down))/2; 
    
    % ==== NN初始化 ====
    NN.disLength = linspace(-1e2, 1e2, param.n_basis/3);
    NN.visLength = linspace(-1e1, 1e1, param.n_basis/3);
    NN.cisLength = linspace(-1e1, 1e1, param.n_basis/3);
    NN.Theta_c = @(s) [tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.visLength * s(3))]';
    NN.w_c = rand(param.n_basis, 1); % 初始化权重
    NN.eta = 3e-2; % 学习率
    
    % ==== 状态变量 ====
    Iter.s_ctrl = zeros(param.n_state, 1); % 当前屏障变量
    Iter.s_hat = Iter.s_ctrl;              % 事件触发时的屏障变量
    
    % 定义系统维度
    sizes = simsizes;
    sizes.NumContStates  = 0;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 2;
    sizes.NumInputs      = 4;
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 0;
    sys=simsizes(sizes);
    x0=[]; str=[]; ts  = [];
end

function sys=mdlOutputs(t, x, newExp)
    global param NN Iter
    
    if t == 0
        % 初始化状态
        Iter.s_ctrl = zeros(param.n_state, 1);
        Iter.s_hat = Iter.s_ctrl;
    end
    
    % 控制律
    gradV = numerical_gradient(@(ss) NN.Theta_c(ss)' * NN.w_c, Iter.s_ctrl);
    B = [1; 1; 1];
    u = -param.ka1 * tanh(0.5 / param.ka1 * B' * gradV) + param.ka2;
    w = 0; % 暂不考虑外部扰动
    
    % 状态更新
    s_next = barrier(newExp(1:3), param.c, param.d);
    
    % 权重更新
    delta_theta = NN.Theta_c(Iter.s_ctrl) - NN.Theta_c(Iter.s_hat);
    GU = NN.Theta_c(Iter.s_ctrl) * u;
    HW = NN.Theta_c(Iter.s_ctrl) * w;
    cost_int = Iter.s_ctrl' * param.Q * Iter.s_ctrl - param.gamma^2 * (w' * w);
    delta_hat = NN.w_c' * delta_theta + GU + HW + cost_int;
    grad = NN.Theta_c(Iter.s_ctrl) .* delta_hat;
    NN.w_c = NN.w_c - NN.eta * grad;
    
    % 状态更新
    Iter.s_ctrl = s_next;
    Iter.s_hat = Iter.s_ctrl; % 更新触发状态
    
    % 输出控制信号
    sys(1) = real(u);
    sys(2) = real(norm(cost_int));
end

function s = barrier(x, c, d)
    s = log((d ./ c) .* ((c - x) ./ (d - x)));
end

function grad = numerical_gradient(f, s)
    eps = 1e-5;
    grad = zeros(length(s), 1);
    for i = 1:length(s)
        s1 = s; s2 = s;
        s1(i) = s1(i) + eps;
        s2(i) = s2(i) - eps;
        grad(i) = (f(s1) - f(s2)) / (2 * eps);
    end
end
