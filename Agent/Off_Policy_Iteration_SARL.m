function [sys,x0,str,ts]=Off_Policy_Iteration_SARL(t,x,u,flag)
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
    sizes.NumOutputs     = 4;
    sizes.NumInputs      = 3;
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 0;
    sys=simsizes(sizes);
    x0=[]; str=[]; ts  = [];

function sys=mdlOutputs(t, x1, newExp)
    persistent buffer param hat NN det iter
    if isempty(iter)
        iter = 1;
        % ==== Barrier约束 ====
        param.c = [0.1; -10; 0.1];
        param.d = [ 10;  10;  10];
        
        % ==== RL参数 ====
        param.n_state = 3;  
        param.n_action = 4; 
        param.n_basis = 30;
        param.gamma = 1;
        param.Q = 1e-3 * diag([1e1 1e2 1e1]);
        param.R = 1 * diag([1e0 1e0 1e0 1e0]);
        param.x_latet = param.n_state;
        param.s = zeros(param.n_state,1);
        
        % ==== 事件触发机制参数 ====
        det.eta1 = 0.95; det.Lu = 1e1; det.b_g = 1e1; det.delta1 = 0.7; det.alpha = 0.1; det.alpha =0.6;
        
        % ==== NN初始化 ====
        NN.disLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.visLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.cisLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.Theta_c = @(s) [tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.cisLength * s(3));...
                           tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.cisLength * s(3));...
                           tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.cisLength * s(3));...
                           tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.cisLength * s(3))]';
        NN.w_c = [0.5377;1.8339;-2.2588;0.8622;0.3188;-1.3077;-0.4336;0.3426;3.5784;2.7694;-1.3499;3.0349;0.7254;
        -0.0631;0.7147;-0.2050;-0.1241;1.4897;1.4090;1.4172;0.6715;-1.2075;0.7172;1.6302;0.4889;1.0347;
         0.7269;-0.3034;0.2939;-0.7873];
        NN.eta = 1e-3; % 2e-2
        NN.buffer_size = 100;
        NN.window_size = 10; % 轨迹长度（积分窗口）
        NN.update_interval = 5; % 每多少步更新一次
        
        % ==== 估计策略初始化（可选）====
        hat.u_hat = 0; 
        hat.w_hat = 0;
        
        % ==== 经验回放缓存 ====
        buffer.buffer = {}; % 每行为{S_traj, U_traj, U_hat_traj, W_traj, W_hat_traj, t_traj}

        % ==== 轨迹暂存器 ====
        buffer.S_traj = zeros(param.n_state, NN.window_size);
        buffer.U_traj = zeros(param.n_action, NN.window_size);
        buffer.U_hat_traj = zeros(param.n_action, NN.window_size);
        buffer.W_traj = zeros(param.n_action, NN.window_size);
        buffer.W_hat_traj = zeros(param.n_action, NN.window_size);
        buffer.t_traj = zeros(1, NN.window_size);

        % ==== 事件触发变量 ====
        param.s_ctrl = param.s;         % 当前屏障变量
        param.s_hat = param.s_ctrl;     % 事件触发时的屏障变量
        param.ej = zeros(param.n_state,1);    % 事件误差

        ka1up   =  8e2; 
        ka1down =  0;
        param.ka1 = (ka1up-(ka1down))/2; 
        param.ka2 = (ka1up+(ka1down))/2; % 控制上下界
    end
        
        param.s_hat = param.s_ctrl;
        
        % 2. NN近似值函数 (用事件触发的 s_hat)
        V = NN.Theta_c(param.s_hat)' * NN.w_c;
        gradV = numerical_gradient(@(ss) NN.Theta_c(ss)'*NN.w_c, param.s_hat);
        
        % 3. 控制律
        B = [1 1 1 1;1 1 1 1;1 1 1 1]; % 控制矩阵
        u = -param.ka1 * tanh(0.5/param.ka1 * B'*gradV) + param.ka2; 
        w = 0; % 暂不考虑外部扰动
        
        % 估计策略（这里假设与实际策略一致，可替换为你自己的估计）
        hat.u_hat = u;
        hat.w_hat = w;
        s_next = barrier(newExp(1:3), param.c, param.d);
        
        % 5. 记录轨迹
        idx = mod(iter-1, NN.window_size) + 1;
        buffer.S_traj(:,idx) = param.s_ctrl;
        buffer.U_traj(:,idx) = u;
        buffer.U_hat_traj(:,idx) = hat.u_hat;
        buffer.W_traj(:,idx) = w;
        buffer.W_hat_traj(:,idx) = hat.w_hat;
        buffer.t_traj(idx) = t;

        % 6. 到达窗口末尾时存入buffer
        if idx == NN.window_size
            buffer.buffer = [buffer.buffer; {buffer.S_traj, buffer.U_traj, buffer.U_hat_traj, buffer.W_traj, buffer.W_hat_traj, buffer.t_traj}];
            if size(buffer.buffer,1) > NN.buffer_size
                buffer.buffer(1,:) = [];
            end
        end

        % 7. 定期批量权重更新（严格积分法）
        if mod(iter, NN.update_interval)==0 && size(buffer.buffer,1) > 10
            batch_size = min(5, size(buffer.buffer,1));
            idxs = randperm(size(buffer.buffer,1), batch_size);
            gradsum = zeros(size(NN.w_c));
            for j = 1:batch_size
                % 取出一段轨迹
                S = buffer.buffer{idxs(j), 1};
                U_ = buffer.buffer{idxs(j), 2};
                U_hat_ = buffer.buffer{idxs(j), 3};
                W_ = buffer.buffer{idxs(j), 4};
                W_hat_ = buffer.buffer{idxs(j), 5};
                t_win = buffer.buffer{idxs(j), 6};
    
                % 1. delta_theta
                delta_theta = NN.Theta_c(S(:,end)) - NN.Theta_c(S(:,1));
    
                % 2. 各项积分
                GU = zeros(param.n_basis, NN.window_size);
                GU_hat = zeros(param.n_basis, NN.window_size);
                HW = zeros(param.n_basis, NN.window_size);
                HW_hat = zeros(param.n_basis, NN.window_size);
                cost_int = zeros(1, NN.window_size);
    
                for k = 1:NN.window_size
                    GU(:,k)     = NN.Theta_c(S(:,k)) * U_(:,k);
                    GU_hat(:,k) = NN.Theta_c(S(:,k)) * U_hat_(:,k);
                    HW(:,k)     = NN.Theta_c(S(:,k)) * W_(:,k);
                    HW_hat(:,k) = NN.Theta_c(S(:,k)) * W_hat_(:,k);
                    cost_int(k) = S(:,k)'*param.Q*S(:,k) - param.gamma^2 * (W_hat_(:,k)'*W_hat_(:,k));
                end
    
                int_GU     = trapz(t_win, GU, 2);
                int_GU_hat = trapz(t_win, GU_hat, 2);
                int_HW     = trapz(t_win, HW, 2);
                int_HW_hat = trapz(t_win, HW_hat, 2);
                int_cost   = trapz(t_win, cost_int);
    
                % 3. 误差项
                delta_hat = NN.w_c' * delta_theta ...
                    + NN.w_c' * (int_GU - int_GU_hat) ...
                    + NN.w_c' * (int_HW - int_HW_hat) ...
                    + int_cost;

                gradsum = gradsum + NN.Theta_c(S(:,1)) * delta_hat';
            end
            gradsum = gradsum / batch_size;
            NN.w_c = NN.w_c - NN.eta * gradsum;
        end

        % 8. 状态更新
        param.s_ctrl = s_next;

        % 9. 事件误差更新
        param.ej = param.s_hat - param.s_ctrl;  iter = iter + 1;

        % 输出1维控制信号，直接将标量复制到4个分量（实际可根据需要调整）
        sys(1) =  real(u(1));
        sys(2) =  real(u(2));
        sys(3) =  real(u(3));
        sys(4) =  real(u(4)); 

% =================== 辅助函数 ===================
function s = barrier(x, c, d)
    s = log((d./c).*((c-x)./(d-x)));


function grad = numerical_gradient(f, s)
    eps = 1e-5;
    grad = zeros(length(s),1);
    for i=1:length(s)
        s1 = s; s2 = s;
        s1(i) = s1(i)+eps;
        s2(i) = s2(i)-eps;
        grad(i,:) = sum((f(s1)-f(s2))/(2*eps))';
    end






