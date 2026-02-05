function [sys,x0,str,ts]=Off_Policy_Iteration_DET_LR_Test(t,x,u,flag)
    switch flag 
        case 0
            [sys,x0,str,ts]=mdlInitializeSizes;
        case 1
            sys=mdlDerivatives(t,x,u);
        case 3
            sys=mdlOutputs(t,x,u);
        case {2, 4, 9}
            sys = [];
        otherwise
            error(['Unhandled flag = ',num2str(flag)]);
    end
end


function [sys,x0,str,ts]=mdlInitializeSizes
    sizes = simsizes;
    sizes.NumContStates  = 1;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 4;
    sizes.NumInputs      = 5;
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 0;
    sys = simsizes(sizes);
    x0 = 0; 
    str = []; 
    ts = [];
end


function sys=mdlDerivatives(~, x1, u)
    ej  = u(4); 
    s   = u(5); 
    Lu  = 1e0; 
    b_g = 1e0; 
    Q = 1.2e-1 * diag([1e1 1e3 1e1]); 
    
    % Normalized error
    eps_s = 1e-1;
    ebar = ej / (s + eps_s); 
    
    % Adaptive factor
    ebar_ref = 1e1; 
    alpha_ref = 1e-1;
    kappa_e = atanh(1 - alpha_ref) / ebar_ref;
    alpha1 = 1 - tanh(kappa_e * ebar);
    alpha1 = min(max(alpha1, eps), 1);
    sys(1) = -alpha1 * x1(1) ...
             + (1 - alpha1) * min(eig(Q)) * norm(s)^2 ...
             - (Lu^2 * b_g^2 * norm(ej)^2);
end


function sys=mdlOutputs(t, ~, newExp)
    persistent buffer param NN det iter
    persistent policy_old  
    persistent u_behavior w_behavior 
    persistent Trigger_latch

    if isempty(iter)
        iter = 1;
        % ==== Barrier ====
        param.c = [0.1; -30; 30];
        param.d = [110;  30; 30]; 
        
        % ==== RL parameters ====
        param.n_state = 3;  
        param.n_action = 1; 
        param.n_basis = 30;
        param.gamma = 1; 
        param.Q = 1.2e-1 * diag([1e1 1e3 1e1]);
        param.R = 1;
        param.rho = 1;
        param.s = zeros(param.n_state, 1);
        
        % ==== Event Trigger ====
        det.eta1 = 0.95; 
        det.Lu = 1e0; 
        det.b_g = 1e0; 
        det.delta1 = 1e0; 
        det.alpha = 0.6;
            
        % ==== NN initialization ====
        NN.disLength = linspace(-1e2, 1e2, param.n_basis/3);
        NN.visLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.cisLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.Theta_c = @(s) [tanh(NN.disLength * s(1)), ...
                          tanh(NN.visLength * s(2)), ...
                          tanh(NN.cisLength * s(3))]';
        
        NN.w_c = 0.1 * randn(param.n_basis, 1);
        
        % off-policy
        policy_old.w_c = NN.w_c;  
        
        NN.eta = 3e-2; 
        NN.buffer_size = 200;
        NN.window_size = 10;
        NN.update_interval = 5;
        NN.policy_update_interval = 32;
        
        % ==== Control boundary ====
        ka1up = 8e2; 
        ka1down = 0;
        param.ka1 = (ka1up - ka1down) / 2; 
        param.ka2 = (ka1up + ka1down) / 2;
        
        % ==== Control matrix ====
        param.B = [1; 1; 1];
        param.H = [0.1; 0.1; 0.1];
        
        % ==== Experience replay ====
        buffer.data = {};
        
        % ==== Trajectory register ====
        buffer.S_traj = zeros(param.n_state, NN.window_size);
        buffer.U_traj = zeros(param.n_action, NN.window_size);
        buffer.U_old_traj = zeros(param.n_action, NN.window_size);
        buffer.W_traj = zeros(param.n_action, NN.window_size);
        buffer.W_old_traj = zeros(param.n_action, NN.window_size);
        buffer.t_traj = zeros(1, NN.window_size);

        % ==== Event-triggered variable ====
        param.s_ctrl = param.s;
        param.s_hat = param.s_ctrl;
        param.ej = zeros(param.n_state, 1);
        
        % ==== Control initialization ====
        u_behavior = 0;
        w_behavior = 0;
        Trigger_latch = 0;
    end 
    
    % Keeping policy
    Trigger = Trigger_latch;
    v = 0;
    % Triggering Threshold
    eps_s = 1e-1;
    ebar = norm(param.ej) / (norm(param.s_ctrl) + eps_s);
    ebar_ref = 1e1; alpha_ref = 1e-1;
    kappa_e = atanh(1 - alpha_ref) / ebar_ref;
    alpha1 = 1 - tanh(kappa_e * ebar);
    eps_a = 1e-6;
    eta1 = min(max(alpha1, eps_a), 1);
    
    % ========== 1. Event trigger judgment ==========
      eT1  = sqrt(eta1^2 * min(eig(param.Q)) * norm(param.s_ctrl)^2 / (det.Lu^2 * det.b_g^2) ...
             + v / (det.delta1 * det.Lu^2 * det.b_g^2));

    if norm(param.ej) > eT1
        param.s_hat = param.s_ctrl; 
        param.ej = zeros(param.n_state, 1);
        Trigger_latch = 1;
        Trigger = Trigger_latch;
    else
        Trigger_latch = 0;
    end
    
    % ========== 2. Target strategy ==========
    gradV_current = numerical_gradient(@(ss) NN.Theta_c(ss)' * NN.w_c, param.s_hat);
    u_target = -param.ka1 * tanh(0.5/param.ka1 * param.B' * gradV_current) + param.ka2;
    w_target = 1/(2*param.rho^2) * param.H' * gradV_current;
    
    % ========== 3. Off-policy correction ==========
    gradV_old = numerical_gradient(@(ss) NN.Theta_c(ss)' * policy_old.w_c, param.s_hat);
    u_old = -param.ka1 * tanh(0.5/param.ka1 * param.B' * gradV_old) + param.ka2;
    w_old = 1/(2*param.rho^2) * param.H' * gradV_old;
    
    % ========== 4. Actor policy ==========
    exploration_noise = 0.1 * sin(2*pi*t) * exp(-0.01*t);
    u_behavior = u_target + exploration_noise;
    u_behavior = max(min(u_behavior, param.ka1 + param.ka2), -param.ka1 + param.ka2);  % 饱和
    w_behavior = 0.05 * randn(1);  % Random disturbances
    
    % ========== 5. State Update ==========
    s_next = barrier(newExp(1:3), param.c, param.d);
    
    % ========== 6. Trajectory record ==========
    idx = mod(iter-1, NN.window_size) + 1;
    buffer.S_traj(:, idx) = param.s_ctrl;
    buffer.U_traj(:, idx) = u_behavior;    
    buffer.U_old_traj(:, idx) = u_old;    
    buffer.W_traj(:, idx) = w_behavior;   
    buffer.W_old_traj(:, idx) = w_old;
    buffer.t_traj(idx) = t;

    % ========== 7. Store in the cache when the window is full ==========
    if idx == NN.window_size
        buffer.data = [buffer.data; {buffer.S_traj, buffer.U_traj, buffer.U_old_traj, ...
                                     buffer.W_traj, buffer.W_old_traj, buffer.t_traj}];
        if size(buffer.data, 1) > NN.buffer_size
            buffer.data(1, :) = [];
        end
    end

    % ========== 8. Off-Policy weight update ==========
    if mod(iter, NN.update_interval) == 0 && size(buffer.data, 1) >= 10
        batch_size = min(5, size(buffer.data, 1));
        idxs = randperm(size(buffer.data, 1), batch_size);
        gradsum = zeros(size(NN.w_c));
        
        for j = 1:batch_size
            S = buffer.data{idxs(j), 1};
            U_behav = buffer.data{idxs(j), 2};  
            U_old = buffer.data{idxs(j), 3};   
            W_behav = buffer.data{idxs(j), 4}; 
            W_old = buffer.data{idxs(j), 5};  
            t_win = buffer.data{idxs(j), 6};
            
            % --- The first item: Θ(s(t+δt)) - Θ(s(t)) ---
            delta_theta = NN.Theta_c(S(:, end)) - NN.Theta_c(S(:, 1));
            
            % --- Calculation of integral terms ---
            n_win = size(S, 2);
            
            % Off-policy correction: ∫ ∇Θ^T * G * (u - u_old) dτ
            int_GU_diff = zeros(param.n_basis, 1);
            % Off-policy correction: ∫ ∇Θ^T * H * (w - w_old) dτ
            int_HW_diff = zeros(param.n_basis, 1);
            % Cost integral: ∫ (s^T Q s + S(u_old) - ρ² w_old^T w_old) dτ
            cost_integrand = zeros(1, n_win);
            
            for k = 1:n_win
                gradTheta_k = numerical_gradient_basis(NN.Theta_c, S(:, k));
                
                % G(s) * (u - u_old)
                G_times_u_diff = param.B * (U_behav(:, k) - U_old(:, k));
                int_GU_diff = int_GU_diff + gradTheta_k * G_times_u_diff;
                
                % H(s) * (w - w_old)
                H_times_w_diff = param.H * (W_behav(:, k) - W_old(:, k));
                int_HW_diff = int_HW_diff + gradTheta_k * H_times_w_diff;
                
                % Cost function
                S_u_old = compute_S_function(U_old(:, k), param.ka1, param.ka2);
                cost_integrand(k) = S(:, k)' * param.Q * S(:, k) ...
                                    + S_u_old ...
                                    - param.rho^2 * (W_old(:, k)' * W_old(:, k));
            end
            
            % trapezoidal integral
            dt_avg = mean(diff(t_win));
            int_GU_diff = int_GU_diff * dt_avg;
            int_HW_diff = int_HW_diff * dt_avg;
            int_cost = trapz(t_win, cost_integrand);
            
            % --- Construct matrix A and vector B ---
            A_i = [delta_theta; 
                   int_GU_diff + int_HW_diff];
            B_i = int_cost;
            
            % --- Calculate the residual δ_hat ---
            w_extended = [NN.w_c; NN.w_c];
            delta_hat = w_extended(1:param.n_basis)' * delta_theta ...
                        + NN.w_c' * (int_GU_diff + int_HW_diff) ...
                        + B_i;
            
            % --- Normalized gradient ---
            A_bar = A_i / (A_i' * A_i + 1);
            gradsum = gradsum + A_bar(1:param.n_basis) * delta_hat;
        end
        
        gradsum = gradsum / batch_size;
        NN.w_c = NN.w_c - NN.eta * gradsum;
    end
    
    % ========== 9. Strategy iteration and update ==========
    if mod(iter, NN.policy_update_interval) == 0
        policy_old.w_c = NN.w_c;
    end

    % ========== 10. Status and error update ==========
    param.s_ctrl = s_next;
    param.ej = param.s_hat - param.s_ctrl;
    iter = iter + 1;

    % ========== Output ==========
    sys(1) = real(u_behavior);
    sys(2) = real(norm(param.ej));
    sys(3) = real(norm(param.s_ctrl));
    sys(4) = Trigger;
end


% =================== Auxiliary function ===================

function s = barrier(x, c, d)
    s = zeros(size(x));
    for i = 1:length(x)
        if x(i) <= c(i) || x(i) >= d(i)
            x(i) = max(min(x(i), d(i) - 1e-6), c(i) + 1e-6);
        end
        s(i) = log(abs(d(i)/c(i)) * abs((c(i) - x(i)) / (d(i) - x(i))));
    end
end


function grad = numerical_gradient(f, s)
    eps_val = 1e-5;
    grad = zeros(length(s), 1);
    for i = 1:length(s)
        s1 = s; s2 = s;
        s1(i) = s1(i) + eps_val;
        s2(i) = s2(i) - eps_val;
        grad(i) = (f(s1) - f(s2)) / (2 * eps_val);
    end
end


function gradTheta = numerical_gradient_basis(Theta_func, s)
    eps_val = 1e-5;
    n_basis = length(Theta_func(s));
    n_state = length(s);
    gradTheta = zeros(n_basis, n_state);
    
    for i = 1:n_state
        s1 = s; s2 = s;
        s1(i) = s1(i) + eps_val;
        s2(i) = s2(i) - eps_val;
        gradTheta(:, i) = (Theta_func(s1) - Theta_func(s2)) / (2 * eps_val);
    end
end


function S_val = compute_S_function(u, ka1, ka2)
    S_val = 0;
    for l = 1:length(u)
        u_normalized = (u(l) - ka2) / ka1;
        u_normalized = max(min(u_normalized, 0.999), -0.999);
        
        if abs(u_normalized) < 1e-6
            S_val = S_val + 0;
        else
            S_val = S_val + 2 * ka1 * (u(l) * atanh(u_normalized) ...
                    + ka1/2 * log(1 - u_normalized^2));
        end
    end
end