function [sys,x0,str,ts]=Off_Policy_Iteration_ADET_Steering(t,x,u,flag)
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
    global param det
    
    % ==== Barrier ====
    param.c = [-30; -10; -10];
    param.d = [ 30;  10;  10];
    
    % ==== RL parameters ====
    param.n_state = 3;  
    param.n_action = 1; 
    param.n_basis = 30;
    param.gamma = 1;
    param.Q = 1e-2* diag([1e1 1e1 1e1]); % 1e-2 1e2
    param.R = 1e0;  % 1e3
    param.x_latet = param.n_state;
    param.s = zeros(param.n_state,1);
    ka1up   =  1e2; ka1down = -1e2;
    param.ka1 = (ka1up-(ka1down))/2; 
    param.ka2 = (ka1up+(ka1down))/2; 
    
    % ==== Event Trigger ====
    det.eta1 = 0.95; det.Lu = 1e0; det.b_g = 1e0; det.delta1 = 0.7; det.alpha = 0.1; det.alpha =0.6;
    
    % Define the system dimension
    sizes = simsizes;
    sizes.NumContStates  = 0;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 5;
    sizes.NumInputs      = 4;
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 0;
    sys=simsizes(sizes);
    x0=[]; str=[]; ts  = [];
 
function sys=mdlOutputs(t, ~, newExp)
    global param det
    persistent buffer hat NN iter Iter
    if isempty(iter)
        iter = 1;
        % ==== NN initialization ====
        NN.disLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.visLength = linspace(-1e1, 1e1, param.n_basis/3);
        NN.cisLength = linspace(-1e2, 1e2, param.n_basis/3);
        NN.Theta_c = @(s) [tanh(NN.disLength * s(1)), tanh(NN.visLength * s(2)), tanh(NN.visLength * s(3))]';
        NN.w_c = [0.5377;1.8339;-2.2588;0.8622;0.3188;-1.3077;-0.4336;0.3426;3.5784;2.7694;-1.3499;3.0349;0.7254;
        -0.0631;0.7147;-0.2050;-0.1241;1.4897;1.4090;1.4172;0.6715;-1.2075;0.7172;1.6302;0.4889;1.0347;
         0.7269;-0.3034;0.2939;-0.7873];
        NN.eta = 0.0001; 
        NN.buffer_size = 200; 
        NN.window_size = 10;
        NN.update_interval = 5;

        % ==== Control initialization ====
        hat.u_hat = 0; 
        hat.w_hat = 0; 
        
        % ==== Experience replay cache ====
        buffer.buffer = {};

        % ==== Trajectory register ====
        buffer.S_traj = zeros(param.n_state, NN.window_size);
        buffer.U_traj = zeros(param.n_action, NN.window_size);
        buffer.U_hat_traj = zeros(param.n_action, NN.window_size);
        buffer.W_traj = zeros(param.n_action, NN.window_size);
        buffer.W_hat_traj = zeros(param.n_action, NN.window_size);
        buffer.t_traj = zeros(1, NN.window_size);

        % ==== Event-triggered variable ====
        Iter.s_ctrl = param.s;         
        Iter.s_hat = Iter.s_ctrl;      
        Iter.ej = zeros(param.n_state,1);  
    end

        v = newExp(4); % ADET
        
        % Triggering Threshold
        eps_s = 1e-1;
        ebar = norm(Iter.ej) / (norm(Iter.s_ctrl) + eps_s);
        ebar_ref = 1e1; alpha_ref = 1e-1;
        kappa_e = atanh(1 - alpha_ref) / ebar_ref;
        alpha1 = 1 - tanh(kappa_e * ebar);
        eps_a = 1e-6;
        eta1 = min(max(alpha1, eps_a), 1);
        
        % ========== Event trigger judgment ==========
        eT1  = sqrt(eta1^2 * min(eig(param.Q)) * norm(Iter.s_ctrl)^2 / (det.Lu^2 * det.b_g^2) ...
                 + v / (det.delta1 * det.Lu^2 * det.b_g^2));
        
        if norm(Iter.ej) > eT1
            Iter.s_hat = Iter.s_ctrl; 
            Iter.ej = zeros(3, 1);
        end
      
        % 2. NN Approximation V = NN.Theta_c(Iter.s_hat)' * NN.w_c;
        gradV = numerical_gradient(@(ss) NN.Theta_c(ss)'*NN.w_c, Iter.s_hat);
        
        % 3. Control law
        B = [1;1;1]; 
        u = -param.ka1 * tanh(0.5/param.ka1 * B'*gradV) + param.ka2 +25; 
        w = 0;
        
        % 4. Estimation strategy
        hat.u_hat = u;
        hat.w_hat = w;
        s_next = barrier(newExp(1:3), param.c, param.d);
        
        % 5. Record the trajectory
        idx = mod(iter-1, NN.window_size) + 1;
        buffer.S_traj(:,idx) = Iter.s_ctrl;
        buffer.U_traj(:,idx) = u;
        buffer.U_hat_traj(:,idx) = hat.u_hat;
        buffer.W_traj(:,idx) = w;
        buffer.W_hat_traj(:,idx) = hat.w_hat;
        buffer.t_traj(idx) = t;

        % 6. Store in the cache when the window is full
        if idx == NN.window_size
            buffer.buffer = [buffer.buffer; {buffer.S_traj, buffer.U_traj, buffer.U_hat_traj, buffer.W_traj, buffer.W_hat_traj, buffer.t_traj}];
            if size(buffer.buffer,1) > NN.buffer_size
                buffer.buffer(1,:) = [];
            end
        end

        % 7. Strict integral method
        if mod(iter, NN.update_interval)==0 && size(buffer.buffer,1) >  NN.update_interval
            batch_size = min(5, size(buffer.buffer,1));
            idxs = randperm(size(buffer.buffer,1), batch_size);
            gradsum = zeros(size(NN.w_c));
            for j = 1:batch_size
                S = buffer.buffer{idxs(j), 1};
                U_ = buffer.buffer{idxs(j), 2};
                U_hat_ = buffer.buffer{idxs(j), 3};
                W_ = buffer.buffer{idxs(j), 4};
                W_hat_ = buffer.buffer{idxs(j), 5};
                t_win = buffer.buffer{idxs(j), 6};
    
                % 1. delta_theta
                delta_theta = NN.Theta_c(S(:,end)) - NN.Theta_c(S(:,1));
    
                % 2. All Int
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
    
                % 3. Error items
                delta_hat = NN.w_c' * delta_theta ...
                    + NN.w_c' * (int_GU - int_GU_hat) ...
                    + NN.w_c' * (int_HW - int_HW_hat) ...
                    + int_cost;
                gradsum = gradsum + NN.Theta_c(S(:,1)) * delta_hat;
            end
            gradsum = gradsum / batch_size;
            % NN.w_c_new = NN.w_c - NN.eta * gradsum;
            NN.w_c = NN.w_c - NN.eta * gradsum;
        end
    
        % 8. States update
        Iter.s_ctrl = s_next;

        % 9. Event error update
        Iter.ej = Iter.s_hat - Iter.s_ctrl;  iter = iter + 1;

    sys(1) = real(-u/10);
    sys(2) = real(norm(Iter.ej));  %   param.trigger
    sys(3) = real(norm(s_next)); 
    sys(4) = real(norm(Iter.ej)  - eT1);
    sys(5) = real(eT1);

% =================== Auxiliary function ===================
function s = barrier(x, c, d)
    s = log((d./c).*((c-x)./(d-x)));


function grad = numerical_gradient(f, s)
    eps = 1e-5;
    grad = zeros(length(s),1);
    for i=1:length(s)
        s1 = s; s2 = s;
        s1(i) = s1(i)+eps;
        s2(i) = s2(i)-eps;
        grad(i) = (f(s1)-f(s2))/(2*eps);
    end






