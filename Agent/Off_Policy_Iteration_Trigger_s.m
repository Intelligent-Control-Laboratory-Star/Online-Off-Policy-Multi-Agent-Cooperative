function [sys,x0,str,ts]=Off_Policy_Iteration_Trigger_s(t,x,u,flag)
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

function [sys,x0,str,ts]=mdlInitializeSizes
    % 定义系统维度
    sizes = simsizes;
    sizes.NumContStates  = 1;
    sizes.NumDiscStates  = 0;
    sizes.NumOutputs     = 1;
    sizes.NumInputs      = 2;
    sizes.DirFeedthrough = 1;
    sizes.NumSampleTimes = 0;
    sys=simsizes(sizes);
    x0=0; str=[]; ts  = [];

function sys=mdlDerivatives(t,x1,u)
    ej = u(1); s = u(2);
    Lu = 1e0;  
    b_g = 1e0; 
    Q = 1.2e-1 * diag([1e1 1e3 1e1]); 
    
    % 归一化误差
    eps_s = 1e-1;
    ebar = ej / (s + eps_s); 
    
    % 自适应因子
    ebar_ref = 1e1; 
    alpha_ref = 1e-1;
    kappa_e = atanh(1 - alpha_ref) / ebar_ref;
    alpha1 = 1 - tanh(kappa_e * ebar);
    alpha1 = min(max(alpha1, eps), 1);
    
    sys(1) = -alpha1 * x1(1) ...
             + (1 - alpha1) * min(eig(Q)) * norm(s)^2 ...
             - (Lu^2 * b_g^2 * norm(ej)^2);
    % sys(1)=0;
 
function sys=mdlOutputs(t, x1, newExp)
    sys(1) = x1;




