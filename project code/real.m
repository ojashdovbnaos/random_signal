%% demo_ssm_insaf_nodown_fixed.m
% 正确修正版：No downsampling, 适配 AR(1) 输入, 快速收敛

clear; clc; close all; rng(1);

%% 1) 全局参数
fs      = 8000;       
T       = 5;          
L       = fs * T;      
M       = 512;         
Nsub    = 16;          
Lp      = 256;         
mu0     = 0.3;         
delta   = 1e-6;        
t_bound = 2;           
P       = 2;           
beta    = 0.9;         
SNRdB   = 20;          

%% 2) 生成 AR(1) 输入
w_noise = randn(L,1);
u       = filter(1,[1 -0.9], w_noise);    % 确认是 AR(1)

% 真实回声通道
w_true  = fir1(M-1,0.1)';
clean   = filter(w_true,1,u);
noiseP  = var(clean)/10^(SNRdB/10);
d       = clean + sqrt(noiseP)*randn(L,1);

%% 3) 构建余弦调制滤波器组
proto = fir1(Lp-1,1/Nsub,hamming(Lp))*(2/sqrt(Nsub));
h     = zeros(Nsub, Lp);
for i = 0:(Nsub-1)
  for n = 0:(Lp-1)
    h(i+1,n+1) = 2 * proto(n+1) * ...
      cos((2*i+1)*pi/(4*Nsub)*(2*n-(Lp-1)) + (-1)^i*pi/4);
  end
end

%% 4) 子带分解（无降采样）
u_sb = zeros(L, Nsub);
d_sb = zeros(L, Nsub);
nsb  = zeros(L, Nsub);
noise = sqrt(noiseP)*randn(L,1);
for i = 1:Nsub
  u_sb(:,i) = filter(h(i,:),1,u);
  d_sb(:,i) = filter(h(i,:),1,d);
  nsb(:,i)  = filter(h(i,:),1,noise);
end

%% 5) 子带噪声方差估计 & 阈值 gamma_i
sigma_eta = var(nsb,0,1)';      
gamma_i   = t_bound * sigma_eta;

%% 6) 初始化变量
W_past    = zeros(M, P);          
sigma_eps = zeros(Nsub,1);        
upd_cnt   = zeros(Nsub,1);        
nmsd_list = [];

%% 7) 主循环
step = Nsub;  % 每隔 Nsub 点更新一次（相当于“伪降采样”）
total_it = floor((L-M)/step);
fprintf('Total iterations = %d\n', total_it);

for idx = 1:total_it
  k = M + (idx-1)*step;
  
  wbar = mean(W_past,2);
  dW   = zeros(M,1);
  
  for i = 1:Nsub
    % 取当前 u 子段
    uvec = u_sb(k:-1:k-M+1,i);
    e_i  = d_sb(k,i) - wbar' * uvec;
    
    % 平滑绝对误差
    sigma_eps(i) = beta * sigma_eps(i) + (1-beta)*abs(e_i);
    
    % 满足门限才更新
    if min(abs(e_i), sigma_eps(i)) > gamma_i(i)
      mu_i = 1 - gamma_i(i) / sigma_eps(i);
      dW   = dW + (mu0 * mu_i * e_i * uvec) / (uvec'*uvec + delta);
      upd_cnt(i) = upd_cnt(i) + 1;
    end
  end
  
  w_new    = wbar + dW;
  W_past   = [w_new, W_past(:,1:end-1)];
  
  % 记录 NMSD
  nmsd = 10*log10( norm(w_true - w_new)^2 / (norm(w_true)^2 + eps) );
  nmsd_list(idx) = nmsd;
  
  if mod(idx,200)==0
    fprintf('Iter %d/%d | NMSD = %.2f dB\n', idx, total_it, nmsd);
    drawnow;
  end
end

%% 8) 总结结果
fprintf('Done. Avg update rate = %.2f%%\n', mean(upd_cnt)/total_it*100);

figure;
plot(linspace(0,T,numel(nmsd_list)), nmsd_list, 'LineWidth',1.5);
xlabel('Time (s)'); ylabel('NMSD (dB)');
title(sprintf('SSM-INSAF No Downsampling (M=%d, Nsub=%d)', M, Nsub));
ylim([-50 5]); grid on;
