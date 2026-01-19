clear; clc; close all;

mu0_list = [0.1, 0.3, 0.5];  % 不同的全局步长
colors = {'-b','-r','-g'};   % 曲线颜色

%% 固定参数
fs = 8000; T = 5; L = fs*T; M = 512; Nsub = 8; Lp = 256;
delta = 1e-6; t_bound = 2; P = 2; beta = 0.9; SNRdB = 20;
step = 4;
total_it = floor((L-M)/step);

%% 信号生成
w_noise = randn(L,1);
u = filter(1,[1 -0.9], w_noise);
w_true = fir1(M-1, 0.1)';
clean = filter(w_true, 1, u);
noiseP = var(clean)/10^(SNRdB/10);
d = clean + sqrt(noiseP)*randn(L,1);

% 余弦调制滤波器组
proto = fir1(Lp-1,1/Nsub,hamming(Lp))*(2/sqrt(Nsub));
h = zeros(Nsub, Lp);
for i = 0:Nsub-1
    for n = 0:Lp-1
        h(i+1,n+1) = 2 * proto(n+1) * cos((2*i+1)*pi/(4*Nsub)*(2*n-(Lp-1)) + (-1)^i*pi/4);
    end
end

% 子带分解
u_sb = zeros(L, Nsub); d_sb = zeros(L, Nsub); nsb = zeros(L, Nsub);
noise = sqrt(noiseP)*randn(L,1);
for i = 1:Nsub
    u_sb(:,i) = filter(h(i,:),1,u);
    d_sb(:,i) = filter(h(i,:),1,d);
    nsb(:,i)  = filter(h(i,:),1,noise);
end
sigma_eta = var(nsb,0,1)';
gamma_i = t_bound * sqrt(sigma_eta);

%% 结果记录
nmsd_results = zeros(numel(mu0_list), total_it);

for m = 1:numel(mu0_list)
    mu0 = mu0_list(m);
    W_past = zeros(M, P);
    sigma_eps = zeros(Nsub,1);
    
    for idx = 1:total_it
        k = M + (idx-1)*step;
        wbar = mean(W_past,2);
        dW = zeros(M,1);
        
        for i = 1:Nsub
            uvec = u_sb(k:-1:k-M+1,i);
            e_i = d_sb(k,i) - wbar' * uvec;

            sigma_eps(i) = beta*sigma_eps(i) + (1-beta)*abs(e_i);
            if min(abs(e_i), sigma_eps(i)) > gamma_i(i)
                mu_i = 1 - gamma_i(i)/max(sigma_eps(i),1e-12);
                dW = dW + (mu0 * mu_i * e_i * uvec)/(uvec'*uvec + delta);
            end
        end
        w_new = wbar + dW;
        W_past = [w_new, W_past(:,1:end-1)];
        nmsd_results(m,idx) = 10*log10(norm(w_true - w_new)^2 / (norm(w_true)^2 + eps));
    end
end

%% 画图
iters = 1:total_it;
figure;
hold on;
for m = 1:numel(mu0_list)
    plot(iters, nmsd_results(m,:), colors{m}, 'LineWidth', 1.5);
end
xlim([0 5000]);
legend('\mu_0 = 0.1', '\mu_0 = 0.3', '\mu_0 = 0.5');
xlabel('Iteration'); ylabel('NMSD (dB)');
title('Effect of \mu_0 on SSM-INSAF Performance');
grid on; ylim([-50 5]);
