%% LMS vs NLMS + NMSD 演示
close all; clearvars; rng(1,'twister');

fs = 48000; t = 2; L = t*fs; tv = (1:L)/fs;
N = 64;      % 滤波器长度
mu = 0.1;    % 步长 (0<mu<2)
delta = 1e-3; % NLMS 正则化常数

% 生成信号
x = 0.1*randn(1,L);
b = fir1(N-1,0.1);

% 1) 干净回声
cleanOut = filter(b,1,x);

% 2) 按 20dB 加噪
SNR = 20;
noisePower = var(cleanOut)/10^(SNR/10);
d = cleanOut + sqrt(noisePower)*randn(size(cleanOut));

% 3) 然后再做 LMS/NLMS 更新，并计算 NMSD

% 预分配
h1 = zeros(N,1);              % LMS 权值
h2 = zeros(N,1);              % NLMS 权值
e1 = zeros(1,L); e2 = zeros(1,L);
H1 = zeros(N, L - N + 1);     % 存储每次更新后的 h1
H2 = zeros(N, L - N + 1);     % 存储 h2

idx = 0;
for k = N:L
    idx = idx + 1;
    x1 = x(k:-1:k-N+1)';      % 当前输入向量

    % --- LMS 更新 ---
    y1 = h1' * x1;
    e1(k) = d(k) - y1;
    h1 = h1 + mu * e1(k) * x1;
    H1(:,idx) = h1;

    % --- 正规化 LMS (NLMS) 更新 ---
    y2 = h2' * x1;
    e2(k) = d(k) - y2;
    norm_x1 = delta + (x1'*x1);      % 正确的归一化因子
    h2 = h2 + (mu * e2(k) / norm_x1) * x1;
    H2(:,idx) = h2;
end

% 计算 NMSD
norm_b2 = norm(b)^2 + eps;
nmsd1 = 10*log10( sum((H1 - b(:)).^2,1) / norm_b2 );
nmsd2 = 10*log10( sum((H2 - b(:)).^2,1) / norm_b2 );

% 绘图
%% —— 在前面计算好 nmsd1, nmsd2 之后，改用迭代次数作图 ——%%
iters = 1 : size(H1,2);  % H1,H2 的列数就是迭代次数 (从 k=N 到 L)

figure;
plot(iters, nmsd1, 'b','LineWidth',1.2); hold on;
plot(iters, nmsd2, 'r','LineWidth',1.2);
xlabel('迭代次数');
ylabel('NMSD (dB)');
title('LMS vs. NLMS 的 NMSD 收敛曲线');
legend('LMS','NLMS','Location','northeast');
grid on;
ylim([-50, 10]);  % 根据需要调整
